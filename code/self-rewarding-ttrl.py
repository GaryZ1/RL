"""Label-free RL for LLMs: TTRL (majority-vote pseudo-labels), INTUITOR-style self-certainty reward,
entropy-minimisation reward, and Absolute-Zero-style proposer 'learnability' reward.

(a) Reward functions on tensors / lists:
    - majority_vote_rewards(answers, G)          : TTRL
    - self_certainty(logits, mask)                : INTUITOR, mean_t KL(Uniform || p_t)
    - negative_entropy(logits, mask)              : entropy-minimisation reward
    - azr_proposer_reward(solve_rate)             : Absolute Zero proposer reward
(b) Toy: a tabular "pretrained" model answers each prompt by first picking a reasoning path
    (4 options) and then an answer (8 options). Some paths are confident & right, one is
    diffuse, one is CONFIDENTLY WRONG; on 25% of prompts the model holds a misconception.
    We train with ground-truth reward vs. each label-free reward and track TRUE accuracy.

Run:  python3 code/self-rewarding-ttrl.py     (CPU, ~10-30 s)
"""
import math
from collections import Counter

import torch
import torch.nn.functional as F

torch.set_num_threads(1)


# ----------------------------------------------------------------------------------------
# (a) Label-free rewards
# ----------------------------------------------------------------------------------------
def majority_vote_rewards(answers, G):
    """TTRL: for each prompt, the most frequent answer among its G samples is the pseudo-label;
    reward 1 for samples agreeing with it. answers: list (len n*G) of hashable answers."""
    rewards, labels = [], []
    for g in range(0, len(answers), G):
        grp = answers[g:g + G]
        maj = Counter(grp).most_common(1)[0][0]
        labels.append(maj)
        rewards += [1.0 if a == maj else 0.0 for a in grp]
    return torch.tensor(rewards), labels


def self_certainty(logits, mask):
    """INTUITOR (Zhao et al. 2025): average over response tokens of KL(U || p_t)
    = -log V - (1/V) sum_j log p_t(j). Higher = more confident. logits [B,T,V], mask [B,T]."""
    V = logits.shape[-1]
    logp = F.log_softmax(logits, -1)
    kl_u = -math.log(V) - logp.mean(-1)
    return (kl_u * mask).sum(1) / mask.sum(1).clamp(min=1)


def negative_entropy(logits, mask):
    """Entropy-minimisation reward: -mean_t H(p_t)."""
    logp = F.log_softmax(logits, -1)
    H = -(logp.exp() * logp).sum(-1)
    return -(H * mask).sum(1) / mask.sum(1).clamp(min=1)


def azr_proposer_reward(solve_rate: torch.Tensor) -> torch.Tensor:
    """Absolute Zero proposer reward: tasks that are always or never solved teach nothing;
    r = 1 - mean_solve_rate if 0 < rate < 1 else 0."""
    return torch.where((solve_rate > 0) & (solve_rate < 1), 1 - solve_rate, torch.zeros_like(solve_rate))


# ----------------------------------------------------------------------------------------
# (b) Toy "pretrained" model: path -> answer
# ----------------------------------------------------------------------------------------
N_PROMPTS, N_PATHS, N_ANS = 40, 4, 8


def make_model(seed=0):
    g = torch.Generator().manual_seed(seed)
    truth = torch.randint(0, N_ANS, (N_PROMPTS,), generator=g)
    wrong = (truth + torch.randint(1, N_ANS, (N_PROMPTS,), generator=g)) % N_ANS
    misconception = torch.rand(N_PROMPTS, generator=g) < 0.25
    path_logits = torch.zeros(N_PROMPTS, N_PATHS)
    ans_logits = 0.3 * torch.randn(N_PROMPTS, N_PATHS, N_ANS, generator=g)
    idx = torch.arange(N_PROMPTS)
    ans_logits[idx, 0, truth] += torch.where(misconception, 1.0, 3.0)     # path 0: careful, right
    ans_logits[idx, 1, truth] += 1.2                                      # path 1: weakly right
    #                                                                       path 2: diffuse guess
    ans_logits[idx, 3, wrong] += torch.where(misconception, 3.5, 2.0)     # path 3: confidently WRONG
    return truth, misconception, path_logits.requires_grad_(), ans_logits.requires_grad_()


def sample(path_logits, ans_logits, prompts):
    paths = torch.distributions.Categorical(logits=path_logits[prompts]).sample()
    answers = torch.distributions.Categorical(logits=ans_logits[prompts, paths]).sample()
    return paths, answers


def response_logits(path_logits, ans_logits, prompts, paths):
    """Per-token logits of the 2-token response [path, answer], padded to a common vocab so we
    can compute self-certainty / entropy with a mask (-inf padding gives zero prob)."""
    V = max(N_PATHS, N_ANS)
    pad = lambda x: F.pad(x, (0, V - x.shape[-1]), value=-30.0)
    return torch.stack([pad(path_logits[prompts]), pad(ans_logits[prompts, paths])], 1)


def train(reward_type, steps=150, n=20, G=16, lr=5.0, seed=0):
    truth, misc, path_logits, ans_logits = make_model()
    torch.manual_seed(seed)
    opt = torch.optim.SGD([path_logits, ans_logits], lr=lr)
    hist = []
    for step in range(steps + 1):
        prompts = torch.randint(0, N_PROMPTS, (n,)).repeat_interleave(G)
        paths, answers = sample(path_logits.detach(), ans_logits.detach(), prompts)
        if reward_type == "ground truth":
            r = (answers == truth[prompts]).float()
        elif reward_type == "TTRL majority vote":
            r, _ = majority_vote_rewards(answers.tolist(), G)
        else:
            with torch.no_grad():
                lg = response_logits(path_logits, ans_logits, prompts, paths)
                m = torch.ones(lg.shape[:2])
                r = self_certainty(lg, m) if reward_type == "self-certainty" else negative_entropy(lg, m)
        rg = r.view(-1, G)
        adv = ((rg - rg.mean(1, keepdim=True)) / (rg.std(1, keepdim=True) + 1e-6)).view(-1)
        logp = (F.log_softmax(path_logits[prompts], -1).gather(1, paths[:, None]).squeeze(1) +
                F.log_softmax(ans_logits[prompts, paths], -1).gather(1, answers[:, None]).squeeze(1))
        loss = -(logp * adv).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0:
            hist.append(evaluate(truth, misc, path_logits.detach(), ans_logits.detach()))
    return hist


@torch.no_grad()
def evaluate(truth, misc, path_logits, ans_logits, k=16):
    """True accuracy of single samples, overall and on misconception prompts; maj@k accuracy."""
    prompts = torch.arange(N_PROMPTS).repeat_interleave(k)
    _, ans = sample(path_logits, ans_logits, prompts)
    correct = (ans == truth[prompts]).float().view(N_PROMPTS, k)
    _, labels = majority_vote_rewards(ans.tolist(), k)
    maj_acc = (torch.tensor(labels) == truth).float().mean().item()
    return correct.mean().item(), correct[misc].mean().item(), maj_acc


if __name__ == "__main__":
    r, labels = majority_vote_rewards([3, 3, 5, 3, 1, 1, 2, 2], G=4)
    print("majority-vote rewards:", r.tolist(), "pseudo-labels:", labels)
    peaked = torch.tensor([[[5.0, 0, 0, 0]]])
    flat = torch.zeros(1, 1, 4)
    m = torch.ones(1, 1)
    print(f"self-certainty peaked {self_certainty(peaked, m).item():.3f} vs flat "
          f"{self_certainty(flat, m).item():.3f}; neg-entropy peaked "
          f"{negative_entropy(peaked, m).item():.3f} vs flat {negative_entropy(flat, m).item():.3f}")
    print("AZR proposer reward for solve rates 0, .25, .5, 1:",
          azr_proposer_reward(torch.tensor([0.0, 0.25, 0.5, 1.0])).tolist(), "\n")

    print(f"{'reward':20s} " + " | ".join(f"step {s:3d}: acc / acc(misconc.) / maj@16" for s in (0, 50, 100, 150)))
    for rt in ["ground truth", "TTRL majority vote", "self-certainty", "negative entropy"]:
        h = train(rt)
        print(f"{rt:20s} " + " | ".join(f"      {a:.3f} / {b:.3f} / {c:.3f}      " for a, b, c in h))
