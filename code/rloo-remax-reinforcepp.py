"""Critic-free REINFORCE-style estimators for LLM RL: RLOO, ReMax, REINFORCE++ (+ baseline variant).

(a) Advantage estimators on tensors (sequence rewards [B], per-token KL [B,T], masks [B,T]).
(b) A toy demo that trains the same tiny GRU policy with each estimator on a verifiable task
    ("emit 5 digits in 0..3 whose sum equals the target") and prints learning curves.

Run:  python3 code/rloo-remax-reinforcepp.py     (CPU, ~1 min)
"""
import copy

import torch
import torch.nn as nn

torch.set_num_threads(1)


# ----------------------------------------------------------------------------------------
# (a) Advantage estimators
# ----------------------------------------------------------------------------------------
def rloo_advantages(rewards: torch.Tensor, k: int) -> torch.Tensor:
    """RLOO (Kool et al. 2019; Ahmadian et al. 2024): baseline for sample i = mean reward of
    the OTHER k-1 samples of the same prompt.  A_i = r_i - (sum_j r_j - r_i)/(k-1)
    which equals k/(k-1) * (r_i - mean).  rewards: [n_prompts*k], groups contiguous."""
    r = rewards.view(-1, k)
    baseline = (r.sum(1, keepdim=True) - r) / (k - 1)
    return (r - baseline).view(-1)


def remax_advantages(rewards: torch.Tensor, greedy_rewards: torch.Tensor) -> torch.Tensor:
    """ReMax (Li et al. 2023): baseline = reward of the GREEDY (argmax) completion for the same
    prompt.  rewards: [B]; greedy_rewards: [B] (repeat the per-prompt greedy reward)."""
    return rewards - greedy_rewards


def reinforce_pp_advantages(rewards: torch.Tensor, kl: torch.Tensor, mask: torch.Tensor,
                            beta: float = 0.01, gamma: float = 1.0,
                            group_size: int = None, eps: float = 1e-8) -> torch.Tensor:
    """REINFORCE++ (Hu et al. 2025, OpenRLHF).
    Token reward: r_t = -beta * KL_t, plus the scalar outcome reward on the LAST valid token.
    Return-to-go G_t = sum_{s>=t} gamma^(s-t) r_s, then *global batch* normalisation
    (whitening over every valid token in the batch) instead of a per-prompt baseline.
    If group_size is given -> 'REINFORCE++-baseline': subtract the group mean of the outcome
    reward first, then whiten globally (this is the variant suggested for RLVR-style tasks)."""
    B, T = mask.shape
    outcome = rewards.clone()
    if group_size is not None:
        g = outcome.view(-1, group_size)
        outcome = (g - g.mean(1, keepdim=True)).view(-1)
    token_r = -beta * kl * mask
    last = (mask.cumsum(1) == mask.sum(1, keepdim=True)) & (mask > 0)    # last valid token
    token_r = token_r + last.float() * outcome.unsqueeze(1)
    ret = torch.zeros_like(token_r)
    running = torch.zeros(B)
    for t in reversed(range(T)):
        running = token_r[:, t] + gamma * running * mask[:, t]
        ret[:, t] = running
    valid = mask.bool()
    mean, std = ret[valid].mean(), ret[valid].std()
    return ((ret - mean) / (std + eps)) * mask                    # [B, T]


def pg_loss(logp, adv, mask, logp_old=None, clip_eps=0.2):
    """Token-level REINFORCE loss; with logp_old gives the PPO-clipped version (as OpenRLHF does)."""
    if adv.dim() == 1:
        adv = adv.unsqueeze(1).expand_as(logp)
    if logp_old is None:
        per_tok = -logp * adv
    else:
        ratio = torch.exp(logp - logp_old)
        per_tok = -torch.min(ratio * adv, ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv)
    return (per_tok * mask).sum() / mask.sum()


# ----------------------------------------------------------------------------------------
# (b) Toy policy and task
# ----------------------------------------------------------------------------------------
VOCAB, LENGTH, MAX_T = 4, 5, 15


class TinyGRUPolicy(nn.Module):
    def __init__(self, n_prompts, vocab, hidden=64):
        super().__init__()
        self.V = vocab
        self.prompt_emb = nn.Embedding(n_prompts, hidden)
        self.tok_emb = nn.Embedding(vocab + 1, hidden)
        self.gru = nn.GRUCell(hidden, hidden)
        self.head = nn.Linear(hidden, vocab)

    def forward(self, prompts, tokens):
        B, T = tokens.shape
        h = torch.tanh(self.prompt_emb(prompts))
        inp = torch.full((B,), self.V, dtype=torch.long)
        out = []
        for t in range(T):
            h = self.gru(self.tok_emb(inp), h)
            out.append(self.head(h))
            inp = tokens[:, t]
        return torch.stack(out, 1)

    def token_logp(self, prompts, tokens):
        return torch.log_softmax(self(prompts, tokens), -1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)

    @torch.no_grad()
    def sample(self, prompts, T, greedy=False):
        B = prompts.shape[0]
        h = torch.tanh(self.prompt_emb(prompts))
        inp = torch.full((B,), self.V, dtype=torch.long)
        toks = []
        for _ in range(T):
            h = self.gru(self.tok_emb(inp), h)
            logits = self.head(h)
            inp = logits.argmax(-1) if greedy else torch.distributions.Categorical(logits=logits).sample()
            toks.append(inp)
        return torch.stack(toks, 1)


def reward_fn(prompts, toks):
    # dense-ish verifiable reward: 1 if exact, 0.5 if off by one, else 0
    d = (toks.sum(1) - prompts).abs()
    return (d == 0).float() + 0.5 * (d == 1).float()


def train(method, steps=200, n_prompts=32, k=4, lr=3e-3, seed=0):
    torch.manual_seed(seed)
    policy = TinyGRUPolicy(MAX_T + 1, VOCAB)
    ref = copy.deepcopy(policy).eval()
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    curve = []
    for step in range(steps):
        base = torch.randint(0, MAX_T + 1, (n_prompts,))
        prompts = base.repeat_interleave(k)
        toks = policy.sample(prompts, LENGTH)
        mask = torch.ones_like(toks, dtype=torch.float)
        r = reward_fn(prompts, toks)
        if method == "RLOO":
            adv = rloo_advantages(r, k)
        elif method == "ReMax":
            # one extra greedy rollout per prompt -> the baseline
            g = reward_fn(base, policy.sample(base, LENGTH, greedy=True)).repeat_interleave(k)
            adv = remax_advantages(r, g)
        elif method.startswith("REINFORCE++"):
            with torch.no_grad():
                kl = policy.token_logp(prompts, toks) - ref.token_logp(prompts, toks)  # k1 per token
            adv = reinforce_pp_advantages(r, kl, mask, beta=0.01,
                                          group_size=k if method.endswith("baseline") else None)
        elif method == "REINFORCE (no baseline)":
            adv = r
        logp = policy.token_logp(prompts, toks)
        loss = pg_loss(logp, adv, mask)
        opt.zero_grad()
        loss.backward()
        opt.step()
        curve.append(r.mean().item())
    with torch.no_grad():
        allp = torch.arange(MAX_T + 1)
        exact = (policy.sample(allp, LENGTH, greedy=True).sum(1) == allp).float().mean().item()
    return curve, exact


if __name__ == "__main__":
    r = torch.tensor([1., 0., 0., 1., 0., 0., 0., 0.])
    print("rewards              :", r.tolist())
    print("RLOO advantages (k=4):", [round(x, 3) for x in rloo_advantages(r, 4).tolist()])
    print("ReMax (greedy r=0.5) :", remax_advantages(r, torch.full_like(r, 0.5)).tolist())
    m = torch.tensor([[1., 1., 1., 0.], [1., 1., 1., 1.]])
    print("REINFORCE++ token adv:\n", reinforce_pp_advantages(torch.tensor([1., 0.]),
                                                             torch.zeros(2, 4), m).round(decimals=3))
    print()
    methods = ["REINFORCE (no baseline)", "RLOO", "ReMax", "REINFORCE++", "REINFORCE++-baseline"]
    for meth in methods:
        curves, exacts = [], []
        for seed in range(2):
            c, e = train(meth, seed=seed)
            curves.append(c)
            exacts.append(e)
        first = sum(sum(c[:20]) / 20 for c in curves) / len(curves)
        last = sum(sum(c[-20:]) / 20 for c in curves) / len(curves)
        print(f"{meth:24s} mean reward first20={first:.3f} last20={last:.3f} "
              f"| greedy exact-match over targets={sum(exacts)/len(exacts):.3f}")
