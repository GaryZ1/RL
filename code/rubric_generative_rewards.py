"""
Rubric / generative rewards for non-verifiable tasks + a GRPO-style update.

* Each prompt carries its own RUBRIC: weighted criteria tagged Essential / Important /
  Optional / Pitfall (negative weight), in the style of "Rubrics as Rewards" (RaR).
* Each criterion is checked either by a deterministic VERIFIER (rule) or by a simulated
  LLM JUDGE (noisy yes/no, like asking a judge model "does the response satisfy X?").
* Aggregation:
    explicit  : reward = sum_j w_j * c_j / sum_{w_j>0} w_j        (per-criterion scores)
    implicit  : the whole rubric is given to a judge that returns ONE holistic 1-10 score
                (simulated here as a noisy, nonlinear function of the criteria)
    hybrid    : DeepSeek-V3.2-flavoured -- add a rule-based verifier signal when the prompt
                has a checkable answer, rubric-based (generative) reward everywhere.
* Policy optimization: GRPO-style group-normalized advantages, clipped ratio, k3 KL to ref.

Run: python3 code/rubric_generative_rewards.py    (CPU, ~40 s)
"""
import copy
import torch
import torch.nn as nn

torch.manual_seed(0)
torch.set_num_threads(1)
V, T, H, P = 16, 6, 64, 4
# tokens: KEY[p]=p (key fact), SUPPORT[p]=4+p, MISCONCEPTION[p]=8+p, 12..15 filler
VERIFIABLE_PROMPTS = {0}           # prompt 0 has a checkable "final answer" (last token = KEY)

# ----------------------------------------------------------------------------------
# (a) Rubrics, judges, aggregation
# ----------------------------------------------------------------------------------

def contains(tok):
    return lambda y: (y == tok).any(-1).float()


def make_rubric(p):
    """List of (description, category, weight, checker, checked_by)."""
    return [
        (f"mentions key fact {p}",           "Essential", 5.0, contains(p),     "verifier"),
        (f"gives supporting detail {4 + p}", "Important", 3.0, contains(4 + p), "judge"),
        ("leads with the direct answer",     "Important", 2.0, lambda y: (y[:, 0] == p).float(), "verifier"),
        ("concise: <=2 filler tokens",       "Optional",  1.0, lambda y: ((y >= 12).sum(-1) <= 2).float(), "judge"),
        (f"states misconception {8 + p}",    "Pitfall",  -4.0, contains(8 + p), "judge"),  # penalized if present
    ]


RUBRICS = [make_rubric(p) for p in range(P)]


def judge_criterion(sat, flip=0.1):
    """Simulated LLM judge: answers yes/no about one criterion, wrong w.p. `flip`."""
    wrong = torch.rand_like(sat) < flip
    return torch.where(wrong, 1 - sat, sat)


def criterion_scores(prompts, y, noisy=True):
    """[B, n_criteria] matrix of 0/1 judgements and the weight vector per row."""
    n_c = len(RUBRICS[0])
    C, W = torch.zeros(y.size(0), n_c), torch.zeros(y.size(0), n_c)
    for p in range(P):
        idx = (prompts == p).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        for j, (_, _, w, check, by) in enumerate(RUBRICS[p]):
            sat = check(y[idx])
            if noisy and by == "judge":
                sat = judge_criterion(sat)
            C[idx, j], W[idx, j] = sat, w
    return C, W


def explicit_aggregate(C, W):
    """RaR explicit aggregation: normalized weighted sum (pitfalls subtract)."""
    return (C * W).sum(1) / W.clamp(min=0).sum(1)


def implicit_aggregate(C, W, noise=0.08):
    """RaR implicit aggregation (simulated): a judge reads the full rubric and outputs a
    holistic Likert score 1..10 -- a nonlinear, noisy view of the same criteria."""
    s = explicit_aggregate(C, W)
    likert = torch.round(1 + 9 * torch.sigmoid(6 * (s - 0.5) + noise * 10 * torch.randn_like(s)))
    return (likert - 1) / 9


def verifier_reward(prompts, y):
    """Rule-based check for verifiable prompts: final token must be the key answer."""
    return (y[:, -1] == prompts).float()


def hybrid_reward(prompts, y, rubric_reward, w_verifier=0.5):
    """Verifiable prompts: mix the hard rule-based check with the rubric score
    (verifier-only reward would ignore style/quality criteria); others: rubric only."""
    verifiable = torch.tensor([int(p) in VERIFIABLE_PROMPTS for p in prompts.tolist()])
    mixed = w_verifier * verifier_reward(prompts, y) + (1 - w_verifier) * rubric_reward
    return torch.where(verifiable, mixed, rubric_reward)

# ----------------------------------------------------------------------------------
# (b) GRPO-style loss (as in TRL GRPOTrainer / verl, token-level, with k3 KL)
# ----------------------------------------------------------------------------------

def group_advantages(rewards, group_size, eps=1e-6):
    r = rewards.view(-1, group_size)
    return ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + eps)).view(-1)


def grpo_loss(logp, old_logp, ref_logp, adv, mask, clip_eps=0.2, beta=0.02):
    """logp/old_logp/ref_logp: [B, T] token log-probs; adv: [B] sequence advantages."""
    ratio = torch.exp(logp - old_logp)
    a = adv[:, None]
    pg = -torch.min(ratio * a, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * a)
    log_r = ref_logp - logp
    kl = torch.exp(log_r) - log_r - 1                    # k3 estimator, >= 0
    per_tok = pg + beta * kl
    return ((per_tok * mask).sum(1) / mask.sum(1)).mean()

# ----------------------------------------------------------------------------------
# (c) Tiny policy + training loop
# ----------------------------------------------------------------------------------

class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb, self.gru, self.head = nn.Embedding(V, H), nn.GRU(H, H, batch_first=True), nn.Linear(H, V)

    def forward(self, x):
        return self.head(self.gru(self.emb(x))[0])


def token_logps(m, p, y):
    inp = torch.cat([p[:, None], y[:, :-1]], 1)
    return torch.log_softmax(m(inp), -1).gather(-1, y[..., None]).squeeze(-1)


@torch.no_grad()
def generate(m, p):
    tok, h, out = p[:, None], None, []
    for _ in range(T):
        x, h = m.gru(m.emb(tok), h)
        tok = torch.distributions.Categorical(logits=m.head(x[:, -1])).sample()[:, None]
        out.append(tok)
    return torch.cat(out, 1)


def sft_init():
    """Weak SFT model: mostly filler, sometimes key/support/misconception tokens."""
    m = TinyLM()
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for _ in range(200):
        p = torch.randint(0, P, (128,))
        choices = torch.stack([p, 4 + p, 8 + p] + [torch.full_like(p, 12 + k) for k in range(4)], 1)
        probs = torch.tensor([0.15, 0.15, 0.15, 0.1375, 0.1375, 0.1375, 0.1375])
        y = choices.gather(1, torch.multinomial(probs, 128 * T, replacement=True).view(128, T))
        loss = -token_logps(m, p, y).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return m


@torch.no_grad()
def evaluate(m, n=2000):
    p = torch.randint(0, P, (n,))
    y = generate(m, p)
    C, W = criterion_scores(p, y, noisy=False)          # TRUE criterion satisfaction
    rates = C.mean(0).tolist()
    return explicit_aggregate(C, W).mean().item(), rates, verifier_reward(p[p == 0], y[p == 0]).mean().item()


def train(aggregation, steps=150, n_prompts=16, G=8):
    torch.manual_seed(1)
    policy, ref = sft_init(), None
    ref = copy.deepcopy(policy).eval()
    opt = torch.optim.Adam(policy.parameters(), lr=2e-3)
    for _ in range(steps):
        p = torch.randint(0, P, (n_prompts,)).repeat_interleave(G)
        y = generate(policy, p)
        C, W = criterion_scores(p, y, noisy=True)
        if aggregation == "explicit":
            r = explicit_aggregate(C, W)
        elif aggregation == "implicit":
            r = implicit_aggregate(C, W)
        else:   # hybrid: verifier + rubric on verifiable prompts, rubric elsewhere
            r = hybrid_reward(p, y, explicit_aggregate(C, W))
        adv = group_advantages(r, G)
        with torch.no_grad():
            old, ref_lp = token_logps(policy, p, y), token_logps(ref, p, y)
        for _ in range(2):                              # 2 inner PPO-style epochs
            loss = grpo_loss(token_logps(policy, p, y), old, ref_lp, adv, torch.ones(p.size(0), T))
            opt.zero_grad(); loss.backward(); opt.step()
    return policy


def main():
    names = [c[0] for c in RUBRICS[0]]
    s, rates, v = evaluate(sft_init())
    print(f"{'model':<16}{'true rubric score':>18}{'final-answer ok (p=0)':>23}   per-criterion satisfaction rates")
    print(f"{'SFT':<16}{s:>18.3f}{v:>23.3f}   " + " ".join(f"{x:.2f}" for x in rates))
    for agg in ["explicit", "implicit", "hybrid"]:
        s, rates, v = evaluate(train(agg))
        print(f"{'GRPO+' + agg:<16}{s:>18.3f}{v:>23.3f}   " + " ".join(f"{x:.2f}" for x in rates))
    print("criteria order:", names, "(last one is a pitfall: lower is better)")


if __name__ == "__main__":
    main()
