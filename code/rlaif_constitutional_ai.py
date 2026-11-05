"""
Constitutional AI / RLAIF in miniature.

Stage 0  "helpful-only" model (SFT on helpful demos; it happily emits HARM tokens on harmful requests)
Stage 1  SL-CAI : sample -> AI critique against a principle -> AI revision -> SFT on revisions
Stage 2  RL-CAI : sample response pairs -> AI labeler (a randomly drawn constitutional principle
                  per comparison) gives SOFT preference labels -> train preference model (soft
                  Bradley-Terry) -> RL against the PM with a KL penalty to the SL-CAI model.

Vocabulary (V=10): 0-3 content tokens (the on-topic answer to prompt p is token p),
4-6 filler, 7 = REFUSE (bare, evasive refusal), 8 = EXPLAIN (harmless, engaged explanation),
9 = HARM (dangerous detail). Prompts 0,1 are benign; prompts 2,3 are harmful requests.

Run: python3 code/rlaif_constitutional_ai.py      (CPU, ~10 s)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_num_threads(1)
V, T, P = 10, 6, 4
REFUSE, EXPLAIN, HARM = 7, 8, 9
HARMFUL_PROMPTS = torch.tensor([False, False, True, True])

# ----------------------------------------------------------------------------------
# The "constitution": each principle is a function scoring a response (higher = better).
# In real CAI each principle is a natural-language instruction given to a feedback LLM,
# e.g. "Choose the response that is less harmful, but not evasive". Here we simulate the
# feedback model's judgement with rule functions.
# ----------------------------------------------------------------------------------

def count(y, tok):
    return (y == tok).sum(-1).float()


def p_harmless(p, y):           # "Which response is less harmful?"
    return -2.0 * count(y, HARM)


def p_helpful(p, y):            # "Which response is more helpful / on-topic?"
    on_topic = (y == p[:, None]).sum(-1).float()
    return on_topic + 0.5 * count(y, EXPLAIN) * HARMFUL_PROMPTS[p].float()


def p_non_evasive(p, y):        # "Which response engages thoughtfully instead of refusing?"
    return -1.0 * count(y, REFUSE)


CONSTITUTION = [p_harmless, p_helpful, p_non_evasive]


def ai_label(p, y_a, y_b, temperature=1.0, label_noise=0.1):
    """Simulated AI feedback model. One principle is sampled per comparison (as in CAI).
    Returns SOFT label P(a > b) -- CAI/RLAIF use the labeler's normalized probabilities
    of answering "(A)" vs "(B)" rather than hard 0/1 labels."""
    idx = torch.randint(0, len(CONSTITUTION), (p.size(0),))
    s_a = torch.stack([f(p, y_a) for f in CONSTITUTION], 1).gather(1, idx[:, None]).squeeze(1)
    s_b = torch.stack([f(p, y_b) for f in CONSTITUTION], 1).gather(1, idx[:, None]).squeeze(1)
    prob = torch.sigmoid((s_a - s_b) / temperature)
    return (1 - label_noise) * prob + label_noise * 0.5      # labeler is imperfect


def critique_and_revise(p, y):
    """SL-CAI: critique = find tokens violating principles; revision = rewrite them.
    (A real system prompts the model: 'Identify ways the response is harmful...' then
    'Rewrite the response to remove harmful content'.)"""
    y = y.clone()
    y[y == HARM] = EXPLAIN          # remove harmful details, explain instead
    y[y == REFUSE] = EXPLAIN        # CAI aimed for harmless AND non-evasive
    return y

# ----------------------------------------------------------------------------------
# Models: tabular position-wise softmax policy, linear preference model
# ----------------------------------------------------------------------------------

class TablePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(P, T, V))

    def dist(self, p):
        return torch.distributions.Categorical(logits=self.logits[p])

    def sample(self, p):
        with torch.no_grad():
            return self.dist(p).sample()

    def seq_logp(self, p, y):
        return self.dist(p).log_prob(y).sum(-1)


class PreferenceModel(nn.Module):
    """r(p, y) = w[p] . token_counts(y)  -- prompt-conditioned linear PM."""
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(P, V))

    def forward(self, p, y):
        return (self.w[p] * F.one_hot(y, V).float().sum(1)).sum(-1)


def sft(policy, data_fn, steps=300, lr=0.1):
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    for _ in range(steps):
        p, y = data_fn()
        loss = -policy.seq_logp(p, y).mean()
        opt.zero_grad(); loss.backward(); opt.step()


def helpful_only_demos(n=256):
    p = torch.randint(0, P, (n,))
    on_topic = p[:, None].expand(n, T)
    filler = torch.randint(4, 7, (n, T))
    u = torch.rand(n, T)
    benign = torch.where(u < 0.5, on_topic, filler)
    harmful = torch.where(u < 0.4, torch.full_like(on_topic, HARM),
                          torch.where(u < 0.6, on_topic,
                                      torch.where(u < 0.7, torch.full_like(on_topic, REFUSE), filler)))
    y = torch.where(HARMFUL_PROMPTS[p][:, None], harmful, benign)
    return p, y


@torch.no_grad()
def report(name, policy, n=4000):
    p = torch.randint(0, P, (n,))
    y = policy.sample(p)
    h = HARMFUL_PROMPTS[p]
    harm = (y[h] == HARM).float().mean().item()
    refuse = (y[h] == REFUSE).float().mean().item()
    helpful = (y[~h] == p[~h][:, None]).float().mean().item()
    print(f"{name:<18} harm-token rate {harm:.3f}   refusal rate {refuse:.3f}   "
          f"benign on-topic rate {helpful:.3f}")


def main():
    # Stage 0: helpful-only model
    helpful_only = TablePolicy()
    sft(helpful_only, helpful_only_demos)
    report("helpful-only", helpful_only)

    # Stage 1: SL-CAI (critique -> revision -> SFT on revised responses)
    def revised_data(n=256):
        p = torch.randint(0, P, (n,))
        return p, critique_and_revise(p, helpful_only.sample(p))
    sl_cai = TablePolicy()
    sl_cai.load_state_dict(helpful_only.state_dict())
    sft(sl_cai, revised_data, steps=150, lr=0.05)
    report("SL-CAI", sl_cai)

    # Stage 2a: AI preference labels on pairs sampled from SL-CAI -> train preference model
    n = 8000
    p = torch.randint(0, P, (n,))
    y_a, y_b = sl_cai.sample(p), sl_cai.sample(p)
    soft = ai_label(p, y_a, y_b)
    pm = PreferenceModel()
    opt = torch.optim.Adam(pm.parameters(), lr=0.05)
    for _ in range(300):
        loss = F.binary_cross_entropy_with_logits(pm(p, y_a) - pm(p, y_b), soft)
        opt.zero_grad(); loss.backward(); opt.step()
    print("learned PM weights for harmful prompt 2 (tokens 0..9):",
          [round(v, 2) for v in pm.w[2].tolist()])

    # Stage 2b: RL-CAI -- REINFORCE with leave-one-out baseline, KL penalty to SL-CAI
    policy = TablePolicy()
    policy.load_state_dict(sl_cai.state_dict())
    opt = torch.optim.Adam(policy.parameters(), lr=0.05)
    beta, K = 0.05, 4
    for step in range(200):
        p = torch.randint(0, P, (64,)).repeat_interleave(K)
        y = policy.sample(p)
        logp = policy.seq_logp(p, y)
        with torch.no_grad():
            kl = logp - sl_cai.seq_logp(p, y)                 # per-sequence log-ratio
            r = pm(p, y) - beta * kl                          # KL-shaped reward
            r = r.view(-1, K)
            adv = (r - (r.sum(1, keepdim=True) - r) / (K - 1)).view(-1)   # RLOO baseline
        loss = -(adv * logp).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    report("RL-CAI", policy)


if __name__ == "__main__":
    main()
