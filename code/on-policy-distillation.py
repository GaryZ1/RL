"""On-policy distillation: per-token reverse KL to a teacher on STUDENT samples (Thinking Machines /
Qwen3 style), GKD's generalized JSD, MiniLLM-style reverse KL -- vs. off-policy SFT and plain RL.

(a) Loss functions on tensors:
    - sampled_reverse_kl_pg_loss : per-token reward -(log p_s - log p_t) on the sampled token, used as
      a dense advantage in a policy-gradient loss (what the Tinker / TML recipe does).
    - full_reverse_kl_loss       : exact sum_v p_s (log p_s - log p_t) per position (needs teacher logits
      over the whole vocabulary).
    - generalized_jsd_loss       : GKD's JSD_beta (beta=0 -> forward KL, beta=1 -> reverse KL limits).
(b) Toy: teacher = a near-oracle policy for "4 digits in 0..4 that sum to a target"; the student is a
    tiny GRU trained from scratch with each method for the same number of steps.

Run:  python3 code/on-policy-distillation.py     (CPU, ~1 min)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)

VOCAB, LENGTH, MAX_T = 5, 4, 16


# ----------------------------------------------------------------------------------------
# (a) Losses
# ----------------------------------------------------------------------------------------
def sampled_reverse_kl_pg_loss(student_logp, teacher_logp, mask, student_logp_old=None):
    """student_logp, teacher_logp: [B,T] log-probs of the SAMPLED tokens (sampled from the student).
    Per-token advantage A_t = -(log p_s(x_t) - log p_t(x_t)) (detached, no discounting); loss is the
    usual (importance-weighted) policy-gradient loss. Its expectation is the gradient of the
    per-token reverse KL KL(p_s || p_t) on the student's own distribution."""
    adv = -(student_logp - teacher_logp).detach()
    if student_logp_old is None:
        weight = torch.ones_like(student_logp)
    else:                        # off-policy minibatches: importance ratio as in PPO (no clipping here)
        weight = torch.exp(student_logp - student_logp_old)
        return -(weight * adv * mask).sum() / mask.sum()
    return -(student_logp * adv * mask).sum() / mask.sum()


def full_reverse_kl_loss(student_logits, teacher_logits, mask):
    """Exact per-position reverse KL over the vocabulary, on student-generated prefixes."""
    ls, lt = F.log_softmax(student_logits, -1), F.log_softmax(teacher_logits, -1)
    kl = (ls.exp() * (ls - lt)).sum(-1)
    return (kl * mask).sum() / mask.sum()


def generalized_jsd_loss(student_logits, teacher_logits, mask, beta=0.5):
    """GKD (Agarwal et al.): JSD_beta(p_t || p_s) = beta KL(p_t || m) + (1-beta) KL(p_s || m),
    m = beta p_t + (1-beta) p_s. Small beta behaves like forward KL, beta near 1 like reverse KL
    (up to scaling). Computed per position over the vocabulary."""
    ls, lt = F.log_softmax(student_logits, -1), F.log_softmax(teacher_logits, -1)
    log_m = torch.logsumexp(torch.stack([lt + math.log(beta), ls + math.log(1 - beta)]), 0)
    jsd = beta * (lt.exp() * (lt - log_m)).sum(-1) + (1 - beta) * (ls.exp() * (ls - log_m)).sum(-1)
    return (jsd * mask).sum() / mask.sum()


# ----------------------------------------------------------------------------------------
# (b) Teacher, student, task
# ----------------------------------------------------------------------------------------
def teacher_logits(prompts, tokens):
    """Near-oracle teacher: at each step, digits that keep the target reachable get logit 0, the
    rest -8. Prefers smaller digits slightly (a 'style' the student should also pick up)."""
    B, T = tokens.shape
    out = []
    partial = torch.zeros(B, dtype=torch.long)
    digits = torch.arange(VOCAB)
    for t in range(T):
        remaining = prompts[:, None] - partial[:, None] - digits[None]      # after choosing digit
        slots = T - t - 1
        ok = (remaining >= 0) & (remaining <= (VOCAB - 1) * slots)
        out.append(torch.where(ok, -0.2 * digits.float()[None], torch.full((B, VOCAB), -8.0)))
        partial = partial + tokens[:, t]
    return torch.stack(out, 1)


@torch.no_grad()
def teacher_sample(prompts, T):
    toks = torch.zeros(prompts.shape[0], 0, dtype=torch.long)
    for t in range(T):
        pad = torch.cat([toks, torch.zeros(prompts.shape[0], 1, dtype=torch.long)], 1)
        logits = teacher_logits(prompts, pad)[:, t]
        toks = torch.cat([toks, torch.distributions.Categorical(logits=logits).sample()[:, None]], 1)
    return toks


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

    @torch.no_grad()
    def sample(self, prompts, T):
        B = prompts.shape[0]
        h = torch.tanh(self.prompt_emb(prompts))
        inp = torch.full((B,), self.V, dtype=torch.long)
        toks = []
        for _ in range(T):
            h = self.gru(self.tok_emb(inp), h)
            inp = torch.distributions.Categorical(logits=self.head(h)).sample()
            toks.append(inp)
        return torch.stack(toks, 1)


def gather(logits, toks):
    return F.log_softmax(logits, -1).gather(-1, toks.unsqueeze(-1)).squeeze(-1)


def train(method, steps=150, B=256, lr=3e-3, seed=0, log_every=50):
    torch.manual_seed(seed)
    st = TinyGRUPolicy(MAX_T + 1, VOCAB)
    opt = torch.optim.Adam(st.parameters(), lr=lr)
    hist = []
    for step in range(steps + 1):
        prompts = torch.randint(0, MAX_T + 1, (B,))
        mask = torch.ones(B, LENGTH)
        if method == "SFT on teacher samples (off-policy)":
            toks = teacher_sample(prompts, LENGTH)
            loss = -(gather(st(prompts, toks), toks) * mask).mean()        # forward-KL / NLL
        else:
            toks = st.sample(prompts, LENGTH)                              # ON-POLICY: student samples
            s_logits = st(prompts, toks)
            t_logits = teacher_logits(prompts, toks)
            if method == "on-policy reverse KL (sampled-token reward)":
                loss = sampled_reverse_kl_pg_loss(gather(s_logits, toks), gather(t_logits, toks), mask)
            elif method == "on-policy reverse KL (full vocab)":
                loss = full_reverse_kl_loss(s_logits, t_logits, mask)
            elif method == "GKD on-policy JSD(beta=0.5)":
                loss = generalized_jsd_loss(s_logits, t_logits, mask, beta=0.5)
            elif method == "RL with 0/1 outcome reward (GRPO-ish)":
                r = (toks.sum(1) == prompts).float()
                # per-prompt baseline is unavailable with 1 sample/prompt -> batch-mean baseline
                adv = (r - r.mean()) / (r.std() + 1e-6)
                loss = -(gather(s_logits, toks) * adv[:, None]).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % log_every == 0:
            with torch.no_grad():
                p = torch.arange(MAX_T + 1).repeat_interleave(64)
                tk = st.sample(p, LENGTH)
                acc = (tk.sum(1) == p).float().mean().item()
                rkl = full_reverse_kl_loss(st(p, tk), teacher_logits(p, tk), torch.ones_like(tk).float()).item()
            hist.append((step, acc, rkl))
    return hist


if __name__ == "__main__":
    with torch.no_grad():
        p = torch.arange(MAX_T + 1).repeat_interleave(64)
        tk = teacher_sample(p, LENGTH)
        print(f"teacher accuracy: {(tk.sum(1) == p).float().mean():.3f}\n")
    # sanity: the sampled-token estimator's expected gradient matches the exact reverse-KL gradient
    torch.manual_seed(1)
    logits = torch.randn(1, 1, VOCAB, requires_grad=True)
    t_log = torch.randn(1, 1, VOCAB)
    full_reverse_kl_loss(logits, t_log, torch.ones(1, 1)).backward()
    g_exact = logits.grad.clone()
    logits.grad = None
    xs = torch.distributions.Categorical(logits=logits.detach()).sample((20000,)).view(-1, 1)
    lp = F.log_softmax(logits, -1).expand(20000, 1, VOCAB)
    sampled_reverse_kl_pg_loss(lp.gather(-1, xs[..., None]).squeeze(-1),
                               F.log_softmax(t_log, -1).expand(20000, 1, VOCAB).gather(-1, xs[..., None]).squeeze(-1),
                               torch.ones(20000, 1)).backward()
    print("gradient check (exact vs sampled-token estimator, 20k samples):")
    print("  exact  :", [round(x, 3) for x in g_exact.flatten().tolist()])
    print("  sampled:", [round(x, 3) for x in logits.grad.flatten().tolist()], "\n")

    methods = ["SFT on teacher samples (off-policy)", "on-policy reverse KL (sampled-token reward)",
               "on-policy reverse KL (full vocab)", "GKD on-policy JSD(beta=0.5)",
               "RL with 0/1 outcome reward (GRPO-ish)"]
    print(f"{'method':46s} " + " | ".join(f"step {s:3d}: acc / revKL" for s in (0, 50, 100, 150)))
    for m in methods:
        h = train(m)
        print(f"{m:46s} " + " | ".join(f"     {a:.3f} / {k:.3f}  " for _, a, k in h))
