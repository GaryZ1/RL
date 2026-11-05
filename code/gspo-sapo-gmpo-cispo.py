"""Importance-ratio / clipping redesigns for LLM RL: GSPO, GSPO-token, SAPO, GMPO, CISPO, (+TIS).

(a) Each objective as a loss function on tensors (logp, logp_old: [B,T], adv: [B] or [B,T], mask).
(b) Demo 1: per-token gradient weight dLoss/dlogp as a function of the ratio r (A = +1 / -1).
    Demo 2: token-level vs sequence-level ratios when one token is an outlier.
    Demo 3: short toy training run with each objective (4 off-policy minibatch steps per batch),
            with the rollout sampled from a slightly *mismatched* "inference engine" and an
            optional truncated-importance-sampling (TIS) correction.

Run:  python3 code/gspo-sapo-gmpo-cispo.py      (CPU, ~1-2 min)
"""
import torch
import torch.nn as nn

torch.set_num_threads(1)


def _expand(adv, like):
    return adv.unsqueeze(1).expand_as(like) if adv.dim() == 1 else adv


def token_mean(x, mask):
    return (x * mask).sum() / mask.sum().clamp(min=1)


# ----------------------------------------------------------------------------------------
# (a) Objectives (all return a scalar loss to MINIMISE)
# ----------------------------------------------------------------------------------------
def ppo_token_loss(logp, logp_old, adv, mask, eps_low=0.2, eps_high=0.2):
    """Baseline: GRPO/PPO token-level ratio with hard clipping (eps_high > eps_low = DAPO)."""
    adv = _expand(adv, logp)
    ratio = torch.exp(logp - logp_old)
    pg = -torch.min(ratio * adv, ratio.clamp(1 - eps_low, 1 + eps_high) * adv)
    return token_mean(pg, mask)


def gspo_loss(logp, logp_old, adv, mask, eps_low=3e-4, eps_high=4e-4):
    """GSPO (Zheng et al. 2025): ONE length-normalised ratio per sequence,
    s_i = (pi(y_i|x)/pi_old(y_i|x))^(1/|y_i|) = exp(mean_t log r_it), clipped at the sequence level.
    Clip ranges are tiny (order 1e-4) because s_i is a geometric mean and stays near 1."""
    log_s = ((logp - logp_old) * mask).sum(1) / mask.sum(1).clamp(min=1)
    s = torch.exp(log_s)
    pg = -torch.min(s * adv, s.clamp(1 - eps_low, 1 + eps_high) * adv)
    return pg.mean()


def gspo_token_loss(logp, logp_old, adv_tok, mask, eps_low=3e-4, eps_high=4e-4):
    """GSPO-token: s_it = sg[s_i] * pi(y_it)/sg[pi(y_it)] -- numerically equal to s_i, but each token
    can carry its own advantage A_it (e.g. multi-turn / process rewards)."""
    adv_tok = _expand(adv_tok, logp)
    log_s = ((logp - logp_old) * mask).sum(1, keepdim=True) / mask.sum(1, keepdim=True).clamp(min=1)
    s_tok = torch.exp(log_s.detach() + logp - logp.detach())
    pg = -torch.min(s_tok * adv_tok, s_tok.clamp(1 - eps_low, 1 + eps_high) * adv_tok)
    return ((pg * mask).sum(1) / mask.sum(1).clamp(min=1)).mean()


def sapo_loss(logp, logp_old, adv, mask, tau_pos=1.0, tau_neg=1.05):
    """SAPO (Qwen, 2025): replace the hard clip with a smooth sigmoid gate on the ratio,
    f(r) = (4/tau) * sigmoid(tau * (r - 1)), tau = tau_pos if A>0 else tau_neg.
    df/dr = 4 sigma (1 - sigma): equals 1 on-policy (r=1) and decays smoothly off-policy.
    tau_neg > tau_pos makes negative-advantage tokens decay faster."""
    adv = _expand(adv, logp)
    ratio = torch.exp(logp - logp_old)
    tau = torch.where(adv > 0, torch.full_like(adv, tau_pos), torch.full_like(adv, tau_neg))
    gate = (4.0 / tau) * torch.sigmoid(tau * (ratio - 1.0))
    return token_mean(-gate * adv, mask)


def gmpo_loss(logp, logp_old, adv, mask, eps=0.4):
    """GMPO (Zhao et al. 2025): maximise the GEOMETRIC mean of token-level clipped ratios,
    A_i * exp( mean_t  sgn(A) * min(sgn(A) log r_t, clip(sgn(A) log r_t, -eps, eps)) ).
    Clipping happens in log space, which tolerates a wider range, e.g. (e^-0.4, e^0.4)."""
    sgn = torch.sign(adv).unsqueeze(1)
    log_r = logp - logp_old
    signed = sgn * log_r
    signed_min = torch.min(signed, signed.clamp(-eps, eps))
    log_ratio_seq = (sgn * signed_min * mask).sum(1) / mask.sum(1).clamp(min=1)
    return (-adv * torch.exp(log_ratio_seq)).mean()


def cispo_loss(logp, logp_old, adv, mask, eps_high=0.28, eps_low=None):
    """CISPO (MiniMax-M1, 2025): clip the importance-sampling WEIGHT (stop-gradient) instead of
    the update, so every token -- including rare, low-probability 'fork' tokens -- keeps a gradient:
    loss = - sg(clip(r, 1-eps_low, 1+eps_high)) * A * logp   (token-level mean).
    MiniMax report effectively no lower bound (eps_low large), only an upper cap."""
    adv = _expand(adv, logp)
    ratio = torch.exp(logp - logp_old)
    lo = 0.0 if eps_low is None else 1 - eps_low
    w = ratio.clamp(lo, 1 + eps_high).detach()
    return token_mean(-w * adv * logp, mask)


def tis_weight(logp_old_trainer, logp_rollout, cap=2.0):
    """Truncated importance sampling for training-inference mismatch: tokens were sampled by the
    inference engine (e.g. vLLM, distribution pi_rollout) but the loss uses the trainer's (e.g. FSDP)
    pi_old. Per-token weight min(pi_old / pi_rollout, C), detached; multiply into the per-token loss."""
    return torch.exp(logp_old_trainer - logp_rollout).clamp(max=cap).detach()


# ----------------------------------------------------------------------------------------
# (b) Demo 1: gradient weight vs ratio for a single token
# ----------------------------------------------------------------------------------------
def demo_gradient_weights():
    ratios = [0.5, 0.7, 0.95, 1.0, 1.1, 1.25, 1.5, 2.0]
    objectives = {
        "PPO/GRPO clip 0.2": lambda lp, lo, a, m: ppo_token_loss(lp, lo, a, m),
        "DAPO clip-higher": lambda lp, lo, a, m: ppo_token_loss(lp, lo, a, m, 0.2, 0.28),
        "CISPO (cap 1.28)": lambda lp, lo, a, m: cispo_loss(lp, lo, a, m, eps_high=0.28),
        "SAPO tau=1/1.05": lambda lp, lo, a, m: sapo_loss(lp, lo, a, m),
        "SAPO tau=8 (sharp)": lambda lp, lo, a, m: sapo_loss(lp, lo, a, m, 8.0, 8.4),
        "GMPO eps=0.4 (1 tok)": lambda lp, lo, a, m: gmpo_loss(lp, lo, a, m),
    }
    print("Demo 1: per-token gradient weight -dLoss/dlogp  (0 = token gets NO gradient)")
    print(" " * 25 + "".join(f"r={r:<5}" for r in ratios))
    for A in [1.0, -1.0]:
        for name, fn in objectives.items():
            row = []
            for r in ratios:
                logp = torch.tensor([[0.0]], requires_grad=True)
                logp_old = torch.tensor([[-torch.log(torch.tensor(r)).item()]])
                fn(logp, logp_old, torch.tensor([A]), torch.ones(1, 1)).backward()
                row.append(-logp.grad.item())
            print(f"A={A:+.0f} {name:21s}" + "".join(f"{g:+6.2f} " for g in row))
        print()


# Demo 2: one outlier token in a long sequence
def demo_sequence_ratio():
    torch.manual_seed(0)
    print("Demo 2: sequence of 200 tokens, small noise on log-ratios + ONE outlier token (r=3)")
    log_r = 0.02 * torch.randn(1, 200)
    log_r[0, 57] = torch.log(torch.tensor(3.0))
    r_tok = torch.exp(log_r)
    clipped = ((r_tok - 1).abs() > 0.2).float().mean().item()
    s = torch.exp(log_r.mean()).item()
    print(f"  token ratios: min {r_tok.min():.3f} max {r_tok.max():.3f}, clipped fraction (eps=0.2) "
          f"{clipped:.3f}\n  GSPO sequence ratio s_i = {s:.4f} -> the outlier moves s_i by only "
          f"~log(3)/200,\n  yet with eps ~4e-4 GSPO clips this whole sequence (GSPO clips whole sequences, not tokens).\n")


# ----------------------------------------------------------------------------------------
# Demo 3: toy training with each objective
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

    def token_logp(self, prompts, tokens, logit_noise=0.0):
        logits = self(prompts, tokens)
        if logit_noise > 0:      # crude stand-in for inference-engine numerics (bf16 kernels...)
            logits = logits + logit_noise * torch.randn_like(logits)
        return torch.log_softmax(logits, -1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)

    @torch.no_grad()
    def sample(self, prompts, T, logit_noise=0.0):
        B = prompts.shape[0]
        h = torch.tanh(self.prompt_emb(prompts))
        inp = torch.full((B,), self.V, dtype=torch.long)
        toks, lps = [], []
        for _ in range(T):
            h = self.gru(self.tok_emb(inp), h)
            logits = self.head(h) + logit_noise * torch.randn(B, self.V)
            dist = torch.distributions.Categorical(logits=logits)
            inp = dist.sample()
            toks.append(inp)
            lps.append(dist.log_prob(inp))
        return torch.stack(toks, 1), torch.stack(lps, 1)   # tokens and *rollout-engine* logprobs


def train(loss_name, use_tis=False, steps=120, n_prompts=32, G=8, minibatches=4, lr=3e-3,
          noise=0.5, seed=0):
    torch.manual_seed(seed)
    pol = TinyGRUPolicy(MAX_T + 1, VOCAB)
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    for step in range(steps):
        prompts = torch.randint(0, MAX_T + 1, (n_prompts,)).repeat_interleave(G)
        toks, logp_rollout = pol.sample(prompts, LENGTH, logit_noise=noise)
        mask = torch.ones_like(toks, dtype=torch.float)
        r = (toks.sum(1) == prompts).float()
        rg = r.view(-1, G)
        adv = ((rg - rg.mean(1, keepdim=True)) / (rg.std(1, keepdim=True) + 1e-6)).view(-1)
        with torch.no_grad():
            logp_old = pol.token_logp(prompts, toks)            # trainer-side old logprobs
            w = tis_weight(logp_old, logp_rollout) if use_tis else torch.ones_like(logp_old)
        perm = torch.randperm(len(prompts))
        for mb in perm.chunk(minibatches):                      # off-policy minibatch updates
            logp = pol.token_logp(prompts[mb], toks[mb])
            m = mask[mb] * w[mb]                                # TIS weight folded into the mask
            args = (logp, logp_old[mb], adv[mb], m)
            loss = {"GRPO": ppo_token_loss, "GSPO": gspo_loss, "GSPO-token": gspo_token_loss,
                    "SAPO": sapo_loss, "GMPO": gmpo_loss, "CISPO": cispo_loss}[loss_name](*args)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
            opt.step()
    with torch.no_grad():
        p = torch.arange(MAX_T + 1).repeat_interleave(32)
        t, _ = pol.sample(p, LENGTH)
        return (t.sum(1) == p).float().mean().item()


if __name__ == "__main__":
    demo_gradient_weights()
    demo_sequence_ratio()
    print("Demo 3: sampling accuracy after 120 steps (4 minibatch updates/batch, mismatched rollout "
          "engine, logit noise 0.5; mean of 2 seeds)")
    for name in ["GRPO", "GSPO", "GSPO-token", "SAPO", "GMPO", "CISPO"]:
        accs = [train(name, seed=s) for s in range(2)]
        line = f"  {name:11s} acc {sum(accs)/2:.3f}"
        if name in ("GRPO", "CISPO"):
            accs_tis = [train(name, use_tis=True, seed=s) for s in range(2)]
            line += f"   | with TIS (cap 2) {sum(accs_tis)/2:.3f}"
        print(line)
