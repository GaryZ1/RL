"""DAPO and Dr. GRPO -- loss aggregation modes, Clip-Higher, dynamic sampling, overlong shaping.

(a) Tensor functions: `aggregate_loss` (token-mean / seq-mean-token-mean / seq-mean-token-sum-norm),
    `clipped_pg_per_token` with decoupled eps_low / eps_high, `dr_grpo_advantages`,
    `dynamic_sampling_keep`, `overlong_reward_shaping`.
(b) Demo 1: analytic per-token gradient weights showing the length bias of 1/|o_i| normalisation.
    Demo 2: std-normalisation bias (easy/hard prompts get up-weighted).
    Demo 3: tiny variable-length toy task trained with GRPO, Dr. GRPO and DAPO settings.

Run:  python3 code/dapo-dr-grpo.py     (CPU, ~1 min)
"""
import torch
import torch.nn as nn

torch.set_num_threads(1)


# ----------------------------------------------------------------------------------------
# (a) Loss pieces
# ----------------------------------------------------------------------------------------
def aggregate_loss(per_token: torch.Tensor, mask: torch.Tensor, mode: str, max_len: int = None):
    """Reduce a [B, T] per-token loss to a scalar.
    token-mean               : sum_{i,t} l / sum_i |o_i|                 (DAPO; verl default)
    seq-mean-token-mean      : 1/B sum_i (1/|o_i|) sum_t l               (original GRPO)
    seq-mean-token-sum-norm  : 1/B sum_i sum_t l / L_max  (constant)     (Dr. GRPO)
    seq-mean-token-sum       : 1/B sum_i sum_t l
    """
    if mode == "token-mean":
        return (per_token * mask).sum() / mask.sum().clamp(min=1)
    if mode == "seq-mean-token-mean":
        return ((per_token * mask).sum(1) / mask.sum(1).clamp(min=1)).mean()
    if mode == "seq-mean-token-sum-norm":
        L = max_len if max_len is not None else mask.shape[1]
        return ((per_token * mask).sum(1) / L).mean()
    if mode == "seq-mean-token-sum":
        return (per_token * mask).sum(1).mean()
    raise ValueError(mode)


def clipped_pg_per_token(logp, logp_old, adv, eps_low=0.2, eps_high=0.28):
    """PPO/GRPO per-token surrogate with DAPO's *Clip-Higher*: ratio clipped to
    [1-eps_low, 1+eps_high] with eps_high > eps_low so low-probability tokens with A>0 can grow."""
    if adv.dim() == 1:
        adv = adv.unsqueeze(1).expand_as(logp)
    ratio = torch.exp(logp - logp_old)
    return -torch.min(ratio * adv, ratio.clamp(1 - eps_low, 1 + eps_high) * adv)


def grpo_advantages(rewards, G, eps=1e-6):
    r = rewards.view(-1, G)
    return ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + eps)).view(-1)


def dr_grpo_advantages(rewards, G):
    """Dr. GRPO: A_i = r_i - mean(r_group). No division by the group std."""
    r = rewards.view(-1, G)
    return (r - r.mean(1, keepdim=True)).view(-1)


def dynamic_sampling_keep(rewards, G):
    """DAPO dynamic sampling: keep only groups whose accuracy is strictly between 0 and 1
    (all-correct or all-wrong groups have zero advantage -> zero gradient). Returns [B] bool."""
    r = rewards.view(-1, G)
    acc = (r > 0).float().mean(1)
    keep = (acc > 0) & (acc < 1)
    return keep.repeat_interleave(G)


def overlong_reward_shaping(lengths, max_len, cache_len):
    """DAPO soft overlong punishment, added to the task reward:
    0 if L <= L_max - L_cache;  (L_max - L_cache - L)/L_cache inside the buffer;  -1 beyond L_max."""
    lengths = lengths.float()
    expected = max_len - cache_len
    pen = torch.where(lengths <= expected, torch.zeros_like(lengths), (expected - lengths) / cache_len)
    return torch.where(lengths > max_len, torch.full_like(lengths, -1.0), pen)


# ----------------------------------------------------------------------------------------
# (b) Demo 1 + 2: analytic biases
# ----------------------------------------------------------------------------------------
def demo_length_bias():
    print("Demo 1: per-token gradient weight |dLoss/dlogp_t| for one group (G=4, L_max=16)")
    lengths = torch.tensor([4, 16, 4, 16])
    rewards = torch.tensor([1., 1., 0., 0.])          # short+long correct, short+long wrong
    T = int(lengths.max())
    mask = (torch.arange(T)[None] < lengths[:, None]).float()
    adv = dr_grpo_advantages(rewards, 4)
    for mode in ["seq-mean-token-mean", "token-mean", "seq-mean-token-sum-norm"]:
        logp = torch.zeros(4, T, requires_grad=True)
        per_tok = -(logp * adv[:, None])               # on-policy REINFORCE form
        aggregate_loss(per_tok, mask, mode, max_len=T).backward()
        w = logp.grad.abs().max(1).values
        print(f"  {mode:24s} short-correct {w[0]:.4f} long-correct {w[1]:.4f} "
              f"short-wrong {w[2]:.4f} long-wrong {w[3]:.4f}")
    print("  -> with 1/|o_i| a long WRONG answer is penalised 4x less per token than a short one,\n"
          "     so wrong answers drift longer; token-sum-norm / token-mean treat tokens equally.\n")


def demo_std_bias():
    print("Demo 2: total gradient mass sum_i |A_i| of one group (G=8) vs. group accuracy")
    for n_correct in [1, 2, 4, 6, 7]:
        r = torch.zeros(8)
        r[:n_correct] = 1
        g, d = grpo_advantages(r, 8).abs().sum(), dr_grpo_advantages(r, 8).abs().sum()
        print(f"  {n_correct}/8 correct: GRPO {g:.2f} | Dr. GRPO {d:.2f} | GRPO/Dr ratio (=1/std) {g/d:.2f}")
    print("  -> dividing by std up-weights nearly-solved / nearly-unsolved prompts relative to\n"
          "     prompts of medium difficulty (a question-level difficulty bias).\n")


# ----------------------------------------------------------------------------------------
# (b) Demo 3: variable-length toy task
# ----------------------------------------------------------------------------------------
VOCAB = 5            # digits 0..3 and EOS(=4)
EOS, MAXLEN, MAX_T = 4, 12, 9


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
    def sample(self, prompts, T):
        B = prompts.shape[0]
        h = torch.tanh(self.prompt_emb(prompts))
        inp = torch.full((B,), self.V, dtype=torch.long)
        toks, masks = [], []
        alive = torch.ones(B, dtype=torch.bool)
        for _ in range(T):
            h = self.gru(self.tok_emb(inp), h)
            inp = torch.distributions.Categorical(logits=self.head(h)).sample()
            inp = torch.where(alive, inp, torch.full_like(inp, EOS))
            toks.append(inp)
            masks.append(alive.float())
            alive = alive & (inp != EOS)
        return torch.stack(toks, 1), torch.stack(masks, 1), alive    # alive = truncated


def task_reward(prompts, toks, mask, truncated):
    """Digits before EOS must sum to the prompt id; truncated answers count as wrong."""
    digits = toks.clone()
    digits[digits == EOS] = 0
    s = (digits * mask).sum(1)
    return ((s == prompts) & ~truncated).float()


def first_token_reward(prompts, toks, mask, truncated):
    """Length-bias probe: correct iff the FIRST token equals prompt % 4 (and not truncated).
    Everything after the first token is free 'filler' -- length does not affect correctness."""
    return ((toks[:, 0] == prompts % 4) & ~truncated & (toks[:, 0] != EOS)).float()


def train(cfg, steps=250, n_prompts=32, G=8, lr=3e-3, mu=2, seed=0, reward=task_reward,
          track_len=False):
    torch.manual_seed(seed)
    pol = TinyGRUPolicy(MAX_T + 1, VOCAB)
    wrong_lens, right_lens = [], []
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    for step in range(steps):
        prompts = torch.randint(0, MAX_T + 1, (n_prompts,)).repeat_interleave(G)
        toks, mask, trunc = pol.sample(prompts, MAXLEN)
        acc = reward(prompts, toks, mask, trunc)
        if track_len:
            L = mask.sum(1)
            if (acc == 0).any():
                wrong_lens.append(L[acc == 0].mean().item())
            if (acc == 1).any():
                right_lens.append(L[acc == 1].mean().item())
        r = acc.clone()
        if cfg.get("overlong"):
            r = r + overlong_reward_shaping(mask.sum(1), MAXLEN, cache_len=4)
        adv = dr_grpo_advantages(r, G) if cfg["adv"] == "dr" else grpo_advantages(r, G)
        if cfg.get("dynamic"):                   # (real DAPO keeps sampling until batch is full)
            keep = dynamic_sampling_keep(acc, G)
            if keep.sum() == 0:
                continue
            prompts, toks, mask, adv = prompts[keep], toks[keep], mask[keep], adv[keep]
        with torch.no_grad():
            logp_old = pol.token_logp(prompts, toks)
        for _ in range(mu):
            logp = pol.token_logp(prompts, toks)
            per_tok = clipped_pg_per_token(logp, logp_old, adv, 0.2, cfg.get("eps_high", 0.2))
            loss = aggregate_loss(per_tok, mask, cfg["agg"], max_len=MAXLEN)
            opt.zero_grad()
            loss.backward()
            opt.step()
    if track_len:
        return wrong_lens, right_lens
    with torch.no_grad():   # evaluation
        prompts = torch.arange(MAX_T + 1).repeat_interleave(64)
        toks, mask, trunc = pol.sample(prompts, MAXLEN)
        acc = reward(prompts, toks, mask, trunc)
    return acc.mean().item(), mask.sum(1).mean().item(), trunc.float().mean().item()


if __name__ == "__main__":
    demo_length_bias()
    demo_std_bias()
    print("overlong shaping (L_max=12, cache=4) for L=6,9,12,13:",
          overlong_reward_shaping(torch.tensor([6, 9, 12, 13]), 12, 4).tolist(), "\n")
    print("Demo 3: length-bias probe -- reward depends only on the first token, rest is filler.")
    print("  Mean length of WRONG vs CORRECT samples during training (initial mean length ~4.5):")
    for name, cfg in [("GRPO-style 1/|o_i| (seq-mean-token-mean)", dict(adv="dr", agg="seq-mean-token-mean")),
                      ("Dr. GRPO constant norm (seq-mean-token-sum-norm)",
                       dict(adv="dr", agg="seq-mean-token-sum-norm"))]:
        wl, rl = [], []
        for s in range(3):
            w, r = train(cfg, steps=60, lr=1e-3, seed=s, reward=first_token_reward, track_len=True)
            wl += w[-20:]
            rl += r[-20:]
        print(f"  {name:50s} len(wrong) {sum(wl)/len(wl):.2f} | len(correct) {sum(rl)/len(rl):.2f}")
    print()
    print("Demo 4: full recipes on 'digits before EOS sum to target' (avg over 2 seeds)")
    configs = {
        "GRPO     (std adv, seq-mean-token-mean)": dict(adv="std", agg="seq-mean-token-mean"),
        "Dr. GRPO (mean adv, seq-mean-token-sum-norm)": dict(adv="dr", agg="seq-mean-token-sum-norm"),
        "DAPO     (clip-higher, dyn. sampling, token-mean, overlong)":
            dict(adv="std", agg="token-mean", eps_high=0.28, dynamic=True, overlong=True),
    }
    for name, cfg in configs.items():
        res = [train(cfg, seed=s) for s in range(2)]
        a, L, tr = [sum(x[i] for x in res) / len(res) for i in range(3)]
        print(f"  {name:60s} acc {a:.3f} | mean len {L:.2f} | truncated {tr:.3f}")
