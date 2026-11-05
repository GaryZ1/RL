"""
DPO variants: one loss function per method + a side-by-side comparison on a toy task.

Losses implemented (all on [B] tensors of SUMMED log-probs and response lengths):
  dpo, cdpo (label smoothing), robust_dpo (Chowdhury et al.), ipo, kto (unpaired),
  orpo (reference-free, odds ratio + NLL), simpo (reference-free, length-normalized, margin),
  r_dpo (explicit length regularization), beta_dpo (batch-adaptive beta, simplified).

Toy task: unconditional generation of variable-length sequences (EOS-terminated, max 10).
  Gold quality = fraction of response tokens in {1,2,3}  (length-neutral!).
  Simulated annotators are BIASED toward longer answers and 20% of labels are flipped.
  So a good method should raise quality WITHOUT simply inflating length.

Run: python3 code/dpo_variants.py     (CPU, ~1 min)
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_num_threads(1)

# ----------------------------------------------------------------------------------
# (a) Loss functions.  pc/pr = policy logp(chosen/rejected), rc/rr = reference logp,
#     lc/lr = lengths (#tokens).  h = (pc - rc) - (pr - rr) is the DPO log-ratio margin.
# ----------------------------------------------------------------------------------

def dpo(pc, pr, rc, rr, beta=0.1, **_):
    return -F.logsigmoid(beta * ((pc - rc) - (pr - rr))).mean()


def cdpo(pc, pr, rc, rr, beta=0.1, eps=0.2, **_):
    """Conservative DPO: assume each label is flipped w.p. eps -> label smoothing."""
    z = beta * ((pc - rc) - (pr - rr))
    return (-(1 - eps) * F.logsigmoid(z) - eps * F.logsigmoid(-z)).mean()


def robust_dpo(pc, pr, rc, rr, beta=0.1, eps=0.2, **_):
    """rDPO (Chowdhury et al. 2024): unbiased loss under flip rate eps < 0.5."""
    z = beta * ((pc - rc) - (pr - rr))
    return ((-(1 - eps) * F.logsigmoid(z) + eps * F.logsigmoid(-z)) / (1 - 2 * eps)).mean()


def ipo(pc, pr, rc, rr, tau=0.1, lc=None, lr=None, length_norm=True, **_):
    """IPO (Azar et al. 2023): squared loss regressing the log-ratio margin to 1/(2 tau).
    TRL averages log-probs over tokens for IPO (length_norm=True); the paper uses sums."""
    if length_norm:
        h = (pc - rc) / lc - (pr - rr) / lr
    else:
        h = (pc - rc) - (pr - rr)
    return ((h - 1 / (2 * tau)) ** 2).mean()


def kto(p_logp, r_logp, desirable, z_ref, beta=0.1, lam_d=1.0, lam_u=1.0, **_):
    """KTO (Ethayarajh et al. 2024): UNPAIRED examples, each labelled desirable or not.
    r = log pi/ref (implicit reward); z_ref = detached, clamped estimate of KL(pi||ref) that
    acts as the prospect-theory 'reference point'.  Loss = lambda_y - v(x,y)."""
    r = p_logp - r_logp
    v_d = lam_d * torch.sigmoid(beta * (r - z_ref))
    v_u = lam_u * torch.sigmoid(beta * (z_ref - r))
    lam = torch.where(desirable, torch.full_like(r, lam_d), torch.full_like(r, lam_u))
    return (lam - torch.where(desirable, v_d, v_u)).mean()


def orpo(pc, pr, lc, lr, lam=0.5, **_):
    """ORPO (Hong et al. 2024): NLL(chosen) + lam * -log sigmoid(log odds_c - log odds_r).
    p(y) is the length-normalized likelihood exp(mean token logp); odds = p / (1 - p).
    No reference model."""
    mc, mr = pc / lc, pr / lr                                   # mean token log-probs (<0)
    log_odds = (mc - torch.log1p(-torch.exp(mc).clamp(max=1 - 1e-6))) - \
               (mr - torch.log1p(-torch.exp(mr).clamp(max=1 - 1e-6)))
    return (-mc - lam * F.logsigmoid(log_odds)).mean()


def simpo(pc, pr, lc, lr, beta=2.0, gamma=1.0, **_):
    """SimPO (Meng et al. 2024): reward = beta * average log-prob (no reference model),
    with a target margin gamma:  -log sigmoid(beta*pc/|yc| - beta*pr/|yr| - gamma)."""
    return -F.logsigmoid(beta * pc / lc - beta * pr / lr - gamma).mean()


def r_dpo(pc, pr, rc, rr, lc, lr, beta=0.1, alpha=0.3, **_):
    """R-DPO (Park et al. 2024): DPO with an explicit length term,
    -log sigmoid(beta*h + alpha*(|yc| - |yr|)): pairs where the winner is merely longer
    already get a large margin, so they contribute less gradient (length is 'explained')."""
    return -F.logsigmoid(beta * ((pc - rc) - (pr - rr)) + alpha * (lc - lr)).mean()


class BetaDPO:
    """beta-DPO (Wu et al. 2024), simplified: beta_batch = beta0 * (1 + a * (M_batch - M0)),
    where M is the batch-mean implicit reward margin and M0 an EMA of it. (The paper also
    filters outlier pairs by margin; omitted here.)"""
    def __init__(self, beta0=0.1, a=0.5, momentum=0.9):
        self.beta0, self.a, self.m, self.M0 = beta0, a, momentum, None

    def __call__(self, pc, pr, rc, rr, **_):
        h = (pc - rc) - (pr - rr)
        M = (self.beta0 * h).detach().mean()
        self.M0 = M if self.M0 is None else self.m * self.M0 + (1 - self.m) * M
        beta = (self.beta0 * (1 + self.a * (M - self.M0))).clamp(min=1e-3)
        return -F.logsigmoid(beta * h).mean()

# Step-DPO (Lai et al. 2024) reuses `dpo` unchanged: the "prompt" is question + the verified
# correct reasoning prefix, and chosen/rejected are a single correct vs first-wrong step.

# ----------------------------------------------------------------------------------
# (b) Toy environment and model
# ----------------------------------------------------------------------------------
V, EOS, BOS, LMAX, H = 10, 0, 10, 10, 64      # tokens 1..9 content, 0 = EOS, 10 = BOS
GOOD = torch.tensor([1, 2, 3])


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb, self.gru, self.head = nn.Embedding(V + 1, H), nn.GRU(H, H, batch_first=True), nn.Linear(H, V)

    def forward(self, x):
        return self.head(self.gru(self.emb(x))[0])


def seq_logp(model, y, mask):
    """Summed log-prob of y (incl. EOS) given BOS."""
    inp = torch.cat([torch.full((y.size(0), 1), BOS), y[:, :-1]], 1)
    lp = torch.log_softmax(model(inp), -1).gather(-1, y[..., None]).squeeze(-1)
    return (lp * mask).sum(1)


@torch.no_grad()
def generate(model, n):
    tok, h, out = torch.full((n, 1), BOS), None, []
    for _ in range(LMAX):
        x, h = model.gru(model.emb(tok), h)
        tok = torch.distributions.Categorical(logits=model.head(x[:, -1])).sample()[:, None]
        out.append(tok)
    y = torch.cat(out, 1)
    is_eos = (y == EOS).int()
    # mask = tokens up to and including the first EOS
    mask = ((is_eos.cumsum(1) - is_eos) == 0).float()
    return y, mask


def quality(y, mask):
    content = mask * (y != EOS).float()
    good = (torch.isin(y, GOOD).float() * content).sum(1)
    return good / content.sum(1).clamp(min=1)


def length(mask):
    return mask.sum(1)


def make_sft_model():
    """SFT on uniform random content with lengths 2..6."""
    m = TinyLM()
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for _ in range(300):
        L = torch.randint(2, 7, (128,))
        y = torch.randint(1, V, (128, LMAX))
        y[torch.arange(128), L] = EOS
        mask = (torch.arange(LMAX)[None] <= L[:, None]).float()
        loss = -seq_logp(m, y, mask).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return m


def make_pairs(model, n=6000, len_bias=1.0, flip=0.2):
    ya, ma = generate(model, n)
    yb, mb = generate(model, n)
    logit = 6.0 * (quality(ya, ma) - quality(yb, mb)) + len_bias * (length(ma) - length(mb))
    a_wins = torch.rand(n) < torch.sigmoid(logit)
    a_wins ^= torch.rand(n) < flip                               # label noise
    w = a_wins[:, None]
    return (torch.where(w, ya, yb), torch.where(w, ma, mb),
            torch.where(w, yb, ya), torch.where(w, mb, ma))


def train(name, loss_fn, sft_model, data, ref_lp, steps=300, bs=128, lr=1e-3):
    yc, mc, yr, mr = data
    rc_all, rr_all = ref_lp
    policy = copy.deepcopy(sft_model)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    n = yc.size(0)
    for _ in range(steps):
        i = torch.randint(0, n, (bs,))
        pc, pr = seq_logp(policy, yc[i], mc[i]), seq_logp(policy, yr[i], mr[i])
        lc, lr_ = length(mc[i]), length(mr[i])
        if name == "KTO":
            # break the pairing: half of the batch from chosen (desirable), half from rejected
            j = torch.randint(0, n, (bs,))
            p_lp = torch.cat([pc, seq_logp(policy, yr[j], mr[j])])
            r_lp = torch.cat([rc_all[i], rr_all[j]])
            desirable = torch.cat([torch.ones(bs, dtype=torch.bool), torch.zeros(bs, dtype=torch.bool)])
            z_ref = (p_lp - r_lp).mean().clamp(min=0).detach()   # crude batch KL estimate
            loss = loss_fn(p_lp, r_lp, desirable, z_ref)
        else:
            loss = loss_fn(pc=pc, pr=pr, rc=rc_all[i], rr=rr_all[i], lc=lc, lr=lr_)
        opt.zero_grad(); loss.backward(); opt.step()
    y, m = generate(policy, 2000)
    return quality(y, m).mean().item(), length(m).mean().item()


def main():
    sft_model = make_sft_model()
    data = make_pairs(sft_model)
    ref = copy.deepcopy(sft_model).eval()
    with torch.no_grad():
        ref_lp = (seq_logp(ref, data[0], data[1]), seq_logp(ref, data[2], data[3]))
    y, m = generate(sft_model, 2000)
    lc, lr = length(data[1]), length(data[3])
    print(f"pairs where chosen is longer / shorter: {(lc > lr).float().mean():.2f} / {(lc < lr).float().mean():.2f}")
    print(f"{'method':<11}{'gold quality':>13}{'avg length':>12}")
    print(f"{'SFT (ref)':<11}{quality(y, m).mean():>13.3f}{length(m).mean():>12.2f}")
    methods = {
        "DPO": dpo, "cDPO": cdpo, "rDPO": robust_dpo, "IPO": ipo, "KTO": kto,
        "ORPO": orpo, "SimPO": simpo, "R-DPO": r_dpo, "beta-DPO": BetaDPO(),
    }
    for name, fn in methods.items():
        torch.manual_seed(1)
        q, L = train(name, fn, sft_model, data, ref_lp)
        print(f"{name:<11}{q:>13.3f}{L:>12.2f}")


if __name__ == "__main__":
    main()
