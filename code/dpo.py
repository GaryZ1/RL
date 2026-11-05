"""
Direct Preference Optimization (DPO) -- loss function + two runnable toy demos.

Part A (tabular check): one prompt, K candidate responses, true reward r(y). With BT
        preferences on all pairs, minimizing the DPO loss recovers the closed-form optimum
        of KL-regularized RL:  pi*(y) = pi_ref(y) exp(r(y)/beta) / Z.
Part B (sequence model): a tiny GRU LM is SFT'd on noisy demos, preference pairs are sampled
        from it and labelled by a hidden "gold" reward, and DPO is trained OFFLINE on them.

Run: python3 code/dpo.py      (CPU, ~30 s)
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_num_threads(1)

# ----------------------------------------------------------------------------------
# (a) The DPO loss -- same signature/semantics as TRL's DPOTrainer.dpo_loss
# ----------------------------------------------------------------------------------

def sequence_logps(logits, labels, mask):
    """Sum of per-token log-probs of `labels` under `logits`.
    logits [B, L, V] already aligned with labels [B, L]; mask [B, L] = 1 on response tokens."""
    logp = torch.log_softmax(logits, -1).gather(-1, labels[..., None]).squeeze(-1)
    return (logp * mask).sum(-1)


def dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta=0.1, label_smoothing=0.0):
    """pi_*/ref_*: [B] summed log-probs of chosen / rejected responses under policy / reference.
    loss = -log sigmoid(beta * [(log pi_c - log ref_c) - (log pi_r - log ref_r)])
    label_smoothing = eps > 0 gives conservative DPO (cDPO) for noisy labels.
    Returns (loss, chosen_rewards, rejected_rewards) where rewards are the IMPLICIT rewards
    beta * log(pi/ref) (detached), the quantities TRL logs as rewards/chosen, rewards/rejected."""
    logits = (pi_chosen - ref_chosen) - (pi_rejected - ref_rejected)
    losses = (-F.logsigmoid(beta * logits) * (1 - label_smoothing)
              - F.logsigmoid(-beta * logits) * label_smoothing)
    chosen_rewards = beta * (pi_chosen - ref_chosen).detach()
    rejected_rewards = beta * (pi_rejected - ref_rejected).detach()
    return losses.mean(), chosen_rewards, rejected_rewards

# ----------------------------------------------------------------------------------
# Part A: tabular sanity check of the closed form
# ----------------------------------------------------------------------------------

def tabular_check(K=6, beta=0.5):
    r = torch.tensor([1.0, 0.5, 0.0, -0.5, 2.0, 0.2])
    ref_logits = torch.tensor([0.5, 1.0, 0.0, 0.3, -1.0, 0.0])
    log_ref = torch.log_softmax(ref_logits, 0)
    theta = ref_logits.clone().requires_grad_(True)       # policy init = reference
    opt = torch.optim.Adam([theta], lr=0.05)
    i, j = torch.triu_indices(K, K, 1)                   # all pairs (i < j)
    p_i_wins = torch.sigmoid(r[i] - r[j])                # BT preference probabilities
    for _ in range(3000):
        log_pi = torch.log_softmax(theta, 0)
        h = beta * ((log_pi[i] - log_ref[i]) - (log_pi[j] - log_ref[j]))
        # expected DPO loss over the BT label distribution (both orderings)
        loss = -(p_i_wins * F.logsigmoid(h) + (1 - p_i_wins) * F.logsigmoid(-h)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    pi_dpo = torch.softmax(theta, 0).detach()
    pi_star = torch.softmax(log_ref + r / beta, 0)       # closed-form KL-regularized optimum
    print("Part A  pi_ref :", [f"{v:.3f}" for v in log_ref.exp().tolist()])
    print("        pi_DPO :", [f"{v:.3f}" for v in pi_dpo.tolist()])
    print("        pi*    :", [f"{v:.3f}" for v in pi_star.tolist()])
    print(f"        max |pi_DPO - pi*| = {(pi_dpo - pi_star).abs().max():.2e}\n")

# ----------------------------------------------------------------------------------
# Part B: sequence-level DPO on a tiny GRU LM
# ----------------------------------------------------------------------------------
V, T, H = 10, 6, 64


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb, self.gru, self.head = nn.Embedding(V, H), nn.GRU(H, H, batch_first=True), nn.Linear(H, V)

    def forward(self, tokens):
        h, _ = self.gru(self.emb(tokens))
        return self.head(h)


def logps(model, prompts, responses):
    inp = torch.cat([prompts[:, None], responses[:, :-1]], 1)
    return sequence_logps(model(inp), responses, torch.ones_like(responses, dtype=torch.float))


@torch.no_grad()
def generate(model, prompts):
    tok, h, out = prompts[:, None], None, []
    for _ in range(T):
        x, h = model.gru(model.emb(tok), h)
        tok = torch.distributions.Categorical(logits=model.head(x[:, -1])).sample()[:, None]
        out.append(tok)
    return torch.cat(out, 1)


def gold(prompts, responses):
    """Hidden human preference: counting pattern p+1, p+2, ... (fraction correct)."""
    tgt = (prompts[:, None] + torch.arange(1, T + 1)[None]) % V
    return (responses == tgt).float().mean(1)


def sequence_demo(beta=0.1, n_pairs=4000, steps=600, bs=128):
    sft_model = TinyLM()
    opt = torch.optim.Adam(sft_model.parameters(), lr=3e-3)
    for _ in range(300):                                  # SFT on 40%-correct demos
        p = torch.randint(0, V, (128,))
        tgt = (p[:, None] + torch.arange(1, T + 1)[None]) % V
        demo = torch.where(torch.rand(tgt.shape) < 0.4, tgt, torch.randint(0, V, tgt.shape))
        loss = -logps(sft_model, p, demo).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    # Offline preference dataset: two samples from the SFT model, labelled via BT on gold
    p = torch.randint(0, V, (n_pairs,))
    a, b = generate(sft_model, p), generate(sft_model, p)
    a_wins = torch.rand(n_pairs) < torch.sigmoid(8.0 * (gold(p, a) - gold(p, b)))
    chosen = torch.where(a_wins[:, None], a, b)
    rejected = torch.where(a_wins[:, None], b, a)

    ref = copy.deepcopy(sft_model).eval()
    with torch.no_grad():                                  # precompute reference log-probs once
        ref_c, ref_r = logps(ref, p, chosen), logps(ref, p, rejected)
    policy = copy.deepcopy(sft_model)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    q = torch.randint(0, V, (1000,))
    print(f"Part B  SFT gold reward: {gold(q, generate(sft_model, q)).mean():.3f}")
    for step in range(1, steps + 1):
        idx = torch.randint(0, n_pairs, (bs,))
        pc, pr = logps(policy, p[idx], chosen[idx]), logps(policy, p[idx], rejected[idx])
        loss, rc, rr = dpo_loss(pc, pr, ref_c[idx], ref_r[idx], beta=beta)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0:
            acc = (rc > rr).float().mean().item()
            print(f"  step {step:4d}  loss {loss.item():.3f}  reward acc {acc:.2f}  "
                  f"margin {(rc - rr).mean():.2f}  chosen {rc.mean():+.2f}  rejected {rr.mean():+.2f}  "
                  f"gold(sample) {gold(q, generate(policy, q)).mean():.3f}")


if __name__ == "__main__":
    tabular_check()
    sequence_demo()
