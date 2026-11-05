"""
Reward modeling for RLHF: Bradley-Terry pairwise loss, RM ensembles, and a small
Goodhart / over-optimization demonstration (best-of-n against a proxy RM).

Toy world
---------
* A "response" is a token sequence over a vocabulary of 10 tokens, length 3..12 (BoN policy).
  Tokens {0,1,2} are "helpful"; the rest are filler.
* GOLD (true human) reward = #helpful tokens  - 0.5 * max(0, length - 8)^2
  (people like helpful content but strongly dislike rambling beyond 8 tokens).
* Preference data is collected only on responses of length 3..8 (the SFT model's range),
  labelled by simulated annotators: P(a > b) = sigmoid(gold(a) - gold(b)).
* The reward model (RM) sees token counts + a length one-hot. Lengths 9..16 are never seen
  during training, so the RM must extrapolate there -> classic over-optimization.

Run: python3 code/reward_modeling.py     (CPU, ~15 s)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_num_threads(1)
V, LMAX, HELPFUL = 10, 16, (0, 1, 2)

# ----------------------------------------------------------------------------------
# (a) Losses -- exactly what TRL RewardTrainer / OpenRLHF compute
# ----------------------------------------------------------------------------------

def bradley_terry_loss(r_chosen, r_rejected, margin=None, center_coef=0.0):
    """-log sigmoid(r_c - r_r [- margin]).  r_*: [B] scalar rewards.
    margin: optional [B] tensor (Llama-2 style: larger margin for 'much better' labels).
    center_coef: optional penalty (r_c + r_r)^2 that keeps rewards centred near 0."""
    diff = r_chosen - r_rejected
    if margin is not None:
        diff = diff - margin
    loss = -F.logsigmoid(diff).mean()
    if center_coef > 0:
        loss = loss + center_coef * ((r_chosen + r_rejected) ** 2).mean()
    return loss


def soft_bt_loss(r_a, r_b, p_a_pref):
    """BT with soft labels p = P(a > b) (e.g., from an AI labeler's probabilities)."""
    return F.binary_cross_entropy_with_logits(r_a - r_b, p_a_pref)


def conservative_reward(ensemble_rewards, k=1.0):
    """Ensemble -> pessimistic reward: mean - k * std (uncertainty-weighted optimization,
    cf. Coste et al. 2023). ensemble_rewards: [M, B]."""
    return ensemble_rewards.mean(0) - k * ensemble_rewards.std(0)

# ----------------------------------------------------------------------------------
# (b) Toy data + RM
# ----------------------------------------------------------------------------------

def sample_responses(n, lmin=3, lmax=8, p_helpful=0.3):
    lengths = torch.randint(lmin, lmax + 1, (n,))
    toks = torch.where(torch.rand(n, LMAX) < p_helpful,
                       torch.randint(0, 3, (n, LMAX)), torch.randint(3, V, (n, LMAX)))
    mask = torch.arange(LMAX)[None] < lengths[:, None]
    return toks, mask, lengths


def featurize(toks, mask, lengths):
    counts = F.one_hot(toks, V).float().mul(mask[..., None]).sum(1)          # [n, V]
    len_oh = F.one_hot(lengths - 1, LMAX).float()                          # [n, LMAX]
    return torch.cat([counts, len_oh], dim=1)


def gold_reward(toks, mask, lengths):
    n_help = (torch.isin(toks, torch.tensor(HELPFUL)) & mask).sum(1).float()
    return n_help - 0.5 * torch.clamp(lengths - 8, min=0).float() ** 2


class RewardModel(nn.Module):
    """Stand-in for 'LLM backbone + scalar value head'."""
    def __init__(self, d_in=V + LMAX, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def make_preference_dataset(n_pairs):
    a, b = sample_responses(n_pairs), sample_responses(n_pairs)
    ga, gb = gold_reward(*a), gold_reward(*b)
    a_wins = torch.rand(n_pairs) < torch.sigmoid(ga - gb)       # noisy BT annotators
    fa, fb = featurize(*a), featurize(*b)
    chosen = torch.where(a_wins[:, None], fa, fb)
    rejected = torch.where(a_wins[:, None], fb, fa)
    return chosen, rejected


def train_rm(chosen, rejected, seed, epochs=60, bs=256):
    torch.manual_seed(seed)
    rm = RewardModel()
    opt = torch.optim.AdamW(rm.parameters(), lr=3e-3, weight_decay=1e-2)
    n = chosen.size(0)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = bradley_terry_loss(rm(chosen[idx]), rm(rejected[idx]), center_coef=1e-3)
            opt.zero_grad(); loss.backward(); opt.step()
    return rm


@torch.no_grad()
def pairwise_accuracy(rm, chosen, rejected):
    return (rm(chosen) > rm(rejected)).float().mean().item()


@torch.no_grad()
def best_of_n_curve(score_fn, ns, n_prompts=400):
    """Best-of-n from a policy that CAN produce long outputs (length 3..12).
    KL(BoN || base) = log n - (n-1)/n  (analytic formula, Stiennon et al. 2020 / Gao et al. 2022)."""
    rows = []
    for n in ns:
        toks, mask, lengths = sample_responses(n_prompts * n, 3, 12)
        s = score_fn(featurize(toks, mask, lengths)).view(n_prompts, n)
        g = gold_reward(toks, mask, lengths).view(n_prompts, n)
        best = s.argmax(1)
        rows.append((n, math.log(n) - (n - 1) / n,
                     s.gather(1, best[:, None]).mean().item(),
                     g.gather(1, best[:, None]).mean().item(),
                     lengths.view(n_prompts, n).gather(1, best[:, None]).float().mean().item()))
    return rows


def main():
    chosen, rejected = make_preference_dataset(6000)
    val_c, val_r = make_preference_dataset(2000)
    ensemble = [train_rm(chosen, rejected, seed=s) for s in range(5)]
    rm = ensemble[0]
    print(f"held-out pairwise accuracy (in-distribution): "
          f"{pairwise_accuracy(rm, val_c, val_r):.3f}  (noisy labels => ceiling < 1)")

    ns = [1, 4, 16, 64, 256, 1024]
    print("\nBest-of-n with a SINGLE proxy RM  (watch gold peak then fall = Goodhart)")
    print("   n   KL(nats)  proxy   gold   avg_len")
    for n, kl, s, g, L in best_of_n_curve(lambda x: rm(x), ns):
        print(f"{n:4d}   {kl:6.2f}   {s:6.2f}  {g:6.2f}   {L:5.1f}")

    def ens_score(x):
        return conservative_reward(torch.stack([m(x) for m in ensemble]), k=2.0)
    print("\nBest-of-n with a 5-member ENSEMBLE, score = mean - 2*std (pessimism off-distribution)")
    print("   n   KL(nats)  proxy   gold   avg_len")
    for n, kl, s, g, L in best_of_n_curve(ens_score, ns):
        print(f"{n:4d}   {kl:6.2f}   {s:6.2f}  {g:6.2f}   {L:5.1f}")


if __name__ == "__main__":
    main()
