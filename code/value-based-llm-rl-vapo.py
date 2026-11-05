"""Value-based LLM RL: token-level PPO with a critic, VC-PPO / VAPO tricks.

(a) Tensor functions: token-level GAE with per-sequence lambda, length-adaptive lambda (VAPO),
    PPO actor loss (Clip-Higher, token-mean), clipped value loss, positive-example LM loss.
(b) Demo 1: how fast a terminal reward decays to early tokens with lambda < 1.
    Demo 2: a long-ish binary-token toy task ("emit exactly k ones in 32 tokens"; reward only at
            the end) trained with (i) naive PPO (biased critic init, shared lambda=0.95) and
            (ii) VAPO-style PPO (value pretraining, decoupled lambda: critic 1.0 / actor
            length-adaptive, Clip-Higher, token-level loss, positive-example LM loss).

Run:  python3 code/value-based-llm-rl-vapo.py     (CPU, ~1 min)
"""
import torch
import torch.nn as nn

torch.set_num_threads(1)


# ----------------------------------------------------------------------------------------
# (a) Pieces
# ----------------------------------------------------------------------------------------
def compute_gae(token_rewards, values, mask, gamma=1.0, lam=0.95):
    """Token-level GAE. token_rewards, values, mask: [B, T]; lam: float or [B] tensor
    (per-sequence lambda). V after the last valid token is 0 (episode ends at EOS).
    delta_t = r_t + gamma V_{t+1} - V_t ;  A_t = delta_t + gamma*lam*A_{t+1}."""
    B, T = token_rewards.shape
    lam = torch.as_tensor(lam, dtype=values.dtype).expand(B) if not torch.is_tensor(lam) or lam.dim() == 0 else lam
    adv = torch.zeros_like(values)
    next_v = torch.zeros(B)
    next_a = torch.zeros(B)
    for t in reversed(range(T)):
        m = mask[:, t]
        delta = token_rewards[:, t] + gamma * next_v - values[:, t]
        a = delta + gamma * lam * next_a
        adv[:, t] = a * m
        next_v = torch.where(m > 0, values[:, t], next_v * 0)   # padding positions reset
        next_a = torch.where(m > 0, a, next_a * 0)
    returns = adv + values * mask
    return adv, returns


def length_adaptive_lambda(lengths, alpha=0.05):
    """VAPO: lambda_policy = 1 - 1/(alpha * l). Long sequences get lambda -> 1 (less bias),
    short ones a smaller lambda (less variance). alpha=0.05 in the paper (lengths ~1e2-1e4);
    clamp for tiny toy lengths."""
    return (1.0 - 1.0 / (alpha * lengths.float())).clamp(0.0, 1.0)


def ppo_actor_loss(logp, logp_old, adv, mask, eps_low=0.2, eps_high=0.28):
    ratio = torch.exp(logp - logp_old)
    pg = -torch.min(ratio * adv, ratio.clamp(1 - eps_low, 1 + eps_high) * adv)
    return (pg * mask).sum() / mask.sum()                      # token-level mean


def value_loss(v, v_old, returns, mask, clip=0.2):
    v_clip = v_old + (v - v_old).clamp(-clip, clip)
    l = torch.max((v - returns) ** 2, (v_clip - returns) ** 2)
    return 0.5 * (l * mask).sum() / mask.sum()


def positive_lm_loss(logp, mask, outcome):
    """VAPO's extra NLL on correct (reward=1) samples (weight ~0.1 in the paper)."""
    pos = (outcome > 0).float().unsqueeze(1) * mask
    return -(logp * pos).sum() / pos.sum().clamp(min=1)


# ----------------------------------------------------------------------------------------
# (b) Toy environment: emit L binary tokens, reward 1 iff exactly k of them are 1
# ----------------------------------------------------------------------------------------
L = 32
K_MAX = L
CRITIC_INIT_BIAS = 1.0   # VC-PPO observation: a critic initialised from a reward model is biased
                         # (here: it starts out believing every prefix will succeed)


def features(k, pos, count):
    """Markov state = (target k, position, ones so far); one-hot concatenated."""
    return torch.cat([nn.functional.one_hot(k, K_MAX + 1), nn.functional.one_hot(pos, L),
                      nn.functional.one_hot(count, L + 1)], -1).float()


def mlp(out):
    return nn.Sequential(nn.Linear(K_MAX + 1 + L + L + 1, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(),
                         nn.Linear(64, out))


@torch.no_grad()
def rollout(actor, k):
    B = k.shape[0]
    count = torch.zeros(B, dtype=torch.long)
    toks, feats = [], []
    for t in range(L):
        f = features(k, torch.full((B,), t), count)
        a = torch.distributions.Categorical(logits=actor(f)).sample()
        toks.append(a)
        feats.append(f)
        count = count + a
    toks, feats = torch.stack(toks, 1), torch.stack(feats, 1)          # [B,L], [B,L,F]
    outcome = (count == k).float()
    return toks, feats, outcome


def logp_of(actor, feats, toks):
    return torch.log_softmax(actor(feats), -1).gather(-1, toks.unsqueeze(-1)).squeeze(-1)


def run(vapo: bool, iters=150, B=256, seed=0, epochs=2):
    torch.manual_seed(seed)
    actor, critic = mlp(2), mlp(1)
    with torch.no_grad():
        critic[-1].bias.fill_(CRITIC_INIT_BIAS)
    opt_a = torch.optim.Adam(actor.parameters(), lr=1e-3)
    opt_c = torch.optim.Adam(critic.parameters(), lr=3e-3)
    mask = torch.ones(B, L)

    def token_rewards(outcome):
        r = torch.zeros(B, L)
        r[:, -1] = outcome                         # outcome reward on the last token only
        return r

    if vapo:   # --- value pretraining: fit critic to MC returns (lambda=1) of the FIXED initial policy
        for _ in range(150):
            k = torch.randint(0, K_MAX + 1, (B,))
            toks, feats, outcome = rollout(actor, k)
            with torch.no_grad():
                v = critic(feats).squeeze(-1)
            _, ret = compute_gae(token_rewards(outcome), v, mask, lam=1.0)
            vl = 0.5 * ((critic(feats).squeeze(-1) - ret) ** 2).mean()
            opt_c.zero_grad()
            vl.backward()
            opt_c.step()

    accs = []
    for it in range(iters):
        k = torch.randint(0, K_MAX + 1, (B,))
        toks, feats, outcome = rollout(actor, k)
        with torch.no_grad():
            v_old = critic(feats).squeeze(-1)
            logp_old = logp_of(actor, feats, toks)
        r = token_rewards(outcome)
        if vapo:   # decoupled GAE: actor uses length-adaptive lambda, critic uses lambda = 1
            lam_actor = length_adaptive_lambda(mask.sum(1), alpha=2.0)  # alpha rescaled for L=32
            adv, _ = compute_gae(r, v_old, mask, lam=lam_actor)
            _, ret = compute_gae(r, v_old, mask, lam=1.0)
            eps_high = 0.28
        else:      # naive PPO: same lambda for both
            adv, ret = compute_gae(r, v_old, mask, lam=0.95)
            eps_high = 0.2
        if it == 0:   # diagnostics on the very first PPO batch
            diag = {"V(t=0) - success_rate": (v_old[:, 0].mean() - outcome.mean()).item(),
                    "mean A on failed samples, t=0": adv[outcome == 0, 0].mean().item(),
                    "mean A on successful samples, t=0": adv[outcome == 1, 0].mean().item()}
        for _ in range(epochs):
            logp = logp_of(actor, feats, toks)
            loss = ppo_actor_loss(logp, logp_old, adv, mask, 0.2, eps_high)
            if vapo:
                loss = loss + 0.1 * positive_lm_loss(logp, mask, outcome)
            opt_a.zero_grad()
            loss.backward()
            opt_a.step()
            vl = value_loss(critic(feats).squeeze(-1), v_old, ret, mask)
            opt_c.zero_grad()
            vl.backward()
            opt_c.step()
        accs.append(outcome.mean().item())
    return accs, diag


if __name__ == "__main__":
    print("Demo 1: weight of a terminal reward reaching token 0 through GAE (perfect-critic-free "
          "view: (gamma*lambda)^(T-1))")
    for lam in [0.9, 0.95, 0.99, 1.0]:
        print(f"  lambda={lam:<5} T=16: {lam**15:.3f}   T=1000: {lam**999:.2e}   T=8000: {lam**7999:.2e}")
    print("  length-adaptive lambda (alpha=0.05) for l = 100, 1000, 8000:",
          [round(x, 4) for x in length_adaptive_lambda(torch.tensor([100, 1000, 8000])).tolist()])

    # GAE sanity check: lambda=1 and zero values -> advantage = return-to-go
    a, _ = compute_gae(torch.tensor([[0., 0., 1.]]), torch.zeros(1, 3), torch.ones(1, 3), lam=1.0)
    assert torch.allclose(a, torch.ones(1, 3))
    print("\nDemo 2: 'exactly k ones in 32 tokens', reward only at the end (mean of 2 seeds)")
    for name, flag in [("naive PPO (biased critic init, lambda=0.95 shared)", False),
                       ("VAPO-style (value pretrain, decoupled/adaptive lambda, ...)", True)]:
        res = [run(flag, seed=s) for s in range(2)]
        curves = [c for c, _ in res]
        print(f"  {name}\n    first-batch diagnostics: " +
              ", ".join(f"{k} = {v:+.3f}" for k, v in res[0][1].items()))
        first = sum(sum(c[:10]) / 10 for c in curves) / 2
        mid = sum(sum(c[40:60]) / 20 for c in curves) / 2
        last = sum(sum(c[-10:]) / 10 for c in curves) / 2
        print(f"    success rate: first10 {first:.3f} | it40-60 {mid:.3f} | last10 {last:.3f}")
