"""
REINFORCE (Williams 1992) with and without baselines, on a self-contained NumPy CartPole.

What this script shows
  1. Gradient-variance measurement at the initial policy for three estimators:
       (a) whole-episode return G_0 for every action (original REINFORCE)
       (b) reward-to-go G_t (causality)
       (c) reward-to-go minus a baseline
  2. Training with three variants:
       "none"  : reward-to-go, no baseline
       "value" : reward-to-go minus a learned state-value baseline V(s_t)
       "rloo"  : one scalar score per episode minus the leave-one-out mean of the OTHER episodes
                 in the batch (the estimator behind RLOO / GRPO-style LLM training)
Run:  python3 code/reinforce.py      (about 30-60 s on CPU)
"""
import math

import numpy as np
import torch
import torch.nn as nn


class CartPole:
    """Classic cart-pole physics (same constants as Gym CartPole), capped at max_steps."""
    def __init__(self, max_steps=200, seed=0):
        self.gravity, self.m_cart, self.m_pole = 9.8, 1.0, 0.1
        self.length, self.force_mag, self.tau = 0.5, 10.0, 0.02
        self.x_limit, self.theta_limit = 2.4, 12 * 2 * math.pi / 360
        self.max_steps, self.rng = max_steps, np.random.default_rng(seed)
        self.obs_dim, self.n_actions = 4, 2

    def reset(self):
        self.state, self.t = self.rng.uniform(-0.05, 0.05, size=4), 0
        return self.state.astype(np.float32)

    def step(self, action):
        x, x_dot, th, th_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        cos, sin = math.cos(th), math.sin(th)
        total_m, pm_l = self.m_cart + self.m_pole, self.m_pole * self.length
        temp = (force + pm_l * th_dot ** 2 * sin) / total_m
        th_acc = (self.gravity * sin - cos * temp) / (
            self.length * (4.0 / 3.0 - self.m_pole * cos ** 2 / total_m))
        x_acc = temp - pm_l * th_acc * cos / total_m
        x, x_dot = x + self.tau * x_dot, x_dot + self.tau * x_acc
        th, th_dot = th + self.tau * th_dot, th_dot + self.tau * th_acc
        self.state, self.t = np.array([x, x_dot, th, th_dot]), self.t + 1
        terminated = bool(abs(x) > self.x_limit or abs(th) > self.theta_limit)
        return self.state.astype(np.float32), 1.0, terminated, self.t >= self.max_steps


def mlp(n_in, n_out, hidden=64):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.Tanh(), nn.Linear(hidden, n_out))


def rollout(env, policy):
    """Run one episode with the stochastic policy; return obs [T,4], actions [T], rewards [T]."""
    obs, acts, rews = [], [], []
    o, done = env.reset(), False
    while not done:
        with torch.no_grad():
            a = torch.distributions.Categorical(logits=policy(torch.as_tensor(o))).sample().item()
        o2, r, term, trunc = env.step(a)
        obs.append(o); acts.append(a); rews.append(r)
        o, done = o2, term or trunc
    return torch.as_tensor(np.array(obs)), torch.as_tensor(acts), np.array(rews, np.float32)


def reward_to_go(rews, gamma):
    """G_t = sum_{k>=t} gamma^(k-t) r_k, computed backwards."""
    g, out = 0.0, np.zeros_like(rews)
    for t in reversed(range(len(rews))):
        g = rews[t] + gamma * g
        out[t] = g
    return torch.as_tensor(out)


def pg_loss(policy, obs, acts, weights):
    """Surrogate whose gradient is the REINFORCE estimator: -E[ log pi(a|s) * weight ]."""
    logp = torch.distributions.Categorical(logits=policy(obs)).log_prob(acts)
    return -(logp * weights).sum()


def gradient_variance_demo(seed=0, n_batches=30, episodes_per_batch=5, gamma=0.99):
    torch.manual_seed(seed)
    env, policy = CartPole(seed=seed), mlp(4, 2)
    grads = {"(a) total return G_0": [], "(b) reward-to-go G_t": [], "(c) G_t - baseline": []}
    for _ in range(n_batches):
        eps = [rollout(env, policy) for _ in range(episodes_per_batch)]
        rtgs = [reward_to_go(r, gamma) for _, _, r in eps]
        # simple baseline: average reward-to-go at the same time step across the batch
        T = max(len(g) for g in rtgs)
        padded = torch.zeros(len(rtgs), T)
        mask = torch.zeros(len(rtgs), T)
        for i, g in enumerate(rtgs):
            padded[i, :len(g)], mask[i, :len(g)] = g, 1
        b_t = padded.sum(0) / mask.sum(0).clamp(min=1)
        for key in grads:
            policy.zero_grad()
            loss = 0
            for (o, a, r), g in zip(eps, rtgs):
                if key.startswith("(a)"):
                    w = torch.full_like(g, float(g[0]))
                elif key.startswith("(b)"):
                    w = g
                else:
                    w = g - b_t[:len(g)]
                loss = loss + pg_loss(policy, o, a, w) / episodes_per_batch
            loss.backward()
            grads[key].append(torch.cat([p.grad.flatten() for p in policy.parameters()]).clone())
    print("Gradient-estimator variance at the initial policy (sum of per-parameter variances):")
    base = None
    for key, gs in grads.items():
        v = torch.stack(gs).var(0).sum().item()
        base = base or v
        print(f"  {key:24s} {v:10.2f}   ({v / base:5.2f}x of (a))")


def train(variant, n_updates=60, episodes_per_update=10, gamma=0.99, lr=1e-2, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    env = CartPole(seed=seed)
    policy, value = mlp(4, 2), mlp(4, 1)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    v_opt = torch.optim.Adam(value.parameters(), lr=1e-2)
    history = []
    for it in range(n_updates):
        eps = [rollout(env, policy) for _ in range(episodes_per_update)]
        returns = np.array([r.sum() for _, _, r in eps])
        history.append(returns.mean())
        loss = 0.0
        if variant == "rloo":
            # outcome-level score (like an LLM's sequence reward): undiscounted episode return,
            # baseline = mean return of the other K-1 episodes in the batch (leave-one-out)
            K = len(eps)
            loo_baseline = (returns.sum() - returns) / (K - 1)
            for (o, a, _), R, b in zip(eps, returns, loo_baseline):
                loss = loss + pg_loss(policy, o, a, torch.full((len(a),), float(R - b)))
        else:
            all_obs = torch.cat([o for o, _, _ in eps])
            all_g = torch.cat([reward_to_go(r, gamma) for _, _, r in eps])
            if variant == "value":
                with torch.no_grad():
                    adv = all_g - value(all_obs).squeeze(-1)
                # fit the baseline V(s) ~ G_t by regression (a few gradient steps)
                for _ in range(5):
                    v_loss = ((value(all_obs).squeeze(-1) - all_g) ** 2).mean()
                    v_opt.zero_grad(); v_loss.backward(); v_opt.step()
            else:
                adv = all_g
            all_a = torch.cat([a for _, a, _ in eps])
            loss = pg_loss(policy, all_obs, all_a, adv)
        # average over episodes (the "1/m" in the Monte-Carlo estimate)
        loss = loss / episodes_per_update
        opt.zero_grad(); loss.backward(); opt.step()
    return history


if __name__ == "__main__":
    torch.set_num_threads(1)
    gradient_variance_demo()
    print("\nTraining (10 episodes per update, 60 updates = 600 episodes; max return 200)")
    for variant in ["none", "value", "rloo"]:
        finals = []
        for seed in range(2):
            h = train(variant, seed=seed)
            finals.append((np.mean(h[:5]), np.mean(h[-5:])))
        f = np.array(finals)
        print(f"  baseline={variant:6s} avg return first 5 updates {f[:, 0].mean():6.1f}  ->  "
              f"last 5 updates {f[:, 1].mean():6.1f}   (mean of 2 seeds)")
