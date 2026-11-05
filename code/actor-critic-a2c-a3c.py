"""
Advantage Actor-Critic (A2C, the synchronous variant of A3C, Mnih et al. 2016) with
Generalized Advantage Estimation (GAE, Schulman et al. 2015) on a self-contained NumPy CartPole.
Also includes a V-trace (IMPALA, Espeholt et al. 2018) target function with a sanity check.

Run:  python3 code/actor-critic-a2c-a3c.py     (about 1 minute on CPU)
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


class VecCartPole:
    """N independent CartPoles stepped in lock-step, auto-resetting finished episodes."""
    def __init__(self, n, seed=0):
        self.envs = [CartPole(seed=seed + i) for i in range(n)]
        self.ep_ret, self.finished = np.zeros(n), []

    def reset(self):
        return np.stack([e.reset() for e in self.envs])

    def step(self, actions):
        obs, rew, term, trunc, final_obs = [], [], [], [], []
        for i, (env, a) in enumerate(zip(self.envs, actions)):
            o, r, te, tr = env.step(int(a))
            self.ep_ret[i] += r
            final_obs.append(o)                    # true next state (before any reset)
            if te or tr:
                self.finished.append(self.ep_ret[i]); self.ep_ret[i] = 0.0
                o = env.reset()
            obs.append(o); rew.append(r); term.append(te); trunc.append(tr)
        return (np.stack(obs), np.array(rew, np.float32), np.array(term), np.array(trunc),
                np.stack(final_obs))


def compute_gae(rewards, values, dones, last_values, gamma=0.99, lam=0.95):
    """GAE(gamma, lambda). Inputs are [T, N] tensors (last_values: [N]); dones[t]=1 if the
    episode ended at step t (so no bootstrapping across it). Returns (advantages, returns)."""
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    gae = torch.zeros_like(last_values)
    for t in reversed(range(T)):
        next_v = last_values if t == T - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * nonterminal - values[t]      # TD error
        gae = delta + gamma * lam * nonterminal * gae                      # discounted sum of deltas
        adv[t] = gae
    return adv, adv + values


def vtrace(behaviour_logp, target_logp, rewards, values, bootstrap_value, discounts,
           rho_bar=1.0, c_bar=1.0):
    """V-trace targets v_s (IMPALA). All inputs [T] except bootstrap_value (scalar).
    discounts[t] = gamma * (1 - done_t). Returns (vs, policy-gradient advantages)."""
    rhos = torch.exp(target_logp - behaviour_logp)
    clipped_rho, cs = rhos.clamp(max=rho_bar), rhos.clamp(max=c_bar)
    values_tp1 = torch.cat([values[1:], bootstrap_value.view(1)])
    deltas = clipped_rho * (rewards + discounts * values_tp1 - values)
    acc, vs_minus_v = torch.zeros(()), torch.zeros_like(values)
    for t in reversed(range(len(rewards))):
        acc = deltas[t] + discounts[t] * cs[t] * acc
        vs_minus_v[t] = acc
    vs = vs_minus_v + values
    vs_tp1 = torch.cat([vs[1:], bootstrap_value.view(1)])
    pg_adv = clipped_rho * (rewards + discounts * vs_tp1 - values)
    return vs, pg_adv


def mlp(n_in, n_out, hidden=64):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                         nn.Linear(hidden, n_out))


def evaluate(actor, n_episodes=10, seed=123):
    env, total = CartPole(seed=seed), 0.0
    for _ in range(n_episodes):
        o, done = env.reset(), False
        while not done:
            with torch.no_grad():
                a = int(actor(torch.as_tensor(o)).argmax())
            o, r, term, trunc = env.step(a)
            total, done = total + r, term or trunc
    return total / n_episodes


def train(total_steps=150_000, n_envs=8, n_steps=16, gamma=0.99, lam=0.95, lr=1e-3,
          ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    envs = VecCartPole(n_envs, seed=seed)
    actor, critic = mlp(4, 2), mlp(4, 1)
    params = list(actor.parameters()) + list(critic.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    before = evaluate(actor)
    obs = torch.as_tensor(envs.reset())
    n_updates = total_steps // (n_envs * n_steps)
    for update in range(1, n_updates + 1):
        # ---------- collect an n_steps x n_envs rollout with the current policy ----------
        b_obs, b_act, b_rew, b_done = [], [], [], []
        for _ in range(n_steps):
            with torch.no_grad():
                a = torch.distributions.Categorical(logits=actor(obs)).sample()
            nobs, r, term, trunc, final_obs = envs.step(a.numpy())
            r = torch.as_tensor(r)
            if trunc.any():   # time-limit cut: bootstrap with V(true final state), not 0
                with torch.no_grad():
                    v_final = critic(torch.as_tensor(final_obs)).squeeze(-1)
                r = r + gamma * v_final * torch.as_tensor(trunc & ~term, dtype=torch.float32)
            b_obs.append(obs); b_act.append(a); b_rew.append(r)
            b_done.append(torch.as_tensor(term | trunc, dtype=torch.float32))
            obs = torch.as_tensor(nobs)
        b_obs, b_act = torch.stack(b_obs), torch.stack(b_act)          # [T, N, 4], [T, N]
        b_rew, b_done = torch.stack(b_rew), torch.stack(b_done)        # [T, N]

        # ---------- advantages with GAE; critic targets = GAE returns ----------
        values = critic(b_obs).squeeze(-1)                             # [T, N] (with grad)
        with torch.no_grad():
            adv, ret = compute_gae(b_rew, values.detach(), b_done,
                                   critic(obs).squeeze(-1), gamma, lam)
        dist = torch.distributions.Categorical(logits=actor(b_obs))
        policy_loss = -(dist.log_prob(b_act) * adv).mean()             # actor: grad log pi * A
        value_loss = ((values - ret) ** 2).mean()                      # critic: regression
        entropy = dist.entropy().mean()                                # exploration bonus
        loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(params, max_grad_norm)
        opt.step()
        if update % 200 == 0:
            print(f"update {update:4d}  env steps {update * n_envs * n_steps:7d}  "
                  f"avg return (last 20 eps) {np.mean(envs.finished[-20:]):6.1f}  "
                  f"entropy {entropy.item():.3f}")
    return before, np.mean(envs.finished[-20:]), evaluate(actor)


def _sanity_checks():
    # 1) GAE with lambda=1 equals Monte-Carlo return minus V; with lambda=0 equals the TD error
    T = 5
    r, v = torch.rand(T, 1), torch.rand(T, 1)
    d = torch.zeros(T, 1); d[-1] = 1
    adv1, _ = compute_gae(r, v, d, torch.zeros(1), gamma=0.9, lam=1.0)
    mc = torch.tensor([sum(0.9 ** k * r[t + k, 0].item() for k in range(T - t)) for t in range(T)])
    assert torch.allclose(adv1[:, 0], mc - v[:, 0], atol=1e-5)
    adv0, _ = compute_gae(r, v, d, torch.zeros(1), gamma=0.9, lam=0.0)
    td = r[:, 0] + 0.9 * torch.cat([v[1:, 0], torch.zeros(1)]) * (1 - d[:, 0]) - v[:, 0]
    assert torch.allclose(adv0[:, 0], td, atol=1e-6)
    # 2) on-policy V-trace (rho = c = 1) reduces to the n-step Bellman target = GAE(lambda=1) return
    lp = torch.log(torch.rand(T))
    disc = 0.9 * (1 - d[:, 0])
    vs, _ = vtrace(lp, lp, r[:, 0], v[:, 0], torch.tensor(0.0), disc)
    assert torch.allclose(vs, mc, atol=1e-5)
    print("sanity checks passed (GAE lambda=0/1 limits, on-policy V-trace == MC return)")


if __name__ == "__main__":
    torch.set_num_threads(1)
    _sanity_checks()
    before, train_ret, after = train()
    print(f"\nA2C greedy return before training: {before:.1f}")
    print(f"A2C avg training-episode return at the end: {train_ret:.1f}")
    print(f"A2C greedy return after training : {after:.1f}   (max 200)")
