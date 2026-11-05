"""TD3 (Fujimoto, van Hoof & Meger, 2018) on a NumPy re-implementation of Pendulum-v1.

Self-contained: needs only numpy + torch. Runs in about a minute on CPU.
    python3 code/td3.py
"""
import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)


# ----------------------------------------------------------------------------------------------
# Environment: Pendulum-v1 in NumPy
# ----------------------------------------------------------------------------------------------
class Pendulum:
    max_speed, max_torque, dt, g, m, l, horizon = 8.0, 2.0, 0.05, 10.0, 1.0, 1.0, 200
    obs_dim, act_dim = 3, 1

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.th, self.thdot, self.t = self.rng.uniform(-np.pi, np.pi), self.rng.uniform(-1, 1), 0
        return self._obs()

    def _obs(self):
        return np.array([np.cos(self.th), np.sin(self.th), self.thdot], dtype=np.float32)

    def step(self, action):
        u = float(np.clip(action, -self.max_torque, self.max_torque)[0])
        th_norm = ((self.th + np.pi) % (2 * np.pi)) - np.pi
        cost = th_norm ** 2 + 0.1 * self.thdot ** 2 + 0.001 * u ** 2
        self.thdot += (3 * self.g / (2 * self.l) * np.sin(self.th) + 3.0 / (self.m * self.l ** 2) * u) * self.dt
        self.thdot = np.clip(self.thdot, -self.max_speed, self.max_speed)
        self.th += self.thdot * self.dt
        self.t += 1
        return self._obs(), -cost, False, self.t >= self.horizon   # obs, reward, terminated, truncated


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size=100_000):
        self.o, self.a = np.zeros((size, obs_dim), np.float32), np.zeros((size, act_dim), np.float32)
        self.r, self.o2 = np.zeros((size, 1), np.float32), np.zeros((size, obs_dim), np.float32)
        self.d = np.zeros((size, 1), np.float32)
        self.ptr, self.n, self.size = 0, 0, size

    def add(self, o, a, r, o2, d):
        i = self.ptr
        self.o[i], self.a[i], self.r[i], self.o2[i], self.d[i] = o, a, r, o2, d
        self.ptr, self.n = (i + 1) % self.size, min(self.n + 1, self.size)

    def sample(self, bs):
        idx = np.random.randint(0, self.n, size=bs)
        return [torch.as_tensor(x[idx]) for x in (self.o, self.a, self.r, self.o2, self.d)]


def mlp(inp, out, hidden=128):
    return nn.Sequential(nn.Linear(inp, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, out))


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, max_action):
        super().__init__()
        self.net, self.max_action = mlp(obs_dim, act_dim), max_action

    def forward(self, s):
        return self.max_action * torch.tanh(self.net(s))


class TwinCritic(nn.Module):
    """Two independent Q-networks in one module (trick #1: clipped double-Q)."""

    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.q1, self.q2 = mlp(obs_dim + act_dim, 1), mlp(obs_dim + act_dim, 1)

    def forward(self, s, a):
        sa = torch.cat([s, a], dim=-1)
        return self.q1(sa), self.q2(sa)


@torch.no_grad()
def polyak_update(net, target, tau):
    for p, p_t in zip(net.parameters(), target.parameters()):
        p_t.mul_(1 - tau).add_(tau * p)


class TD3:
    def __init__(self, obs_dim, act_dim, max_action, gamma=0.99, tau=0.005, lr=1e-3,
                 policy_noise=0.2, noise_clip=0.5, policy_delay=2):
        self.actor, self.critic = Actor(obs_dim, act_dim, max_action), TwinCritic(obs_dim, act_dim)
        self.actor_t, self.critic_t = copy.deepcopy(self.actor), copy.deepcopy(self.critic)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.gamma, self.tau, self.max_action = gamma, tau, max_action
        # noise values are given relative to max_action, as in the paper (0.2, clipped at 0.5)
        self.policy_noise, self.noise_clip = policy_noise * max_action, noise_clip * max_action
        self.policy_delay, self.it = policy_delay, 0

    @torch.no_grad()
    def act(self, obs, noise_std=0.0):
        a = self.actor(torch.as_tensor(obs).unsqueeze(0))[0].numpy()
        a = a + noise_std * self.max_action * np.random.randn(*a.shape)
        return np.clip(a, -self.max_action, self.max_action).astype(np.float32)

    def update(self, batch):
        self.it += 1
        s, a, r, s2, d = batch
        with torch.no_grad():
            # trick #3: target policy smoothing -- add clipped noise to the target action
            eps = (torch.randn_like(a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            a2 = (self.actor_t(s2) + eps).clamp(-self.max_action, self.max_action)
            # trick #1: clipped double-Q -- take the MIN of the two target critics
            q1_t, q2_t = self.critic_t(s2, a2)
            y = r + self.gamma * (1 - d) * torch.min(q1_t, q2_t)
        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)   # both critics regress on the SAME target
        self.opt_c.zero_grad()
        critic_loss.backward()
        self.opt_c.step()

        # trick #2: delayed policy (and target) updates -- once every `policy_delay` critic steps
        if self.it % self.policy_delay == 0:
            actor_loss = -self.critic.q1(torch.cat([s, self.actor(s)], -1)).mean()  # DPG through Q1 only
            self.opt_a.zero_grad()
            actor_loss.backward()
            self.opt_a.step()
            polyak_update(self.actor, self.actor_t, self.tau)
            polyak_update(self.critic, self.critic_t, self.tau)
        return critic_loss.item()

    @torch.no_grad()
    def q_estimate(self, s):
        """Q1(s, mu(s)) -- used below to measure over-estimation bias."""
        return self.critic.q1(torch.cat([s, self.actor(s)], -1))


def run_episode(env, policy_fn):
    o, ret, rewards, obs = env.reset(), 0.0, [], []
    while True:
        obs.append(o)
        o, r, term, trunc = env.step(policy_fn(o))
        rewards.append(r)
        if term or trunc:
            return np.array(obs), np.array(rewards)


def main(total_steps=15_000, start_steps=1_000, batch_size=128, expl_noise=0.1, seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    env, eval_env = Pendulum(seed), Pendulum(seed + 100)
    agent = TD3(env.obs_dim, env.act_dim, env.max_torque)
    buf = ReplayBuffer(env.obs_dim, env.act_dim)

    o, ep_ret, ep_rets, t0 = env.reset(), 0.0, [], time.time()
    for t in range(1, total_steps + 1):
        a = (np.random.uniform(-2, 2, size=1).astype(np.float32) if t <= start_steps
             else agent.act(o, expl_noise))
        o2, r, term, trunc = env.step(a)
        buf.add(o, a, r, o2, float(term))
        o, ep_ret = o2, ep_ret + r
        if term or trunc:
            ep_rets.append(ep_ret)
            o, ep_ret = env.reset(), 0.0
        if t > start_steps:
            agent.update(buf.sample(batch_size))
        if t % 3000 == 0:
            print(f"step {t:6d} | last-5 train return {np.mean(ep_rets[-5:]):8.1f} | {time.time() - t0:5.1f}s")

    # Evaluate and compare the critic's predicted value with the true discounted Monte-Carlo return
    rets, q_pred, q_true = [], [], []
    for _ in range(10):
        obs, rews = run_episode(eval_env, lambda x: agent.act(x, 0.0))
        rets.append(rews.sum())
        disc = np.array([np.sum(rews[i:] * 0.99 ** np.arange(len(rews) - i)) for i in range(len(rews))])
        q_pred.append(agent.q_estimate(torch.as_tensor(obs[:100])).numpy().ravel())
        q_true.append(disc[:100])   # first half only: far from the time-limit, so truncation matters less
    print(f"TD3 (greedy)   : avg return {np.mean(rets):8.1f}   (~-150 is near-optimal)")
    print(f"critic bias    : mean Q1 prediction {np.mean(q_pred):7.1f} vs true discounted return "
          f"{np.mean(q_true):7.1f}  (clipped double-Q keeps these close)")


if __name__ == "__main__":
    main()
