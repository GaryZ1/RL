"""DDPG (Lillicrap et al., 2015) on a NumPy re-implementation of Pendulum-v1.

Self-contained: needs only numpy + torch. Runs in well under 2 minutes on CPU.
    python3 code/ddpg.py
"""
import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)  # tiny nets: one thread is fastest and plays nicely with other jobs


# ----------------------------------------------------------------------------------------------
# 1. Environment: Pendulum-v1 dynamics in ~30 lines of NumPy (same equations as Gym/Gymnasium)
# ----------------------------------------------------------------------------------------------
class Pendulum:
    max_speed, max_torque, dt, g, m, l, horizon = 8.0, 2.0, 0.05, 10.0, 1.0, 1.0, 200
    obs_dim, act_dim = 3, 1

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.th = self.rng.uniform(-np.pi, np.pi)
        self.thdot = self.rng.uniform(-1.0, 1.0)
        self.t = 0
        return self._obs()

    def _obs(self):
        return np.array([np.cos(self.th), np.sin(self.th), self.thdot], dtype=np.float32)

    def step(self, action):
        u = float(np.clip(action, -self.max_torque, self.max_torque)[0])
        th_norm = ((self.th + np.pi) % (2 * np.pi)) - np.pi            # angle in [-pi, pi)
        cost = th_norm ** 2 + 0.1 * self.thdot ** 2 + 0.001 * u ** 2   # 0 = upright & still
        self.thdot += (3 * self.g / (2 * self.l) * np.sin(self.th) + 3.0 / (self.m * self.l ** 2) * u) * self.dt
        self.thdot = np.clip(self.thdot, -self.max_speed, self.max_speed)
        self.th += self.thdot * self.dt
        self.t += 1
        # (obs, reward, terminated, truncated): Pendulum never terminates, it is only time-limited
        return self._obs(), -cost, False, self.t >= self.horizon


# ----------------------------------------------------------------------------------------------
# 2. Replay buffer
# ----------------------------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size=100_000):
        self.o = np.zeros((size, obs_dim), np.float32)
        self.a = np.zeros((size, act_dim), np.float32)
        self.r = np.zeros((size, 1), np.float32)
        self.o2 = np.zeros((size, obs_dim), np.float32)
        self.d = np.zeros((size, 1), np.float32)   # 1.0 only for TRUE terminal states
        self.ptr, self.n, self.size = 0, 0, size

    def add(self, o, a, r, o2, d):
        i = self.ptr
        self.o[i], self.a[i], self.r[i], self.o2[i], self.d[i] = o, a, r, o2, d
        self.ptr = (self.ptr + 1) % self.size
        self.n = min(self.n + 1, self.size)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.n, size=batch_size)
        return [torch.as_tensor(x[idx]) for x in (self.o, self.a, self.r, self.o2, self.d)]


# ----------------------------------------------------------------------------------------------
# 3. Networks: deterministic actor mu(s) and critic Q(s, a)
# ----------------------------------------------------------------------------------------------
def mlp(inp, out, hidden=128):
    return nn.Sequential(nn.Linear(inp, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, out))


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, max_action):
        super().__init__()
        self.net, self.max_action = mlp(obs_dim, act_dim), max_action

    def forward(self, s):
        return self.max_action * torch.tanh(self.net(s))   # squash into [-max_action, max_action]


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = mlp(obs_dim + act_dim, 1)

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))


@torch.no_grad()
def polyak_update(net, target, tau):
    """theta_target <- tau * theta + (1 - tau) * theta_target (soft target update)."""
    for p, p_t in zip(net.parameters(), target.parameters()):
        p_t.mul_(1 - tau).add_(tau * p)


# ----------------------------------------------------------------------------------------------
# 4. DDPG agent
# ----------------------------------------------------------------------------------------------
class DDPG:
    def __init__(self, obs_dim, act_dim, max_action, gamma=0.99, tau=0.005, lr_actor=1e-3, lr_critic=1e-3):
        self.actor = Actor(obs_dim, act_dim, max_action)
        self.critic = Critic(obs_dim, act_dim)
        self.actor_t, self.critic_t = copy.deepcopy(self.actor), copy.deepcopy(self.critic)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.gamma, self.tau, self.max_action = gamma, tau, max_action

    @torch.no_grad()
    def act(self, obs, noise_std=0.0):
        a = self.actor(torch.as_tensor(obs).unsqueeze(0))[0].numpy()
        if noise_std > 0:  # Gaussian exploration noise (simpler than, and as good as, Ornstein-Uhlenbeck)
            a = a + noise_std * self.max_action * np.random.randn(*a.shape)
        return np.clip(a, -self.max_action, self.max_action).astype(np.float32)

    def update(self, batch):
        s, a, r, s2, d = batch
        # ---- critic: regress Q(s,a) onto the bootstrapped target computed with TARGET networks
        with torch.no_grad():
            y = r + self.gamma * (1 - d) * self.critic_t(s2, self.actor_t(s2))
        critic_loss = F.mse_loss(self.critic(s, a), y)
        self.opt_c.zero_grad()
        critic_loss.backward()
        self.opt_c.step()

        # ---- actor: deterministic policy gradient = ascend Q(s, mu(s)) w.r.t. actor params
        actor_loss = -self.critic(s, self.actor(s)).mean()
        self.opt_a.zero_grad()
        actor_loss.backward()   # grads also flow into critic params, but we only step the actor
        self.opt_a.step()

        # ---- slowly track the online nets
        polyak_update(self.actor, self.actor_t, self.tau)
        polyak_update(self.critic, self.critic_t, self.tau)
        return critic_loss.item(), actor_loss.item()


def evaluate(env, policy_fn, episodes=5):
    rets = []
    for _ in range(episodes):
        o, done, ret = env.reset(), False, 0.0
        while not done:
            o, r, term, trunc = env.step(policy_fn(o))
            ret += r
            done = term or trunc
        rets.append(ret)
    return float(np.mean(rets))


def main(total_steps=15_000, start_steps=1_000, batch_size=128, noise_std=0.1, seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    env, eval_env = Pendulum(seed), Pendulum(seed + 100)
    agent = DDPG(env.obs_dim, env.act_dim, env.max_torque)
    buf = ReplayBuffer(env.obs_dim, env.act_dim)

    rand_ret = evaluate(eval_env, lambda o: np.random.uniform(-2, 2, size=1).astype(np.float32))
    print(f"random policy  : avg return {rand_ret:8.1f}")

    o, ep_ret, ep_rets, t0 = env.reset(), 0.0, [], time.time()
    for t in range(1, total_steps + 1):
        if t <= start_steps:     # pure random warm-up fills the buffer with diverse data
            a = np.random.uniform(-env.max_torque, env.max_torque, size=env.act_dim).astype(np.float32)
        else:
            a = agent.act(o, noise_std)
        o2, r, term, trunc = env.step(a)
        buf.add(o, a, r, o2, float(term))   # time-limit truncation is NOT a terminal state
        o, ep_ret = o2, ep_ret + r
        if term or trunc:
            ep_rets.append(ep_ret)
            o, ep_ret = env.reset(), 0.0
        if t > start_steps:
            agent.update(buf.sample(batch_size))
        if t % 3000 == 0:
            print(f"step {t:6d} | last-5 train return {np.mean(ep_rets[-5:]):8.1f} | {time.time() - t0:5.1f}s")

    final = evaluate(eval_env, lambda o: agent.act(o, 0.0), episodes=10)
    print(f"DDPG (greedy)  : avg return {final:8.1f}   (random {rand_ret:.1f}; ~-150 is near-optimal)")


if __name__ == "__main__":
    main()
