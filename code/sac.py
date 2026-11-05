"""SAC (Haarnoja et al., 2018) with automatic temperature tuning on a NumPy Pendulum-v1.

Self-contained: needs only numpy + torch. Runs in about 1-1.5 minutes on CPU.
    python3 code/sac.py
"""
import copy
import math
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
        return self._obs(), -cost, False, self.t >= self.horizon


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


# ----------------------------------------------------------------------------------------------
# Squashed (tanh) Gaussian policy with the reparameterization trick
# ----------------------------------------------------------------------------------------------
class TanhGaussianActor(nn.Module):
    LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0

    def __init__(self, obs_dim, act_dim, max_action):
        super().__init__()
        self.net, self.max_action = mlp(obs_dim, 2 * act_dim), max_action

    def forward(self, s, deterministic=False):
        mu, log_std = self.net(s).chunk(2, dim=-1)
        log_std = log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()
        u = mu if deterministic else mu + std * torch.randn_like(mu)   # u = mu + sigma * eps  (rsample)
        a = torch.tanh(u)
        # log N(u; mu, sigma) summed over action dims
        log_prob = (-0.5 * ((u - mu) / std) ** 2 - log_std - 0.5 * math.log(2 * math.pi)).sum(-1, keepdim=True)
        # change of variables for a = tanh(u): subtract log|da/du| = log(1 - tanh(u)^2), written in a
        # numerically stable form: log(1 - tanh(u)^2) = 2 * (log 2 - u - softplus(-2u))
        log_prob -= (2 * (math.log(2) - u - F.softplus(-2 * u))).sum(-1, keepdim=True)
        # rescaling by max_action is a constant shift of log-prob; it does not affect gradients
        return self.max_action * a, log_prob


class TwinCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.q1, self.q2 = mlp(obs_dim + act_dim, 1), mlp(obs_dim + act_dim, 1)

    def forward(self, s, a):
        sa = torch.cat([s, a], -1)
        return self.q1(sa), self.q2(sa)


@torch.no_grad()
def polyak_update(net, target, tau):
    for p, p_t in zip(net.parameters(), target.parameters()):
        p_t.mul_(1 - tau).add_(tau * p)


class SAC:
    def __init__(self, obs_dim, act_dim, max_action, gamma=0.99, tau=0.005, lr=1e-3):
        self.actor = TanhGaussianActor(obs_dim, act_dim, max_action)
        self.critic = TwinCritic(obs_dim, act_dim)
        self.critic_t = copy.deepcopy(self.critic)        # SAC (v2) has no target ACTOR
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=lr)
        # automatic entropy tuning: learn log(alpha) so that E[-log pi] tracks target_entropy
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy = -float(act_dim)               # heuristic from the SAC paper: -|A|
        self.gamma, self.tau = gamma, tau

    @property
    def alpha(self):
        return self.log_alpha.exp().item()

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        a, _ = self.actor(torch.as_tensor(obs).unsqueeze(0), deterministic)
        return a[0].numpy()

    def update(self, batch):
        s, a, r, s2, d = batch
        alpha = self.log_alpha.exp().detach()

        # ---- soft Bellman backup: y = r + gamma * (min Q_targ(s',a') - alpha * log pi(a'|s')),  a' ~ pi(.|s')
        with torch.no_grad():
            a2, logp2 = self.actor(s2)
            q1_t, q2_t = self.critic_t(s2, a2)
            y = r + self.gamma * (1 - d) * (torch.min(q1_t, q2_t) - alpha * logp2)
        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.opt_c.zero_grad()
        critic_loss.backward()
        self.opt_c.step()

        # ---- policy: minimize E_s E_eps [ alpha * log pi(a~|s) - min Q(s, a~) ],  a~ = reparameterized sample
        a_new, logp = self.actor(s)
        q1_pi, q2_pi = self.critic(s, a_new)
        actor_loss = (alpha * logp - torch.min(q1_pi, q2_pi)).mean()
        self.opt_a.zero_grad()
        actor_loss.backward()
        self.opt_a.step()

        # ---- temperature: minimize E[ -alpha * (log pi + target_entropy) ]
        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.opt_alpha.zero_grad()
        alpha_loss.backward()
        self.opt_alpha.step()

        polyak_update(self.critic, self.critic_t, self.tau)
        return critic_loss.item(), actor_loss.item(), -logp.mean().item()


def evaluate(env, policy_fn, episodes=10):
    rets = []
    for _ in range(episodes):
        o, ret = env.reset(), 0.0
        while True:
            o, r, term, trunc = env.step(policy_fn(o))
            ret += r
            if term or trunc:
                break
        rets.append(ret)
    return float(np.mean(rets))


def main(total_steps=10_000, start_steps=1_000, batch_size=128, seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    env, eval_env = Pendulum(seed), Pendulum(seed + 100)
    agent = SAC(env.obs_dim, env.act_dim, env.max_torque)
    buf = ReplayBuffer(env.obs_dim, env.act_dim)

    o, ep_ret, ep_rets, t0, ent = env.reset(), 0.0, [], time.time(), float("nan")
    for t in range(1, total_steps + 1):
        # no external noise needed: the stochastic policy explores by itself
        a = np.random.uniform(-2, 2, size=1).astype(np.float32) if t <= start_steps else agent.act(o)
        o2, r, term, trunc = env.step(a)
        buf.add(o, a, r, o2, float(term))
        o, ep_ret = o2, ep_ret + r
        if term or trunc:
            ep_rets.append(ep_ret)
            o, ep_ret = env.reset(), 0.0
        if t > start_steps:
            _, _, ent = agent.update(buf.sample(batch_size))
        if t % 2000 == 0:
            print(f"step {t:6d} | last-5 train return {np.mean(ep_rets[-5:]):8.1f} | alpha {agent.alpha:.3f} "
                  f"| policy entropy {ent:5.2f} | {time.time() - t0:5.1f}s")

    det = evaluate(eval_env, lambda x: agent.act(x, deterministic=True))
    sto = evaluate(eval_env, lambda x: agent.act(x, deterministic=False))
    print(f"SAC deterministic (tanh(mean)) : avg return {det:8.1f}")
    print(f"SAC stochastic                 : avg return {sto:8.1f}   (~-150 is near-optimal; random ~-1200)")
    print(f"final temperature alpha = {agent.alpha:.4f} (started at 1.0, tuned toward target entropy "
          f"{agent.target_entropy})")


if __name__ == "__main__":
    main()
