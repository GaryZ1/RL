"""
DQN (Mnih et al. 2013 / 2015) on a self-contained NumPy CartPole.

Ingredients shown here:
  * experience replay buffer (uniform sampling)
  * target network (hard copy every `target_update` steps)
  * Huber (smooth-L1) TD loss
  * linearly decaying epsilon-greedy exploration
Run:  python3 code/dqn.py      (about 30-60 s on a laptop CPU)
"""
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Minimal CartPole (same physics/constants as the classic Barto-Sutton-Anderson
# task and Gym's CartPole). Episodes are capped at `max_steps` (truncation).
# ----------------------------------------------------------------------------
class CartPole:
    def __init__(self, max_steps=200, seed=0):
        self.gravity, self.m_cart, self.m_pole = 9.8, 1.0, 0.1
        self.length, self.force_mag, self.tau = 0.5, 10.0, 0.02  # half pole length, N, s
        self.x_limit, self.theta_limit = 2.4, 12 * 2 * math.pi / 360
        self.max_steps, self.rng = max_steps, np.random.default_rng(seed)
        self.obs_dim, self.n_actions = 4, 2

    def reset(self):
        self.state = self.rng.uniform(-0.05, 0.05, size=4)
        self.t = 0
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
        # explicit Euler integration
        x, x_dot = x + self.tau * x_dot, x_dot + self.tau * x_acc
        th, th_dot = th + self.tau * th_dot, th_dot + self.tau * th_acc
        self.state = np.array([x, x_dot, th, th_dot])
        self.t += 1
        terminated = bool(abs(x) > self.x_limit or abs(th) > self.theta_limit)
        truncated = self.t >= self.max_steps
        return self.state.astype(np.float32), 1.0, terminated, truncated


# ----------------------------------------------------------------------------
# Replay buffer: a circular array of transitions, sampled uniformly.
# ----------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity, obs_dim):
        self.obs = np.zeros((capacity, obs_dim), np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), np.float32)
        self.act = np.zeros(capacity, np.int64)
        self.rew = np.zeros(capacity, np.float32)
        self.done = np.zeros(capacity, np.float32)  # 1 only for TRUE terminals
        self.capacity, self.ptr, self.size = capacity, 0, 0

    def add(self, o, a, r, o2, d):
        i = self.ptr
        self.obs[i], self.act[i], self.rew[i], self.next_obs[i], self.done[i] = o, a, r, o2, d
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (torch.as_tensor(self.obs[idx]), torch.as_tensor(self.act[idx]),
                torch.as_tensor(self.rew[idx]), torch.as_tensor(self.next_obs[idx]),
                torch.as_tensor(self.done[idx]))


def make_q_net(obs_dim, n_actions, hidden=128):
    return nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, n_actions))


def epsilon_at(step, eps_start=1.0, eps_end=0.05, decay_steps=10_000):
    """Linear decay from eps_start to eps_end, then constant (as in the Nature paper)."""
    frac = min(1.0, step / decay_steps)
    return eps_start + frac * (eps_end - eps_start)


def evaluate(q_net, n_episodes=10, seed=123):
    """Greedy (epsilon = 0) average return."""
    env, total = CartPole(seed=seed), 0.0
    for _ in range(n_episodes):
        o, done = env.reset(), False
        while not done:
            with torch.no_grad():
                a = int(q_net(torch.as_tensor(o)).argmax())
            o, r, term, trunc = env.step(a)
            total += r
            done = term or trunc
    return total / n_episodes


def train(total_steps=30_000, gamma=0.99, lr=1e-3, batch_size=64, buffer_size=50_000,
          learning_starts=1_000, train_freq=1, target_update=1_000, seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env = CartPole(seed=seed)
    q_net = make_q_net(env.obs_dim, env.n_actions)
    target_net = make_q_net(env.obs_dim, env.n_actions)
    target_net.load_state_dict(q_net.state_dict())
    opt = torch.optim.Adam(q_net.parameters(), lr=lr)
    buf = ReplayBuffer(buffer_size, env.obs_dim)

    before = evaluate(q_net)
    best_score, best_state = -1.0, None   # DQN can oscillate: keep the best greedy snapshot
    o, ep_ret, ep_returns = env.reset(), 0.0, []
    for step in range(total_steps):
        # ---- act (epsilon-greedy) ----
        if random.random() < epsilon_at(step):
            a = random.randrange(env.n_actions)
        else:
            with torch.no_grad():
                a = int(q_net(torch.as_tensor(o)).argmax())
        o2, r, term, trunc = env.step(a)
        # store `term` (not trunc): a time-limit cut is not a real terminal state
        buf.add(o, a, r, o2, float(term))
        o, ep_ret = o2, ep_ret + r
        if term or trunc:
            ep_returns.append(ep_ret)
            o, ep_ret = env.reset(), 0.0

        # ---- learn ----
        if step >= learning_starts and step % train_freq == 0:
            obs, act, rew, nobs, done = buf.sample(batch_size)
            with torch.no_grad():  # y = r + gamma * max_a' Q_target(s', a')
                target = rew + gamma * (1.0 - done) * target_net(nobs).max(dim=1).values
            q_sa = q_net(obs).gather(1, act.unsqueeze(1)).squeeze(1)
            loss = F.smooth_l1_loss(q_sa, target)  # Huber loss (= error clipping)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
            opt.step()

        if step % target_update == 0:  # hard target-network update
            target_net.load_state_dict(q_net.state_dict())

        if (step + 1) % 5_000 == 0:
            recent = np.mean(ep_returns[-20:]) if ep_returns else 0.0
            score = evaluate(q_net)
            if score >= best_score:
                best_score = score
                best_state = {k: v.clone() for k, v in q_net.state_dict().items()}
            print(f"step {step+1:6d}  eps={epsilon_at(step):.2f}  "
                  f"avg train return (last 20 eps)={recent:6.1f}  greedy eval={score:6.1f}")

    final = evaluate(q_net)
    q_net.load_state_dict(best_state)
    return before, final, evaluate(q_net, seed=999)   # re-test best snapshot on fresh episodes


if __name__ == "__main__":
    torch.set_num_threads(1)
    before, final, best = train()
    print(f"\nGreedy eval return before training : {before:.1f}")
    print(f"Greedy eval return after training  : {final:.1f}   (max possible = 200)")
    print(f"Best snapshot, re-tested on new eps: {best:.1f}")
