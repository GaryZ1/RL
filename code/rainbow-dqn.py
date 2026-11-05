"""
"Rainbow-lite" DQN on a self-contained NumPy CartPole.

Combines the DQN extensions from Hessel et al. (2018), "Rainbow":
  * Double DQN         (van Hasselt et al. 2016)  -> online net picks a', target net evaluates it
  * Dueling network    (Wang et al. 2016)         -> Q = V + A - mean(A)
  * n-step returns     (Sutton 1988; used in A3C/Rainbow)
  * Prioritized replay (Schaul et al. 2016)       -> sum-tree, proportional variant, IS weights
  * Noisy Nets         (Fortunato et al. 2018)    -> optional (USE_NOISY), replaces epsilon-greedy
The distributional (C51) head is shown separately in code/distributional-rl.py.
Run:  python3 code/rainbow-dqn.py      (about 1 minute on CPU)
"""
import math
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


# ----------------------------------------------------------------------------
# Prioritized replay (proportional variant) backed by a sum tree.
# ----------------------------------------------------------------------------
class SumTree:
    """Binary tree where each parent stores the sum of its children; leaves = priorities."""
    def __init__(self, capacity):
        assert capacity & (capacity - 1) == 0, "use a power of two"
        self.capacity, self.tree = capacity, np.zeros(2 * capacity)

    def update(self, idx, priority):
        i = idx + self.capacity
        delta = priority - self.tree[i]
        while i >= 1:          # propagate the change up to the root
            self.tree[i] += delta
            i //= 2

    def total(self):
        return self.tree[1]

    def find(self, value):
        """Return the leaf index whose cumulative-sum interval contains `value`."""
        i = 1
        while i < self.capacity:
            left = 2 * i
            if value <= self.tree[left]:
                i = left
            else:
                value -= self.tree[left]
                i = left + 1
        return i - self.capacity


class PrioritizedReplay:
    def __init__(self, capacity, obs_dim, alpha=0.6, eps=1e-5):
        self.tree, self.alpha, self.eps = SumTree(capacity), alpha, eps
        self.obs = np.zeros((capacity, obs_dim), np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), np.float32)
        self.act = np.zeros(capacity, np.int64)
        self.ret = np.zeros(capacity, np.float32)    # n-step discounted reward sum
        self.disc = np.zeros(capacity, np.float32)   # gamma^k, or 0 if episode terminated
        self.capacity, self.ptr, self.size, self.max_prio = capacity, 0, 0, 1.0

    def add(self, o, a, R, o_n, disc):
        i = self.ptr
        self.obs[i], self.act[i], self.ret[i], self.next_obs[i], self.disc[i] = o, a, R, o_n, disc
        self.tree.update(i, self.max_prio ** self.alpha)   # new data gets max priority
        self.ptr, self.size = (i + 1) % self.capacity, min(self.size + 1, self.capacity)

    def sample(self, batch_size, beta):
        # stratified sampling: one sample from each of `batch_size` equal-mass segments
        seg = self.tree.total() / batch_size
        vals = (np.arange(batch_size) + np.random.rand(batch_size)) * seg
        idx = np.array([min(self.tree.find(v), self.size - 1) for v in vals])
        probs = self.tree.tree[idx + self.capacity] / self.tree.total()
        weights = (self.size * probs) ** (-beta)                # importance-sampling correction
        weights /= weights.max()
        t = torch.as_tensor
        return idx, (t(self.obs[idx]), t(self.act[idx]), t(self.ret[idx]),
                     t(self.next_obs[idx]), t(self.disc[idx]), t(weights, dtype=torch.float32))

    def update_priorities(self, idx, td_errors):
        prios = np.abs(td_errors) + self.eps
        for i, p in zip(idx, prios):
            self.tree.update(int(i), p ** self.alpha)
        self.max_prio = max(self.max_prio, float(prios.max()))


class NStepAccumulator:
    """Turns 1-step transitions into n-step ones: (s_t, a_t, sum_k g^k r_{t+k}, s_{t+n}, g^n)."""
    def __init__(self, n, gamma):
        self.n, self.gamma, self.queue = n, gamma, deque()

    def _pop(self, last_obs, terminated):
        k = len(self.queue)
        R = sum(self.gamma ** j * r for j, (_, _, r) in enumerate(self.queue))
        o, a, _ = self.queue.popleft()
        return o, a, R, last_obs, 0.0 if terminated else self.gamma ** k

    def push(self, o, a, r, o2, terminated, truncated):
        self.queue.append((o, a, r))
        out = []
        if len(self.queue) == self.n:
            out.append(self._pop(o2, terminated))
        if terminated or truncated:          # flush shorter-horizon transitions at episode end
            while self.queue:
                out.append(self._pop(o2, terminated))
        return out


# ----------------------------------------------------------------------------
# Networks: Noisy linear layer + dueling head.
# ----------------------------------------------------------------------------
class NoisyLinear(nn.Module):
    """Factorised-Gaussian noisy layer: w = mu_w + sigma_w * (f(e_out) f(e_in)^T)."""
    def __init__(self, n_in, n_out, sigma0=0.5):
        super().__init__()
        self.n_in, self.n_out = n_in, n_out
        bound = 1 / math.sqrt(n_in)
        self.w_mu = nn.Parameter(torch.empty(n_out, n_in).uniform_(-bound, bound))
        self.b_mu = nn.Parameter(torch.empty(n_out).uniform_(-bound, bound))
        self.w_sigma = nn.Parameter(torch.full((n_out, n_in), sigma0 / math.sqrt(n_in)))
        self.b_sigma = nn.Parameter(torch.full((n_out,), sigma0 / math.sqrt(n_in)))
        self.register_buffer("eps_in", torch.zeros(n_in))
        self.register_buffer("eps_out", torch.zeros(n_out))
        self.reset_noise()

    @staticmethod
    def _f(x):
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        self.eps_in.copy_(self._f(torch.randn(self.n_in)))
        self.eps_out.copy_(self._f(torch.randn(self.n_out)))

    def forward(self, x):
        w = self.w_mu + self.w_sigma * torch.outer(self.eps_out, self.eps_in)
        b = self.b_mu + self.b_sigma * self.eps_out
        return F.linear(x, w, b)


class DuelingQNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128, noisy=False):
        super().__init__()
        Lin = NoisyLinear if noisy else nn.Linear
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU())
        self.value = nn.Sequential(Lin(hidden, hidden), nn.ReLU(), Lin(hidden, 1))
        self.adv = nn.Sequential(Lin(hidden, hidden), nn.ReLU(), Lin(hidden, n_actions))

    def forward(self, x):
        h = self.body(x)
        v, a = self.value(h), self.adv(h)
        return v + a - a.mean(dim=-1, keepdim=True)   # identifiability: mean advantage = 0

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()


def evaluate(net, n_episodes=10, seed=123):
    env, total = CartPole(seed=seed), 0.0
    for _ in range(n_episodes):
        o, done = env.reset(), False
        while not done:
            with torch.no_grad():
                a = int(net(torch.as_tensor(o)).argmax())
            o, r, term, trunc = env.step(a)
            total, done = total + r, term or trunc
    return total / n_episodes


def train(total_steps=25_000, gamma=0.99, n_step=3, lr=5e-4, batch_size=64,
          buffer_size=2 ** 15, learning_starts=1_000, target_update=500,
          alpha=0.6, beta0=0.4, use_noisy=False, seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env = CartPole(seed=seed)
    net = DuelingQNet(env.obs_dim, env.n_actions, noisy=use_noisy)
    target = DuelingQNet(env.obs_dim, env.n_actions, noisy=use_noisy)
    target.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=lr, eps=1.5e-4)
    buf, nstep = PrioritizedReplay(buffer_size, env.obs_dim, alpha), NStepAccumulator(n_step, gamma)

    before = evaluate(net)
    best_score, best_state = -1.0, None
    o, ep_ret, ep_returns = env.reset(), 0.0, []
    for step in range(total_steps):
        eps = 0.0 if use_noisy else max(0.05, 1.0 - step / 8_000)  # noisy nets explore by themselves
        if random.random() < eps:
            a = random.randrange(env.n_actions)
        else:
            with torch.no_grad():
                a = int(net(torch.as_tensor(o)).argmax())
        o2, r, term, trunc = env.step(a)
        for tr in nstep.push(o, a, r, o2, term, trunc):
            buf.add(*tr)
        o, ep_ret = o2, ep_ret + r
        if term or trunc:
            ep_returns.append(ep_ret)
            o, ep_ret = env.reset(), 0.0

        if step >= learning_starts:
            beta = beta0 + (1.0 - beta0) * step / total_steps       # anneal IS exponent -> 1
            idx, (obs, act, R, nobs, disc, w) = buf.sample(batch_size, beta)
            with torch.no_grad():
                # Double DQN: argmax from the ONLINE net, value from the TARGET net
                a_star = net(nobs).argmax(dim=1, keepdim=True)
                q_next = target(nobs).gather(1, a_star).squeeze(1)
                y = R + disc * q_next                                  # n-step target
            q_sa = net(obs).gather(1, act.unsqueeze(1)).squeeze(1)
            td = y - q_sa
            loss = (w * F.smooth_l1_loss(q_sa, y, reduction="none")).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            opt.step()
            buf.update_priorities(idx, td.detach().numpy())
            if use_noisy:
                net.reset_noise(); target.reset_noise()

        if step % target_update == 0:
            target.load_state_dict(net.state_dict())
        if (step + 1) % 5_000 == 0:
            score = evaluate(net)
            if score >= best_score:
                best_score, best_state = score, {k: v.clone() for k, v in net.state_dict().items()}
            print(f"step {step+1:6d}  avg train return (last 20 eps)={np.mean(ep_returns[-20:]):6.1f}"
                  f"  greedy eval={score:6.1f}")
    final = evaluate(net)
    net.load_state_dict(best_state)
    return before, final, evaluate(net, seed=999)


def _sanity_checks():
    # n-step accumulator: 3 rewards of 1 with gamma=0.5, then termination
    acc, out = NStepAccumulator(3, 0.5), []
    for t in range(3):
        out += acc.push(t, 0, 1.0, t + 1, terminated=(t == 2), truncated=False)
    assert [round(x[2], 3) for x in out] == [1.75, 1.5, 1.0] and all(x[4] == 0.0 for x in out)
    # sum tree samples leaves proportionally to priority
    st = SumTree(4)
    for i, p in enumerate([1.0, 0.0, 3.0, 0.0]):
        st.update(i, p)
    counts = np.bincount([st.find(v) for v in np.random.rand(20000) * st.total()], minlength=4)
    assert abs(counts[2] / counts.sum() - 0.75) < 0.02 and counts[1] == counts[3] == 0
    print("sanity checks passed (n-step returns, sum-tree sampling)")


if __name__ == "__main__":
    torch.set_num_threads(1)
    _sanity_checks()
    before, final, best = train()
    print(f"\nDouble+Dueling+3-step+PER greedy return: before {before:.1f} -> after {final:.1f} (max 200)")
    print(f"Best snapshot, re-tested on new episodes: {best:.1f}")
