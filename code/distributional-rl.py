"""
Distributional RL on a self-contained NumPy CartPole: C51, QR-DQN and IQN.

  * C51    (Bellemare, Dabney & Munos 2017): categorical distribution on a fixed support,
           target projected back onto the support, cross-entropy loss.
  * QR-DQN (Dabney, Rowland, Bellemare & Munos 2018): N quantiles at fixed fractions,
           quantile-Huber loss.
  * IQN    (Dabney, Ostrovski, Silver & Munos 2018): quantile fractions tau ~ U(0,1)
           are sampled and fed to the network through a cosine embedding.
All three share the same DQN skeleton (replay buffer, target network, epsilon-greedy).
Run:  python3 code/distributional-rl.py          (all three, ~1.5-2 min on CPU)
      python3 code/distributional-rl.py c51      (just one)
"""
import math
import random
import sys

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


class ReplayBuffer:
    def __init__(self, capacity, obs_dim):
        self.o = np.zeros((capacity, obs_dim), np.float32); self.o2 = np.zeros_like(self.o)
        self.a = np.zeros(capacity, np.int64); self.r = np.zeros(capacity, np.float32)
        self.d = np.zeros(capacity, np.float32)
        self.capacity, self.ptr, self.size = capacity, 0, 0

    def add(self, o, a, r, o2, d):
        i = self.ptr
        self.o[i], self.a[i], self.r[i], self.o2[i], self.d[i] = o, a, r, o2, d
        self.ptr, self.size = (i + 1) % self.capacity, min(self.size + 1, self.capacity)

    def sample(self, n):
        i = np.random.randint(0, self.size, n)
        return [torch.as_tensor(x[i]) for x in (self.o, self.a, self.r, self.o2, self.d)]


# =============================================================================
# C51
# =============================================================================
class C51Net(nn.Module):
    def __init__(self, obs_dim, n_actions, n_atoms=51, v_min=0.0, v_max=100.0, hidden=128):
        super().__init__()
        self.n_actions, self.n_atoms = n_actions, n_atoms
        self.register_buffer("support", torch.linspace(v_min, v_max, n_atoms))
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_actions * n_atoms))

    def log_probs(self, obs):                       # [B, A, N]
        logits = self.net(obs).view(-1, self.n_actions, self.n_atoms)
        return F.log_softmax(logits, dim=-1)

    def q(self, obs):                               # expected value: sum_i p_i z_i
        return (self.log_probs(obs).exp() * self.support).sum(-1)


def categorical_projection(next_probs, rewards, discounts, support):
    """Project the distribution of r + discount * Z(s') onto the fixed support (C51, Alg. 1).
    next_probs: [B, N] probabilities of Z(s', a*); rewards, discounts: [B]; support: [N]."""
    n = support.numel()
    v_min, v_max = support[0].item(), support[-1].item()
    dz = (v_max - v_min) / (n - 1)
    tz = (rewards[:, None] + discounts[:, None] * support[None, :]).clamp(v_min, v_max)
    b = (tz - v_min) / dz                           # fractional atom index in [0, n-1]
    lo, hi = b.floor().long(), b.ceil().long()
    m = torch.zeros_like(next_probs)
    m.scatter_add_(1, lo, next_probs * (hi.float() - b))   # share mass to lower neighbour
    m.scatter_add_(1, hi, next_probs * (b - lo.float()))   # ... and upper neighbour
    m.scatter_add_(1, lo, next_probs * (lo == hi).float()) # exact hit: all mass to that atom
    return m


def c51_loss(net, tgt, batch, gamma):
    o, a, r, o2, d = batch
    with torch.no_grad():
        next_p = tgt.log_probs(o2).exp()                          # [B, A, N]
        a_star = (next_p * tgt.support).sum(-1).argmax(1)         # greedy on expected value
        p_star = next_p[torch.arange(len(a)), a_star]             # [B, N]
        m = categorical_projection(p_star, r, gamma * (1 - d), tgt.support)
    logp = net.log_probs(o)[torch.arange(len(a)), a]              # [B, N]
    return -(m * logp).sum(-1).mean()                             # cross-entropy = KL + const


# =============================================================================
# Quantile regression (shared by QR-DQN and IQN)
# =============================================================================
def quantile_huber_loss(pred, target, taus, kappa=1.0):
    """pred: [B, N] quantile estimates at fractions taus ([B, N] or [N]); target: [B, M] samples.
    rho_tau^kappa(u) = |tau - 1{u<0}| * Huber_kappa(u) / kappa, summed over i, averaged over j."""
    u = target[:, None, :] - pred[:, :, None]                     # [B, N, M] pairwise TD errors
    huber = torch.where(u.abs() <= kappa, 0.5 * u ** 2, kappa * (u.abs() - 0.5 * kappa))
    taus = taus if taus.dim() == 2 else taus.expand(pred.shape[0], -1)
    weight = (taus[:, :, None] - (u.detach() < 0).float()).abs()
    return (weight * huber / kappa).mean(dim=2).sum(dim=1).mean()


class QRNet(nn.Module):
    def __init__(self, obs_dim, n_actions, n_quantiles=32, hidden=128):
        super().__init__()
        self.n_actions, self.n_q = n_actions, n_quantiles
        # quantile midpoints tau_hat_i = (2i + 1) / (2N), i = 0..N-1
        self.register_buffer("taus", (torch.arange(n_quantiles) + 0.5) / n_quantiles)
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_actions * n_quantiles))

    def quantiles(self, obs):                        # [B, A, N]
        return self.net(obs).view(-1, self.n_actions, self.n_q)

    def q(self, obs):                                # mean of the quantiles
        return self.quantiles(obs).mean(-1)


def qr_loss(net, tgt, batch, gamma):
    o, a, r, o2, d = batch
    idx = torch.arange(len(a))
    with torch.no_grad():
        next_q = tgt.quantiles(o2)                                # [B, A, N]
        a_star = next_q.mean(-1).argmax(1)
        target = r[:, None] + gamma * (1 - d)[:, None] * next_q[idx, a_star]   # [B, N]
    pred = net.quantiles(o)[idx, a]                               # [B, N]
    return quantile_huber_loss(pred, target, net.taus)


class IQNNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128, n_cos=64):
        super().__init__()
        self.n_actions = n_actions
        self.psi = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU())          # state embedding
        self.phi = nn.Sequential(nn.Linear(n_cos, hidden), nn.ReLU())            # tau embedding
        self.f = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_actions))
        self.register_buffer("i_pi", math.pi * torch.arange(n_cos).float())

    def quantiles(self, obs, taus):
        """obs [B, obs_dim], taus [B, K] -> quantile values Z_tau(s, .) of shape [B, K, A]."""
        cos = torch.cos(taus[:, :, None] * self.i_pi)                # [B, K, n_cos]
        h = self.psi(obs)[:, None, :] * self.phi(cos)                # Hadamard product
        return self.f(h)

    def q(self, obs, k=32):                                          # average over sampled taus
        taus = torch.rand(obs.shape[0], k)
        return self.quantiles(obs, taus).mean(1)


def iqn_loss(net, tgt, batch, gamma, n=8, n_prime=8):
    o, a, r, o2, d = batch
    B = len(a)
    with torch.no_grad():
        a_star = net.q(o2).argmax(1)                                 # (online net picks a*)
        tq = tgt.quantiles(o2, torch.rand(B, n_prime))               # [B, N', A]
        target = r[:, None] + gamma * (1 - d)[:, None] * tq[torch.arange(B), :, a_star]
    taus = torch.rand(B, n)
    pred = net.quantiles(o, taus).gather(2, a[:, None, None].expand(B, n, 1)).squeeze(2)
    return quantile_huber_loss(pred, target, taus)


# =============================================================================
# Shared DQN-style training loop
# =============================================================================
AGENTS = {"c51": (C51Net, c51_loss), "qr-dqn": (QRNet, qr_loss), "iqn": (IQNNet, iqn_loss)}


def evaluate(net, n_episodes=10, seed=123):
    env, total = CartPole(seed=seed), 0.0
    for _ in range(n_episodes):
        o, done = env.reset(), False
        while not done:
            with torch.no_grad():
                a = int(net.q(torch.as_tensor(o)[None]).argmax())
            o, r, term, trunc = env.step(a)
            total, done = total + r, term or trunc
    return total / n_episodes


def train(name, total_steps=12_000, gamma=0.99, lr=1e-3, batch_size=64, learning_starts=1_000,
          target_update=250, train_freq=2, seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    Net, loss_fn = AGENTS[name]
    env = CartPole(seed=seed)
    net, tgt = Net(env.obs_dim, env.n_actions), Net(env.obs_dim, env.n_actions)
    tgt.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=lr, eps=0.01 / 32)
    buf = ReplayBuffer(50_000, env.obs_dim)
    before, best, best_state = evaluate(net), -1.0, None
    o, ep_ret, rets = env.reset(), 0.0, []
    for step in range(total_steps):
        eps = max(0.05, 1.0 - step / 5_000)
        if random.random() < eps:
            a = random.randrange(env.n_actions)
        else:
            with torch.no_grad():
                a = int(net.q(torch.as_tensor(o)[None]).argmax())
        o2, r, term, trunc = env.step(a)
        buf.add(o, a, r, o2, float(term))
        o, ep_ret = o2, ep_ret + r
        if term or trunc:
            rets.append(ep_ret); o, ep_ret = env.reset(), 0.0
        if step >= learning_starts and step % train_freq == 0:
            loss = loss_fn(net, tgt, buf.sample(batch_size), gamma)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 10.0); opt.step()
        if step % target_update == 0:
            tgt.load_state_dict(net.state_dict())
        if (step + 1) % 3_000 == 0:
            score = evaluate(net)
            if score >= best:
                best, best_state = score, {k: v.clone() for k, v in net.state_dict().items()}
            print(f"  [{name}] step {step+1:6d}  train return (last 20)={np.mean(rets[-20:]):6.1f}"
                  f"  greedy eval={score:6.1f}")
    final = evaluate(net)
    net.load_state_dict(best_state)
    return net, before, final, evaluate(net, seed=999)


def tabular_distributional_demo():
    """Distributional Bellman operator on a 2-state chain, tabular C51-style.
    A --(r=0)--> B --(r=10 w.p. 0.3, else 0)--> end.  gamma = 0.9.
    Expected value of A is 0.9 * 3 = 2.7, but the *distribution* is bimodal: {0: 0.7, 9: 0.3}."""
    rng = np.random.default_rng(0)
    support = torch.linspace(0, 10, 11)
    Z = {s: torch.full((11,), 1 / 11) for s in "AB"}           # start from uniform distributions
    for _ in range(3000):
        r = 10.0 if rng.random() < 0.3 else 0.0                  # sample a transition from B
        m = categorical_projection(Z["B"][None], torch.tensor([r]), torch.tensor([0.0]), support)[0]
        Z["B"] = 0.98 * Z["B"] + 0.02 * m
        m = categorical_projection(Z["B"][None], torch.tensor([0.0]), torch.tensor([0.9]), support)[0]
        Z["A"] = 0.98 * Z["A"] + 0.02 * m                       # transition from A (deterministic)
    pa = Z["A"]
    print("tabular demo: Z(A) mass on atoms 0..10 =", [round(x, 2) for x in pa.tolist()])
    print(f"              E[Z(A)] = {(pa * support).sum():.2f} (true 2.7); "
          f"P(Z(A) >= 8) = {pa[8:].sum():.2f} (true 0.3)")


def _sanity_checks():
    sup = torch.linspace(0, 10, 11)
    p = torch.softmax(torch.randn(3, 11), -1)
    m = categorical_projection(p, torch.tensor([0.3, 1.0, 0.0]), torch.tensor([0.5, 0.5, 0.0]), sup)
    assert torch.allclose(m.sum(-1), torch.ones(3)), "projection must conserve probability mass"
    mean_in = 0.3 + 0.5 * (p[0] * sup).sum()
    assert torch.allclose((m[0] * sup).sum(), mean_in, atol=1e-5), "interior projection keeps mean"
    assert m[2, 0].item() == 1.0, "discount 0 & reward 0 -> all mass on atom z=0"
    # quantile regression on samples from N(3, 1): learned 50% quantile ~ 3
    theta = torch.zeros(1, 3, requires_grad=True)
    opt, taus = torch.optim.Adam([theta], lr=0.05), torch.tensor([0.1, 0.5, 0.9])
    for _ in range(1500):
        loss = quantile_huber_loss(theta, 3 + torch.randn(1, 256), taus, kappa=0.1)
        opt.zero_grad(); loss.backward(); opt.step()
    assert abs(theta[0, 1].item() - 3.0) < 0.15 and theta[0, 0] < theta[0, 1] < theta[0, 2]
    print(f"sanity checks passed (C51 projection; quantile regression on N(3,1) -> "
          f"{[round(x, 2) for x in theta[0].tolist()]}, true ~[1.72, 3.0, 4.28])")


if __name__ == "__main__":
    torch.set_num_threads(1)
    _sanity_checks()
    tabular_distributional_demo()
    names = sys.argv[1:] or list(AGENTS)
    summary = []
    for name in names:
        net, before, final, best = train(name)
        summary.append((name, before, final, best))
    print("\nalgo     greedy return before -> after (best snapshot re-tested), max 200")
    for name, before, final, best in summary:
        print(f"{name:7s}  {before:6.1f} -> {final:6.1f}   ({best:6.1f})")
