"""PETS (Chua et al., 2018): probabilistic ensemble dynamics + CEM model-predictive control,
on a NumPy re-implementation of Pendulum-v1. Ends with an MBPO-style (Janner et al., 2019)
branched short-rollout demo that shows WHY MBPO keeps model rollouts short.

numpy + torch, ~1-2 minutes on CPU.
    python3 code/mbpo-pets.py
"""
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
np.random.seed(0)
torch.manual_seed(0)


# ----------------------------------------------------------------------------------------------
# 1. Environment: Pendulum-v1 in NumPy (+ a vectorized ground-truth step used only for diagnostics)
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


def true_step_batch(obs, act):
    """Ground-truth dynamics on a batch of observations (only for measuring model error)."""
    th, thdot, u = np.arctan2(obs[:, 1], obs[:, 0]), obs[:, 2], np.clip(act[:, 0], -2, 2)
    thdot = np.clip(thdot + (15.0 * np.sin(th) + 3.0 * u) * 0.05, -8, 8)
    th = th + thdot * 0.05
    return np.stack([np.cos(th), np.sin(th), thdot], 1).astype(np.float32)


def reward_fn(obs, act):
    """PETS assumes the reward function is KNOWN (it only learns dynamics). Torch, batched."""
    th = torch.atan2(obs[..., 1], obs[..., 0])
    return -(th ** 2 + 0.1 * obs[..., 2] ** 2 + 0.001 * act[..., 0] ** 2)


# ----------------------------------------------------------------------------------------------
# 2. Probabilistic ensemble: E Gaussian MLPs trained in parallel with batched matmuls
# ----------------------------------------------------------------------------------------------
class EnsembleLinear(nn.Module):
    def __init__(self, E, inp, out):
        super().__init__()
        self.w = nn.Parameter(torch.randn(E, inp, out) / np.sqrt(inp))
        self.b = nn.Parameter(torch.zeros(E, 1, out))

    def forward(self, x):            # x: (E, N, inp)
        return torch.baddbmm(self.b, x, self.w)


class ProbabilisticEnsemble(nn.Module):
    """Each member outputs mean and log-variance of the state CHANGE  delta = s' - s."""

    def __init__(self, obs_dim, act_dim, E=5, hidden=96):
        super().__init__()
        self.E, self.obs_dim = E, obs_dim
        self.layers = nn.ModuleList([EnsembleLinear(E, obs_dim + act_dim, hidden), EnsembleLinear(E, hidden, hidden),
                                     EnsembleLinear(E, hidden, hidden), EnsembleLinear(E, hidden, 2 * obs_dim)])
        # soft bounds on the predicted log-variance (learned), as in the PETS reference implementation
        self.max_logvar = nn.Parameter(torch.full((1, 1, obs_dim), 0.5))
        self.min_logvar = nn.Parameter(torch.full((1, 1, obs_dim), -10.0))
        self.register_buffer("in_mu", torch.zeros(obs_dim + act_dim))
        self.register_buffer("in_std", torch.ones(obs_dim + act_dim))
        self.register_buffer("out_mu", torch.zeros(obs_dim))
        self.register_buffer("out_std", torch.ones(obs_dim))

    def set_normalizer(self, x, y):
        self.in_mu.copy_(x.mean(0)); self.in_std.copy_(x.std(0) + 1e-6)
        self.out_mu.copy_(y.mean(0)); self.out_std.copy_(y.std(0) + 1e-6)

    def forward(self, obs, act):     # (E, N, .) -> normalized mean, logvar of delta
        x = (torch.cat([obs, act], -1) - self.in_mu) / self.in_std
        for layer in self.layers[:-1]:
            x = F.silu(layer(x))
        mean, logvar = self.layers[-1](x).chunk(2, -1)
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean, logvar

    def sample_next(self, obs, act):
        """Sample s' from each member's Gaussian (aleatoric noise) -- obs/act shaped (E, N, .)."""
        mean, logvar = self(obs, act)
        delta = mean + torch.randn_like(mean) * (0.5 * logvar).exp()
        return obs + delta * self.out_std + self.out_mu

    def mean_next(self, obs, act):
        mean, _ = self(obs, act)
        return obs + mean * self.out_std + self.out_mu


def train_ensemble(model, opt, X_obs, X_act, Y, epochs=40, batch=256):
    N = len(Y)
    model.set_normalizer(torch.cat([X_obs, X_act], -1), Y)
    Y_n = (Y - model.out_mu) / model.out_std
    boot = torch.randint(0, N, (model.E, N))                 # each member gets its own bootstrap resample
    for _ in range(epochs):
        perm = boot[:, torch.randperm(N)]
        for i in range(0, N, batch):
            idx = perm[:, i:i + batch]                        # (E, b)
            mean, logvar = model(X_obs[idx], X_act[idx])
            inv_var = (-logvar).exp()
            # Gaussian negative log-likelihood (up to a constant), summed over dims, averaged over data
            nll = (((mean - Y_n[idx]) ** 2) * inv_var + logvar).sum(-1).mean()
            loss = nll + 0.01 * (model.max_logvar.sum() - model.min_logvar.sum())
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():   # report held-in one-step MSE of the ensemble mean (in raw units)
        pred = model.mean_next(X_obs[None].expand(model.E, -1, -1), X_act[None].expand(model.E, -1, -1)).mean(0)
        return F.mse_loss(pred, X_obs + Y).item()


# ----------------------------------------------------------------------------------------------
# 3. CEM planner with trajectory sampling (TS-infinity: each particle sticks to one ensemble member)
# ----------------------------------------------------------------------------------------------
class CEMPlanner:
    def __init__(self, model, horizon=15, pop=200, elites=20, iters=4, alpha=0.1, max_a=2.0):
        self.m, self.H, self.pop, self.n_elite, self.iters, self.alpha, self.max_a = \
            model, horizon, pop, elites, iters, alpha, max_a
        self.mean = torch.zeros(horizon, 1)

    def reset(self):
        self.mean = torch.zeros(self.H, 1)

    @torch.no_grad()
    def evaluate(self, obs, seqs):
        """Expected return of each action sequence, averaged over E particles (one per member)."""
        E, P = self.m.E, seqs.shape[0]
        s = torch.as_tensor(obs).expand(E, P, -1).clone()     # (E, P, obs_dim): particle e uses member e
        total = torch.zeros(E, P)
        for t in range(self.H):
            a = seqs[:, t][None].expand(E, -1, -1)
            s = self.m.sample_next(s, a)
            total += reward_fn(s, a)
        return total.mean(0)                                  # average over particles = model uncertainty

    @torch.no_grad()
    def act(self, obs):
        mean, std = self.mean.clone(), torch.full((self.H, 1), self.max_a)
        for _ in range(self.iters):
            seqs = (mean + std * torch.randn(self.pop, self.H, 1)).clamp(-self.max_a, self.max_a)
            returns = self.evaluate(obs, seqs)
            elite = seqs[returns.topk(self.n_elite).indices]
            # smoothed refit of the sampling distribution to the elites
            mean = self.alpha * mean + (1 - self.alpha) * elite.mean(0)
            std = self.alpha * std + (1 - self.alpha) * elite.std(0)
        self.mean = torch.cat([mean[1:], torch.zeros(1, 1)])  # warm start: shift the plan by one step
        return mean[0].numpy()                                 # MPC: execute only the first action


def run_episode(env, policy):
    o, ret, data = env.reset(), 0.0, []
    while True:
        a = policy(o)
        o2, r, term, trunc = env.step(a)
        data.append((o, a, o2))
        ret, o = ret + r, o2
        if term or trunc:
            return ret, data


# ----------------------------------------------------------------------------------------------
# 4. MBPO ingredient: short branched rollouts from REAL states using the learned ensemble
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def branched_rollouts(model, start_obs, policy, k):
    """Start at real replay states, roll the model k steps; each transition uses a random member (TS1).
    In full MBPO these synthetic transitions fill a 'model buffer' that SAC trains on."""
    s, synthetic = start_obs.clone(), []
    for _ in range(k):
        a = policy(s)
        member = torch.randint(0, model.E, (s.shape[0],))
        nxt = model.sample_next(s[None].expand(model.E, -1, -1), a[None].expand(model.E, -1, -1))
        s2 = nxt[member, torch.arange(s.shape[0])]
        synthetic.append((s, a, reward_fn(s2, a), s2))
        s = s2
    return synthetic


def main(n_mpc_episodes=4):
    env = Pendulum(seed=0)
    model = ProbabilisticEnsemble(env.obs_dim, env.act_dim)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    planner, data, t0 = CEMPlanner(model), [], time.time()

    ret, ep = run_episode(env, lambda o: np.random.uniform(-2, 2, size=1).astype(np.float32))
    data += ep
    print(f"episode 0 (random actions)  return {ret:8.1f}")
    for i in range(1, n_mpc_episodes + 1):
        X_obs = torch.as_tensor(np.array([d[0] for d in data]))
        X_act = torch.as_tensor(np.array([d[1] for d in data]), dtype=torch.float32)
        Y = torch.as_tensor(np.array([d[2] for d in data])) - X_obs
        mse = train_ensemble(model, opt, X_obs, X_act, Y)
        planner.reset()
        ret, ep = run_episode(env, planner.act)
        data += ep
        print(f"episode {i} (PETS / CEM-MPC) return {ret:8.1f} | model 1-step MSE {mse:.5f} | "
              f"data {len(data):4d} transitions | {time.time() - t0:5.1f}s")
    print("(Pendulum-v1: random ~ -1200, good swing-up ~ -150 to -250; model-free agents need ~10k+ steps)")

    # ---- MBPO-style branched rollouts: how fast does model error compound with rollout length k?
    X_obs = torch.as_tensor(np.array([d[0] for d in data]))
    starts = X_obs[torch.randint(0, len(X_obs), (1000,))]
    rand_policy = lambda s: torch.rand(s.shape[0], 1) * 4 - 2
    torch.manual_seed(1)
    syn = branched_rollouts(model, starts, rand_policy, k=20)
    # replay the same action sequences through the TRUE dynamics to measure compounding error
    s_true, errs = starts.numpy(), []
    for (s, a, r, s2) in syn:
        s_true = true_step_batch(s_true, a.numpy())
        errs.append(np.mean(np.sum((s_true - s2.numpy()) ** 2, -1)))
    print("MBPO branched rollouts from 1000 real states (random actions): model-vs-true squared error by step k")
    print("  " + "  ".join(f"k={k}:{errs[k - 1]:.4f}" for k in (1, 2, 5, 10, 20)))
    print(f"  -> error compounds with k; MBPO therefore uses short rollouts (k ~ 1-5) branched from real "
          f"states, giving {len(syn[:5]) * len(starts)} synthetic transitions here for k=5.")


if __name__ == "__main__":
    main()
