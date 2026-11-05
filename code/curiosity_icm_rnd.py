"""
Exploration bonuses on a sparse-reward gridworld: none vs. count-based vs. ICM vs. RND.

Environment: open N x N room, start bottom-left, single reward +1 at the far corner, a time
limit that makes a random walk almost never reach it.  Agent: a small PPO (shared code for
all variants).  Intrinsic reward is added to the extrinsic one:  r = r_ext + c * r_int.

  * count-based : r_int = 1 / sqrt(N(s'))                          (MBIE-EB style)
  * ICM         : r_int = || f_fwd(phi(s), a) - phi(s') ||^2       (Pathak et al., 2017)
  * RND         : r_int = || f_pred(s') - f_target(s') ||^2        (Burda et al., 2018)

Run:  python3 code/curiosity_icm_rnd.py    (roughly 1-2 min on CPU)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)

N, HORIZON, N_ENVS = 12, 50, 32
MOVES = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])


class Room:
    """Vectorised gridworld. Episode ends at the goal or after HORIZON steps."""

    def __init__(self, n):
        self.n = n

    def reset(self):
        self.pos = np.zeros((self.n, 2), int)
        self.t = np.zeros(self.n, int)
        return self.obs()

    def obs(self):
        return (self.pos / (N - 1) * 2 - 1).astype(np.float32)  # coords in [-1, 1]

    def step(self, a):
        self.pos = np.clip(self.pos + MOVES[a], 0, N - 1)
        self.t += 1
        at_goal = (self.pos == N - 1).all(1)
        r = at_goal.astype(np.float32)
        done = at_goal | (self.t >= HORIZON)
        next_obs = self.obs()                # observation BEFORE auto-reset (used for bonuses)
        cells = self.pos.copy()
        self.pos[done] = 0
        self.t[done] = 0
        return self.obs(), next_obs, cells, r, done


def mlp(i, o, h=64):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))


# ----------------------------------------------------------------------------
# Intrinsic-reward modules (each exposes .bonus(...) and .update(...))
# ----------------------------------------------------------------------------
class CountBonus:
    def __init__(self):
        self.counts = np.zeros((N, N))

    def bonus(self, obs, act, next_obs, cells):
        np.add.at(self.counts, (cells[:, 0], cells[:, 1]), 1)
        return torch.tensor(1.0 / np.sqrt(self.counts[cells[:, 0], cells[:, 1]]), dtype=torch.float32)

    def update(self, *args):
        pass


class RND:
    """Random Network Distillation: predict a fixed random network's output."""

    def __init__(self, k=16, lr=1e-3):
        self.target, self.pred = mlp(2, k), mlp(2, k)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.Adam(self.pred.parameters(), lr=lr)

    def _feat_in(self, x):
        return 3.0 * x  # sharper random features -> finer-grained novelty

    def bonus(self, obs, act, next_obs, cells):
        with torch.no_grad():
            x = self._feat_in(next_obs)
            return ((self.pred(x) - self.target(x)) ** 2).sum(-1)

    def update(self, obs, act, next_obs):
        x = self._feat_in(next_obs)
        loss = ((self.pred(x) - self.target(x)) ** 2).sum(-1).mean()
        self.opt.zero_grad(); loss.backward(); self.opt.step()


class ICM:
    """Intrinsic Curiosity Module: forward-model error in a feature space trained by an
    inverse model (so the features only encode what the agent can influence)."""

    def __init__(self, k=16, lr=1e-3, beta=0.2):
        self.phi = mlp(2, k)
        self.inv = mlp(2 * k, 4)
        self.fwd = mlp(k + 4, k)
        self.beta = beta
        params = list(self.phi.parameters()) + list(self.inv.parameters()) + list(self.fwd.parameters())
        self.opt = torch.optim.Adam(params, lr=lr)

    def _losses(self, obs, act, next_obs):
        f, f2 = self.phi(obs), self.phi(next_obs)
        a1h = F.one_hot(act, 4).float()
        fwd_err = 0.5 * ((self.fwd(torch.cat([f, a1h], -1)) - f2.detach()) ** 2).sum(-1)
        inv_loss = F.cross_entropy(self.inv(torch.cat([f, f2], -1)), act)
        return fwd_err, inv_loss

    def bonus(self, obs, act, next_obs, cells):
        with torch.no_grad():
            return self._losses(obs, act, next_obs)[0]

    def update(self, obs, act, next_obs):
        fwd_err, inv_loss = self._losses(obs, act, next_obs)
        loss = (1 - self.beta) * inv_loss + self.beta * fwd_err.mean()
        self.opt.zero_grad(); loss.backward(); self.opt.step()


# ----------------------------------------------------------------------------
# PPO agent (minimal) + training loop
# ----------------------------------------------------------------------------
class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.pi, self.v = mlp(2, 4), mlp(2, 1)


def run(variant, iters=80, coef=0.5, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    agent = Agent()
    opt = torch.optim.Adam(agent.parameters(), lr=1e-3)
    bonus_mod = {"none": None, "count": CountBonus(), "icm": ICM(), "rnd": RND()}[variant]
    env, visited, first_goal, int_std = Room(N_ENVS), np.zeros((N, N), bool), None, 1.0
    o = env.reset()
    for it in range(iters):
        O, A, LP, V, R, D, O2 = [], [], [], [], [], [], []
        for t in range(HORIZON):
            ot = torch.tensor(o)
            with torch.no_grad():
                dist = torch.distributions.Categorical(logits=agent.pi(ot))
                a = dist.sample()
            o, o2, cells, r, done = env.step(a.numpy())
            visited[cells[:, 0], cells[:, 1]] = True
            if r.any() and first_goal is None:
                first_goal = it
            r_int = torch.zeros(N_ENVS) if bonus_mod is None else bonus_mod.bonus(ot, a, torch.tensor(o2), cells)
            O.append(ot); A.append(a); LP.append(dist.log_prob(a)); V.append(agent.v(ot).squeeze(-1).detach())
            R.append((torch.tensor(r), r_int)); D.append(torch.tensor(done, dtype=torch.float32)); O2.append(torch.tensor(o2))
        r_ext = torch.stack([x[0] for x in R]); r_int = torch.stack([x[1] for x in R])
        if bonus_mod is not None:
            int_std = 0.9 * int_std + 0.1 * float(r_int.std() + 1e-8)  # running scale normaliser
            bonus_mod.update(torch.cat(O), torch.cat(A), torch.cat(O2))
        rew = r_ext + coef * r_int / int_std
        V_, D_ = torch.stack(V), torch.stack(D)
        with torch.no_grad():
            last_v = agent.v(torch.tensor(o)).squeeze(-1)
        adv, gae = torch.zeros_like(rew), torch.zeros(N_ENVS)
        for t in reversed(range(HORIZON)):  # GAE(0.95), gamma 0.99
            nv = last_v if t == HORIZON - 1 else V_[t + 1]
            delta = rew[t] + 0.99 * nv * (1 - D_[t]) - V_[t]
            gae = delta + 0.99 * 0.95 * (1 - D_[t]) * gae
            adv[t] = gae
        ret = adv + V_
        Ob, Ab, LPb = torch.cat(O), torch.cat(A), torch.cat(LP)
        advb, retb = adv.reshape(-1), ret.reshape(-1)
        advb = (advb - advb.mean()) / (advb.std() + 1e-8)
        for _ in range(4):
            for idx in torch.randperm(Ob.shape[0]).split(400):
                dist = torch.distributions.Categorical(logits=agent.pi(Ob[idx]))
                ratio = (dist.log_prob(Ab[idx]) - LPb[idx]).exp()
                pg = -torch.min(ratio * advb[idx], ratio.clamp(0.8, 1.2) * advb[idx]).mean()
                loss = pg + 0.5 * F.mse_loss(agent.v(Ob[idx]).squeeze(-1), retb[idx]) - 0.01 * dist.entropy().mean()
                opt.zero_grad(); loss.backward(); opt.step()
    # greedy evaluation of the final policy
    env_e = Room(1)
    oe, succ = env_e.reset(), 0
    for t in range(HORIZON):
        with torch.no_grad():
            a = agent.pi(torch.tensor(oe)).argmax(-1).numpy()
        oe, _, _, r, done = env_e.step(a)
        if r.any():
            succ = 1
            break
    return visited.mean(), first_goal, succ


if __name__ == "__main__":
    print(f"{N}x{N} room, goal at the far corner, horizon {HORIZON}, {N_ENVS} parallel envs, PPO")
    print(f"{'bonus':<8}{'cells visited':>15}{'first goal at iter':>20}{'greedy policy solves':>22}")
    for variant in ["none", "count", "icm", "rnd"]:
        cov, fg, succ = run(variant)
        print(f"{variant:<8}{100*cov:>14.0f}%{str(fg):>20}{('yes' if succ else 'no'):>22}")
    print("\nWithout a bonus the agent rarely stumbles onto the reward; novelty bonuses push it\n"
          "toward unvisited cells until the extrinsic reward is found and can be exploited.")
