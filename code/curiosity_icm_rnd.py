"""
Exploration bonuses on a sparse-reward corridor: none vs. count-based vs. ICM vs. RND.

Environment: a 30 x 4 corridor. Start at the left end, a single reward +1 at the far right
corner, 80-step time limit.  A random walk essentially never reaches the reward.

Agent: Q-learning with a replay buffer and TWO value tables, as in RND's two value heads:
    Q_ext  learns the extrinsic (task) return (episodic),
    Q_int  learns the intrinsic (novelty) return (non-episodic), initialised optimistically,
           which is the tabular stand-in for a neural Q_int generalising to unseen actions,
    behaviour = epsilon-greedy on  Q_ext + BETA * Q_int ;  evaluation = greedy on Q_ext.
The intrinsic reward modules are small PyTorch networks (count-based is a table):

  * count-based : r_int = 1 / sqrt(N(s'))                          (MBIE-EB style)
  * ICM         : r_int = 1/2 || f_fwd(phi(s), a) - phi(s') ||^2   (Pathak et al., 2017)
  * RND         : r_int = || f_pred(s') - f_target(s') ||^2        (Burda et al., 2018)

Run:  python3 code/curiosity_icm_rnd.py    (roughly 30-90 s on CPU)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)

LX, LY, HORIZON = 30, 4, 80
MOVES = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
N_STATES, N_ACT = LX * LY, 4
OBS_DIM = LX + LY                     # observation = one-hot(x) ++ one-hot(y)
GOAL = (LX - 1, LY - 1)


class Corridor:
    def reset(self):
        self.pos, self.t = np.array([0, 0]), 0
        return tuple(self.pos)

    def step(self, a):
        self.pos = np.clip(self.pos + MOVES[a], 0, [LX - 1, LY - 1])
        self.t += 1
        s = tuple(self.pos)
        r = 1.0 if s == GOAL else 0.0
        return s, r, (r > 0) or self.t >= HORIZON, r > 0


def obs(cells):
    """Batch of cells -> float tensor of one-hot features [B, LX+LY]."""
    cells = np.asarray(cells).reshape(-1, 2)
    return torch.tensor(np.concatenate([np.eye(LX)[cells[:, 0]], np.eye(LY)[cells[:, 1]]], 1),
                        dtype=torch.float32)


def mlp(i, o, h=64):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o))


# ----------------------------------------------------------------------------
# Intrinsic-reward modules.  bonus(S, A, S2) -> array of CURRENT novelty for a batch;
# update(S, A, S2) trains the module; observe(s2) lets the count table count.
# ----------------------------------------------------------------------------
class NoBonus:
    def observe(self, s2):
        pass

    def bonus(self, S, A, S2):
        return np.zeros(len(S2))

    def update(self, S, A, S2):
        pass


class CountBonus(NoBonus):
    def __init__(self):
        self.n = np.zeros((LX, LY))

    def observe(self, s2):
        self.n[s2] += 1

    def bonus(self, S, A, S2):
        S2 = np.asarray(S2)
        return 1.0 / np.sqrt(np.maximum(self.n[S2[:, 0], S2[:, 1]], 1))


class RND(NoBonus):
    """A fixed, randomly initialised target network f and a trained predictor f_hat.
    Prediction error is high on rarely-seen inputs and decays as they become familiar."""

    def __init__(self, k=16, lr=1e-3):
        self.target, self.pred = mlp(OBS_DIM, k), mlp(OBS_DIM, k)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.Adam(self.pred.parameters(), lr=lr)

    def error(self, S2):
        x = obs(S2)
        return ((self.pred(x) - self.target(x)) ** 2).sum(-1)

    def bonus(self, S, A, S2):
        with torch.no_grad():
            return self.error(S2).numpy()

    def update(self, S, A, S2):
        loss = self.error(S2).mean()
        self.opt.zero_grad(); loss.backward(); self.opt.step()


class ICM(NoBonus):
    """Features phi trained by an inverse model (predict a from s, s'); the curiosity
    reward is the error of a forward model that predicts phi(s') from phi(s) and a."""

    def __init__(self, k=16, lr=1e-3, beta=0.2):
        self.phi, self.inv, self.fwd = mlp(OBS_DIM, k), mlp(2 * k, N_ACT), mlp(k + N_ACT, k)
        self.beta = beta
        params = [*self.phi.parameters(), *self.inv.parameters(), *self.fwd.parameters()]
        self.opt = torch.optim.Adam(params, lr=lr)

    def losses(self, S, A, S2):
        f, f2 = self.phi(obs(S)), self.phi(obs(S2))
        a1h = F.one_hot(torch.as_tensor(A), N_ACT).float()
        fwd_err = 0.5 * ((self.fwd(torch.cat([f, a1h], -1)) - f2.detach()) ** 2).sum(-1)
        inv_loss = F.cross_entropy(self.inv(torch.cat([f, f2], -1)), torch.as_tensor(A))
        return fwd_err, inv_loss

    def bonus(self, S, A, S2):
        with torch.no_grad():
            return self.losses(S, A, S2)[0].numpy()

    def update(self, S, A, S2):
        fwd_err, inv_loss = self.losses(S, A, S2)
        loss = (1 - self.beta) * inv_loss + self.beta * fwd_err.mean()
        self.opt.zero_grad(); loss.backward(); self.opt.step()


# ----------------------------------------------------------------------------
# Agent: two-headed Q-learning with replay
# ----------------------------------------------------------------------------
def run(bonus_mod, episodes=250, beta=1.0, eps=0.1, alpha=0.5, gamma=0.98, replay_k=8, seed=0, q_int_init=None):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if q_int_init is None:
        q_int_init = 1.0 / (1 - gamma)  # optimistic: 'every untried action is maximally novel forever'
    Qe, Qi = np.zeros((LX, LY, N_ACT)), np.full((LX, LY, N_ACT), q_int_init)
    buf, visited, first_goal, int_scale = [], np.zeros((LX, LY), bool), None, None
    env = Corridor()
    for ep in range(episodes):
        s, done = env.reset(), False
        while not done:
            q = Qe[s] + beta * Qi[s]
            a = int(rng.integers(N_ACT)) if rng.random() < eps else int(rng.choice(np.flatnonzero(q == q.max())))
            s2, r, done, term = env.step(a)
            bonus_mod.observe(s2)
            buf.append((s, a, r, s2, term))
            visited[s2] = True
            if r > 0 and first_goal is None:
                first_goal = ep
            # replay: online transition + a few random old ones (fast reward propagation).
            # Intrinsic rewards are recomputed NOW, so familiar states stop paying a bonus.
            batch = [buf[-1]] + [buf[i] for i in rng.integers(len(buf), size=replay_k)]
            S, A, R, S2, Tm = zip(*batch)
            r_int = bonus_mod.bonus(list(S), list(A), list(S2))
            if int_scale is None and r_int.max() > 0:
                int_scale = float(r_int.max())     # fixed normaliser: first bonus seen ~ 1
            r_int = r_int / int_scale if int_scale else r_int
            for (bs, ba, br, bs2, bt), bri in zip(batch, r_int):
                Qe[bs][ba] += alpha * (br + gamma * (1 - bt) * Qe[bs2].max() - Qe[bs][ba])
                Qi[bs][ba] += alpha * (bri + gamma * Qi[bs2].max() - Qi[bs][ba])  # non-episodic intrinsic
            if len(buf) % 16 == 0:  # train the bonus network on recent transitions
                S, A, _, S2, _ = zip(*buf[-64:])
                bonus_mod.update(list(S), list(A), list(S2))
            s = s2
    # greedy evaluation on the EXTRINSIC value only
    s, done, solved, steps = env.reset(), False, False, 0
    while not done:
        s, r, done, term = env.step(int(Qe[s].argmax()))
        solved, steps = solved or term, steps + 1
    return visited.mean(), first_goal, solved, steps


if __name__ == "__main__":
    print(f"{LX}x{LY} corridor, reward only at the far corner, horizon {HORIZON}, 250 episodes")
    print(f"{'bonus':<19}{'cells visited':>15}{'first reward (episode)':>24}{'greedy solves':>15}{'steps':>7}")
    variants = [("none (eps-greedy)", NoBonus(), 0.0),       # beta = 0: Q_int ignored
                ("optimism only", NoBonus(), 1.0),           # optimistic Q_int, zero bonus
                ("count", CountBonus(), 1.0),
                ("ICM", ICM(), 1.0),
                ("RND", RND(), 1.0)]
    for name, mod, beta in variants:
        cov, fg, solved, steps = run(mod, beta=beta)
        print(f"{name:<19}{100*cov:>14.0f}%{str(fg):>24}{('yes' if solved else 'no'):>15}{steps:>7}")
    print("\nEpsilon-greedy dithering almost never finds the reward. Directed exploration (optimism,\n"
          "counts, RND) sweeps the corridor; once the reward is found, the separate extrinsic Q-table\n"
          f"lets the greedy policy exploit it (optimal path = {LX - 1 + LY - 1} steps). In a small table,\n"
          "optimism ~ counts is enough; learned bonuses (RND/ICM) exist for spaces too big to count.\n"
          "ICM's forward model generalises across this simple deterministic grid, so its bonus fades fast.")
