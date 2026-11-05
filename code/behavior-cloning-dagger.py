"""
Behavior Cloning (BC), noise-injected BC (DART-style) and DAgger
(Ross, Gordon & Bagnell, 2011) on a tiny lane-keeping task.

Car on a winding road.  State = (lateral offset e, heading error psi, road curvature k).
Expert = a feedback controller  u* = k - 2 e - 3 psi.
Deployment has random "gusts" that perturb the heading every step; leaving the lane
(|e| > 0.5) ends the episode as a crash.

Experiment:
  1. BC on PERFECT demos (calm day, car starts exactly centred): the data never shows
     an off-centre state, so the learner never learns how to recover -> it drifts and
     crashes once gusts push it off the centre line (compounding error).
  2. BC on demos where noise was injected into the EXPERT's actions while recording
     (the DART idea): the data now contains recoveries.
  3. DAgger: start from (1), then repeatedly roll out the LEARNER under deployment
     conditions, ask the expert to label every visited state, aggregate, retrain.

Run:  python3 code/behavior-cloning-dagger.py     (about 15-40 s on CPU)
"""
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)
np.random.seed(0)
torch.manual_seed(0)

DT, T_MAX, LANE = 0.1, 300, 0.5


class Road:
    def __init__(self, gust):
        self.gust = gust

    def reset(self):
        self.t, self.phase = 0, np.random.uniform(0, 2 * np.pi)
        self.e = np.random.normal(0, 0.02) if self.gust > 0 else 0.0  # calm demos start centred
        self.psi = 0.0
        return self.obs()

    def curvature(self, t):
        return 0.5 * np.sin(0.04 * t + self.phase)

    def obs(self):
        return np.array([self.e, self.psi, self.curvature(self.t)], np.float32)

    def step(self, u):
        u = float(np.clip(u, -1, 1))
        self.psi += DT * (u - self.curvature(self.t)) + np.random.normal(0, self.gust)
        self.e += DT * np.sin(self.psi)
        self.t += 1
        crash = abs(self.e) > LANE
        return self.obs(), crash or self.t >= T_MAX, crash


def expert(o):
    e, psi, k = o
    return float(np.clip(k - 2.0 * e - 3.0 * psi, -1, 1))


def rollout(policy, n_episodes, gust, action_noise=0.0):
    """Run `policy` and record (state, EXPERT label) for every visited state."""
    env, X, Y, crashes, steps = Road(gust), [], [], 0, 0
    for _ in range(n_episodes):
        o, done = env.reset(), False
        while not done:
            X.append(o); Y.append(expert(o)); steps += 1
            u = policy(o) + (np.random.normal(0, action_noise) if action_noise else 0.0)
            o, done, crash = env.step(u)
            crashes += crash
    return np.array(X), np.array(Y, np.float32), crashes / n_episodes, steps / n_episodes


def bc_loss(policy_net, states, expert_actions):
    """The whole of behavior cloning: supervised regression onto expert actions."""
    return ((policy_net(states) - expert_actions) ** 2).mean()


def fit(X, Y, steps=1000, lr=1e-3):
    net = nn.Sequential(nn.Linear(3, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    Xt, Yt = torch.tensor(X), torch.tensor(Y)[:, None]
    for _ in range(steps):
        idx = torch.randint(0, len(X), (256,))
        loss = bc_loss(net, Xt[idx], Yt[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    return lambda o: net(torch.tensor(o)[None]).item()


def evaluate(policy, gust=0.02, n=50):
    _, _, crash, length = rollout(policy, n, gust)
    return crash, length


if __name__ == "__main__":
    GUST, N_DEMO = 0.02, 3
    c, L = evaluate(expert)
    print(f"Expert under gusts:            crash rate {c:.2f}, mean episode length {L:5.1f}/{T_MAX}")

    X, Y, _, _ = rollout(expert, N_DEMO, gust=0.0)
    print(f"Perfect demos: {len(X)} states, offset range [{X[:,0].min():.1e}, {X[:,0].max():.1e}]")
    c, L = evaluate(fit(X, Y))
    print(f"BC on perfect demos:           crash rate {c:.2f}, mean episode length {L:5.1f}")

    Xn, Yn, _, _ = rollout(expert, N_DEMO, gust=0.0, action_noise=0.3)
    c, L = evaluate(fit(Xn, Yn))
    print(f"BC on noise-injected demos:    crash rate {c:.2f}, mean episode length {L:5.1f}")

    print("\nDAgger (beta = 0 after the first round; each round = 3 learner episodes labelled by the expert)")
    DX, DY = X, Y
    policy = fit(DX, DY)
    for it in range(1, 5):
        Xi, Yi, _, _ = rollout(policy, N_DEMO, gust=GUST)   # states the LEARNER visits
        DX, DY = np.concatenate([DX, Xi]), np.concatenate([DY, Yi])  # aggregate
        policy = fit(DX, DY)
        c, L = evaluate(policy)
        print(f"  round {it}: {len(DX):5d} labelled states   crash rate {c:.2f}, mean length {L:5.1f}")
    print("\nPerfect demonstrations contain no mistakes, so they teach nothing about recovering from\n"
          "mistakes. DAgger (or noise injection) puts the learner's own error states into the data.")
