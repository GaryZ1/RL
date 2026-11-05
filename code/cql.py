"""
Conservative Q-Learning (CQL) vs. naive offline DQN vs. behavior cloning
on a tiny 2-D navigation task with discrete actions and a lava pit.

Everything is self-contained: a NumPy environment, a scripted noisy behavior
policy that generates a small fixed offline dataset, and offline learners that
are trained purely from that dataset (no environment interaction while training).

Run:  python3 code/cql.py      (roughly 20-60 s on CPU)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
np.random.seed(0)
torch.manual_seed(0)

# ----------------------------------------------------------------------------
# 1) Environment: point in [0,1]^2, 8 compass actions, goal top-right,
#    lava block in the lower middle.  Start bottom-left.
# ----------------------------------------------------------------------------
ANG = np.linspace(0, 2 * np.pi, 8, endpoint=False)
ACTIONS = np.stack([np.cos(ANG), np.sin(ANG)], 1).astype(np.float32)  # 0=E,1=NE,2=N,...,7=SE
GOAL = np.array([0.9, 0.9], np.float32)
STEP, HORIZON, GAMMA = 0.08, 40, 0.99


def in_lava(s):
    return 0.35 < s[0] < 0.65 and s[1] < 0.6


class LavaNav:
    def reset(self):
        self.t = 0
        self.s = np.array([0.1, 0.1], np.float32) + np.random.uniform(-0.05, 0.05, 2).astype(np.float32)
        return self.s.copy()

    def step(self, a):
        """Returns (next_state, reward, done, terminal). terminal=True means no bootstrap."""
        self.t += 1
        noise = np.random.normal(0, 0.01, 2).astype(np.float32)
        self.s = np.clip(self.s + STEP * ACTIONS[a] + noise, 0.0, 1.0)
        if np.linalg.norm(self.s - GOAL) < 0.1:
            return self.s.copy(), 1.0, True, True
        if in_lava(self.s):
            return self.s.copy(), -1.0, True, True
        return self.s.copy(), 0.0, self.t >= HORIZON, False


# ----------------------------------------------------------------------------
# 2) Behavior policy (a cautious scripted driver) + offline dataset.
#    It goes North first, then heads to the goal; 30% of the time it wobbles to
#    a neighbouring compass direction.  It never uses W, SW, S (actions 4,5,6),
#    so the dataset says NOTHING about those actions -> out-of-distribution.
# ----------------------------------------------------------------------------
def behavior_action(s, eps=0.3):
    if s[1] < 0.75 and s[0] < 0.35:
        a = 2  # North, around the lava
    else:
        d = GOAL - s
        a = int(np.round(np.arctan2(d[1], d[0]) / (np.pi / 4))) % 8
    if np.random.rand() < eps:
        a = (a + np.random.choice([-1, 1])) % 8
    return a


def collect_dataset(n_episodes=30):
    env, buf, rets = LavaNav(), {k: [] for k in ["s", "a", "r", "s2", "term"]}, []
    for _ in range(n_episodes):
        s, done, G, t = env.reset(), False, 0.0, 0
        while not done:
            a = behavior_action(s)
            s2, r, done, term = env.step(a)
            for k, v in zip(buf, [s, a, r, s2, float(term)]):
                buf[k].append(v)
            G += GAMMA ** t * r
            s, t = s2, t + 1
        rets.append(G)
    data = {k: torch.tensor(np.array(v), dtype=torch.float32) for k, v in buf.items()}
    data["a"] = data["a"].long()
    return data, float(np.mean(rets))


# ----------------------------------------------------------------------------
# 3) The CQL loss (discrete-action CQL(H)) and the training loops.
# ----------------------------------------------------------------------------
def mlp(out_dim):
    return nn.Sequential(nn.Linear(2, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, out_dim))


def cql_loss(q_all, actions, td_target, alpha):
    """
    q_all     : [B, |A|]  Q(s, .) from the online network
    actions   : [B]       dataset actions a ~ pi_beta
    td_target : [B]       r + gamma * (1 - terminal) * max_a' Q_target(s', a')
    alpha     : weight of the conservative regulariser (alpha = 0 -> plain DQN)

    L = 1/2 E[(Q(s,a) - y)^2] + alpha * E_s[ logsumexp_a Q(s,a) - E_{a~data} Q(s,a) ]
    """
    q_data = q_all.gather(1, actions[:, None]).squeeze(1)
    bellman = 0.5 * F.mse_loss(q_data, td_target)
    conservative = (torch.logsumexp(q_all, dim=1) - q_data).mean()
    return bellman + alpha * conservative


def train_q(data, alpha, steps=3000, batch=256, lr=3e-4, tau=0.005):
    q, q_targ = mlp(8), mlp(8)
    q_targ.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    N = data["s"].shape[0]
    for _ in range(steps):
        idx = torch.randint(0, N, (batch,))
        s, a, r, s2, term = (data[k][idx] for k in ["s", "a", "r", "s2", "term"])
        with torch.no_grad():
            y = r + GAMMA * (1 - term) * q_targ(s2).max(1).values
        loss = cql_loss(q(s), a, y, alpha)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():  # Polyak-averaged target network
            for p, pt in zip(q.parameters(), q_targ.parameters()):
                pt.mul_(1 - tau).add_(tau * p)
    return q


def train_bc(data, steps=2000, batch=256, lr=1e-3):
    pi = mlp(8)
    opt = torch.optim.Adam(pi.parameters(), lr=lr)
    N = data["s"].shape[0]
    for _ in range(steps):
        idx = torch.randint(0, N, (batch,))
        loss = F.cross_entropy(pi(data["s"][idx]), data["a"][idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return pi


# ----------------------------------------------------------------------------
# 4) Evaluation: true discounted return vs. what the Q-function believes.
# ----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(net, n=100, is_q=True):
    env, rets, goal, lava, q0 = LavaNav(), [], 0, 0, []
    for _ in range(n):
        s, done, G, t = env.reset(), False, 0.0, 0
        if is_q:
            q0.append(net(torch.tensor(s)[None]).max().item())
        while not done:
            a = int(net(torch.tensor(s)[None]).argmax())
            s, r, done, _ = env.step(a)
            G += GAMMA ** t * r
            goal += r > 0
            lava += r < 0
            t += 1
        rets.append(G)
    return float(np.mean(rets)), goal / n, lava / n, (float(np.mean(q0)) if q0 else float("nan"))


if __name__ == "__main__":
    data, beh_ret = collect_dataset()
    print(f"Dataset: {data['s'].shape[0]} transitions from 30 episodes; behavior return = {beh_ret:.3f}")
    print(f"Action counts (E,NE,N,NW,W,SW,S,SE): {torch.bincount(data['a'], minlength=8).tolist()}\n")

    rows = [("naive offline DQN", evaluate(train_q(data, alpha=0.0))),
            ("CQL alpha=0.1", evaluate(train_q(data, alpha=0.1))),
            ("CQL alpha=0.5", evaluate(train_q(data, alpha=0.5))),
            ("behavior cloning", evaluate(train_bc(data), is_q=False))]
    print(f"{'method':<20}{'true return':>12}{'goal %':>8}{'lava %':>8}{'Q(s0) estimate':>16}")
    for name, (ret, g, l, q0) in rows:
        print(f"{name:<20}{ret:>12.3f}{100*g:>8.0f}{100*l:>8.0f}{q0:>16.3f}")
    print("\nCompare 'Q(s0) estimate' with 'true return': naive offline Q-learning bootstraps from\n"
          "max over never-seen actions, so its values are fiction and its greedy policy can walk into\n"
          "the lava. CQL keeps unseen actions low and stays on the data's support.")
