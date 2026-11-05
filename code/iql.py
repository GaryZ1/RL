"""
Implicit Q-Learning (IQL, Kostrikov, Nair & Levine 2021) on a tiny continuous-action
navigation task, compared with behavior cloning (BC) and filtered BC.

The offline dataset is a MIXTURE: a few noisy expert trajectories that drive around a
lava block to the goal, plus many trajectories from a "reckless" policy that heads straight
for the goal through the lava. BC copies the average (mostly reckless) behavior; IQL learns
which actions are better than the data's average (upper-expectile value) and extracts a
policy by advantage-weighted regression - without ever evaluating unseen actions.

Run:  python3 code/iql.py     (roughly 30-90 s on CPU)
"""
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)
np.random.seed(0)
torch.manual_seed(0)

GOAL = np.array([0.9, 0.9], np.float32)
STEP, HORIZON, GAMMA = 0.08, 60, 0.99


# ----------------------------------------------------------------------------
# 1) Environment: point in [0,1]^2, action a in [-1,1]^2 (velocity), lava block.
# ----------------------------------------------------------------------------
def in_lava(s):
    return 0.35 < s[0] < 0.65 and s[1] < 0.6


class LavaNavCont:
    def reset(self):
        self.t = 0
        self.s = np.array([0.1, 0.1], np.float32) + np.random.uniform(-0.05, 0.05, 2).astype(np.float32)
        return self.s.copy()

    def step(self, a):
        self.t += 1
        a = np.clip(a, -1, 1).astype(np.float32)
        self.s = np.clip(self.s + STEP * a + np.random.normal(0, 0.01, 2).astype(np.float32), 0, 1)
        if np.linalg.norm(self.s - GOAL) < 0.1:
            return self.s.copy(), 1.0, True, True
        if in_lava(self.s):
            return self.s.copy(), -1.0, True, True
        return self.s.copy(), 0.0, self.t >= HORIZON, False


def expert_action(s):
    target = np.array([0.15, 0.8], np.float32) if (s[1] < 0.75 and s[0] < 0.35) else GOAL
    d = target - s
    return d / (np.linalg.norm(d) + 1e-6)


def reckless_action(s):
    """A careless driver: heads straight for the goal - right through the lava."""
    d = GOAL - s
    return d / (np.linalg.norm(d) + 1e-6)


def collect(n_expert=10, n_reckless=90):
    env, buf, rets = LavaNavCont(), {k: [] for k in ["s", "a", "r", "s2", "term", "ep"]}, []
    for ep in range(n_expert + n_reckless):
        s, done, G, t = env.reset(), False, 0.0, 0
        while not done:
            if ep < n_expert:
                a = np.clip(expert_action(s) + np.random.normal(0, 0.3, 2), -1, 1)
            else:
                a = np.clip(reckless_action(s) + np.random.normal(0, 0.5, 2), -1, 1)
            s2, r, done, term = env.step(a)
            for k, v in zip(buf, [s, a, r, s2, float(term), ep]):
                buf[k].append(v)
            G += GAMMA ** t * r
            s, t = s2, t + 1
        rets.append(G)
    data = {k: torch.tensor(np.array(v), dtype=torch.float32) for k, v in buf.items()}
    data["ep"] = data["ep"].long()
    return data, np.array(rets)


# ----------------------------------------------------------------------------
# 2) Networks
# ----------------------------------------------------------------------------
def mlp(i, o, h=128):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))


class GaussianPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.mu = mlp(2, 2)
        self.log_std = nn.Parameter(torch.zeros(2))

    def log_prob(self, s, a):
        mu = torch.tanh(self.mu(s))
        dist = torch.distributions.Normal(mu, self.log_std.clamp(-5, 1).exp())
        return dist.log_prob(a).sum(-1)

    def act(self, s):
        return torch.tanh(self.mu(s))  # deterministic mean action at test time


# ----------------------------------------------------------------------------
# 3) The three IQL losses (these mirror the reference implementation)
# ----------------------------------------------------------------------------
def expectile_loss(diff, tau):
    """L_2^tau(u) = |tau - 1(u < 0)| * u^2   with u = Q(s,a) - V(s)."""
    weight = torch.where(diff > 0, tau, 1 - tau)
    return (weight * diff ** 2).mean()


def iql_value_loss(v, q_target_sa, tau):
    return expectile_loss(q_target_sa.detach() - v, tau)


def iql_q_loss(q1, q2, r, term, v_next):
    y = r + GAMMA * (1 - term) * v_next.detach()   # bootstrap from V(s'), never from max_a Q
    return ((q1 - y) ** 2).mean() + ((q2 - y) ** 2).mean()


def awr_policy_loss(log_prob, q_target_sa, v, beta, max_weight=100.0):
    adv = (q_target_sa - v).detach()
    w = torch.exp(beta * adv).clamp(max=max_weight)
    return -(w * log_prob).mean()


def train_iql(data, steps=6000, tau=0.8, beta=5.0, batch=256, lr=3e-4, polyak=0.005):
    Q1, Q2, V, pi = mlp(4, 1), mlp(4, 1), mlp(2, 1), GaussianPolicy()
    Q1t, Q2t = mlp(4, 1), mlp(4, 1)
    Q1t.load_state_dict(Q1.state_dict()); Q2t.load_state_dict(Q2.state_dict())
    opt_q = torch.optim.Adam(list(Q1.parameters()) + list(Q2.parameters()), lr=lr)
    opt_v = torch.optim.Adam(V.parameters(), lr=lr)
    opt_pi = torch.optim.Adam(pi.parameters(), lr=lr)
    N = data["s"].shape[0]
    for it in range(steps):
        idx = torch.randint(0, N, (batch,))
        s, a, r, s2, term = (data[k][idx] for k in ["s", "a", "r", "s2", "term"])
        sa = torch.cat([s, a], -1)
        with torch.no_grad():
            q_t = torch.min(Q1t(sa), Q2t(sa)).squeeze(-1)  # clipped double-Q target critic
        # (a) V <- expectile of Q over dataset actions
        v = V(s).squeeze(-1)
        loss_v = iql_value_loss(v, q_t, tau)
        opt_v.zero_grad(); loss_v.backward(); opt_v.step()
        # (b) Q <- r + gamma * V(s')
        with torch.no_grad():
            v_next = V(s2).squeeze(-1)
        loss_q = iql_q_loss(Q1(sa).squeeze(-1), Q2(sa).squeeze(-1), r, term, v_next)
        opt_q.zero_grad(); loss_q.backward(); opt_q.step()
        # (c) policy <- advantage-weighted regression on dataset actions
        loss_pi = awr_policy_loss(pi.log_prob(s, a), q_t, v.detach(), beta)
        opt_pi.zero_grad(); loss_pi.backward(); opt_pi.step()
        with torch.no_grad():
            for net, tgt in ((Q1, Q1t), (Q2, Q2t)):
                for p, pt in zip(net.parameters(), tgt.parameters()):
                    pt.mul_(1 - polyak).add_(polyak * p)
    return pi


def train_bc(data, steps=3000, batch=256, lr=3e-4):
    pi = GaussianPolicy()
    opt = torch.optim.Adam(pi.parameters(), lr=lr)
    N = data["s"].shape[0]
    for _ in range(steps):
        idx = torch.randint(0, N, (batch,))
        loss = -pi.log_prob(data["s"][idx], data["a"][idx]).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return pi


@torch.no_grad()
def evaluate(pi, n=100):
    env, rets, goal, lava = LavaNavCont(), [], 0, 0
    for _ in range(n):
        s, done, G, t = env.reset(), False, 0.0, 0
        while not done:
            s, r, done, _ = env.step(pi.act(torch.tensor(s)[None])[0].numpy())
            G += GAMMA ** t * r; t += 1
            goal += r > 0; lava += r < 0
        rets.append(G)
    return float(np.mean(rets)), goal / n, lava / n


if __name__ == "__main__":
    data, ep_rets = collect()
    print(f"Dataset: {data['s'].shape[0]} transitions, {len(ep_rets)} episodes "
          f"(10 noisy-expert, 90 reckless). Mean episode return = {ep_rets.mean():.3f}")

    # Filtered BC ("%BC"): clone only the transitions of episodes that reached the goal.
    good = torch.tensor(np.where(ep_rets > 0)[0])
    mask = torch.isin(data["ep"], good)
    filt = {k: v[mask] for k, v in data.items()}

    results = {"BC (all data)": evaluate(train_bc(data)),
               "filtered BC (successes)": evaluate(train_bc(filt)),
               "IQL (tau=0.8, beta=5)": evaluate(train_iql(data))}
    print(f"\n{'method':<26}{'return':>9}{'goal %':>8}{'lava %':>8}")
    for k, (ret, g, l) in results.items():
        print(f"{k:<26}{ret:>9.3f}{100*g:>8.0f}{100*l:>8.0f}")
    print("\nIQL never queries Q on actions outside the dataset, yet improves on the data by\n"
          "up-weighting actions whose Q exceeds the upper-expectile value V(s).")
