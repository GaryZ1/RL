"""A compact MuZero-style learner (Schrittwieser et al., 2020) on a tiny deterministic maze.

The agent is NEVER given the maze rules. It learns three networks
    h (representation): observation      -> latent state s^0
    g (dynamics)      : (s^k, action)    -> (s^{k+1}, predicted reward r^{k+1})
    f (prediction)    : s^k              -> (policy logits p^k, value v^k)
and plans with MCTS entirely inside the learned latent space.

Simplifications vs. the paper (all noted in the md): scalar value/reward heads with MSE instead of
categorical supports, MLPs instead of ResNets, a tiny replay buffer, no reanalyse, no prioritized replay.
Self-contained: numpy + torch, ~1-2 minutes on CPU.
    python3 code/muzero.py
"""
import math
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
rng = np.random.default_rng(0)
torch.manual_seed(0)

# ----------------------------------------------------------------------------------------------
# 1. Environment: a 5x5 maze. Reward +1 on reaching the goal G (episode ends), else 0.
#      . . . . .        row 2 is a wall with a single gap on the right, so from the top-left corner
#      . . . . .        the agent must first walk AWAY from the goal's column... it has to learn the
#      # # # # .        wall from experience (it is never told the rules) and plan around it.
#      . . . . .
#      . . . . G
# ----------------------------------------------------------------------------------------------
WALLS = {(2, 0), (2, 1), (2, 2), (2, 3)}


class Maze:
    size, n_actions, max_steps = 5, 4, 20
    walls, goal = WALLS, (4, 4)
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
    free_cells = [(r, c) for r in range(5) for c in range(5) if (r, c) not in WALLS and (r, c) != (4, 4)]

    def reset(self, start=None):
        self.pos = start if start is not None else self.free_cells[rng.integers(len(self.free_cells))]
        self.t = 0
        return self.obs()

    def obs(self):
        o = np.zeros(self.size * self.size, np.float32)
        o[self.pos[0] * self.size + self.pos[1]] = 1.0
        return o

    def step(self, a):
        r, c = self.pos[0] + self.moves[a][0], self.pos[1] + self.moves[a][1]
        if 0 <= r < self.size and 0 <= c < self.size and (r, c) not in self.walls:
            self.pos = (r, c)                                   # bumping into a wall = stay in place
        self.t += 1
        terminated = self.pos == self.goal
        return self.obs(), float(terminated), terminated, self.t >= self.max_steps

    def shortest_path(self, start):  # BFS, only used for evaluation
        frontier, dist = deque([start]), {start: 0}
        while frontier:
            p = frontier.popleft()
            for dr, dc in self.moves:
                q = (p[0] + dr, p[1] + dc)
                if 0 <= q[0] < 5 and 0 <= q[1] < 5 and q not in self.walls and q not in dist:
                    dist[q] = dist[p] + 1
                    frontier.append(q)
        return dist[self.goal]


# ----------------------------------------------------------------------------------------------
# 2. The three MuZero networks
# ----------------------------------------------------------------------------------------------
LATENT, HIDDEN, GAMMA = 32, 128, 0.9


def scale_latent(s):
    """MuZero min-max normalizes each hidden state to [0, 1] -- keeps the dynamics unroll well-scaled."""
    mn, mx = s.min(-1, keepdim=True)[0], s.max(-1, keepdim=True)[0]
    return (s - mn) / (mx - mn + 1e-5)


class MuZeroNet(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.n_actions = n_actions
        self.h = nn.Sequential(nn.Linear(obs_dim, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, LATENT))
        self.g = nn.Sequential(nn.Linear(LATENT + n_actions, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU())
        self.g_state, self.g_reward = nn.Linear(HIDDEN, LATENT), nn.Linear(HIDDEN, 1)
        self.f = nn.Sequential(nn.Linear(LATENT, HIDDEN), nn.ReLU())
        self.f_policy, self.f_value = nn.Linear(HIDDEN, n_actions), nn.Linear(HIDDEN, 1)

    def represent(self, obs):
        return scale_latent(self.h(obs))

    def dynamics(self, s, a):
        x = self.g(torch.cat([s, F.one_hot(a, self.n_actions).float()], -1))
        return scale_latent(self.g_state(x)), self.g_reward(x).squeeze(-1)

    def predict(self, s):
        x = self.f(s)
        return self.f_policy(x), self.f_value(x).squeeze(-1)


# ----------------------------------------------------------------------------------------------
# 3. MCTS in latent space (single-player: rewards on edges, discounting, min-max normalized Q)
# ----------------------------------------------------------------------------------------------
class MinMax:
    def __init__(self):
        self.lo, self.hi = float("inf"), -float("inf")

    def update(self, x):
        self.lo, self.hi = min(self.lo, x), max(self.hi, x)

    def norm(self, x):
        return (x - self.lo) / (self.hi - self.lo) if self.hi > self.lo else x


class LNode:
    __slots__ = ("prior", "N", "value_sum", "reward", "latent", "children")

    def __init__(self, prior):
        self.prior, self.N, self.value_sum, self.reward, self.latent, self.children = prior, 0, 0.0, 0.0, None, {}

    def value(self):
        return self.value_sum / self.N if self.N else 0.0


@torch.no_grad()
def expand(node, net, latent):
    logits, v = net.predict(latent)
    p = torch.softmax(logits, -1)[0].numpy()
    node.latent = latent
    for a in range(net.n_actions):
        node.children[a] = LNode(p[a])
    return float(v[0])


@torch.no_grad()
def run_mcts(net, obs, n_sims=25, c1=1.25, add_noise=True, dir_alpha=0.3, dir_eps=0.25):
    root, mm = LNode(1.0), MinMax()
    root_v = expand(root, net, net.represent(torch.as_tensor(obs)[None]))
    if add_noise:
        noise = rng.dirichlet([dir_alpha] * net.n_actions)
        for a, n in enumerate(noise):
            root.children[a].prior = (1 - dir_eps) * root.children[a].prior + dir_eps * n
    for _ in range(n_sims):
        node, path, actions = root, [root], []
        while node.children:                               # SELECT with pUCT on normalized Q
            sqrt_n = math.sqrt(node.N + 1)
            scores = [(mm.norm(ch.reward + GAMMA * ch.value()) if ch.N else 0.0)
                      + c1 * ch.prior * sqrt_n / (1 + ch.N) for ch in node.children.values()]
            a = int(np.argmax(scores))
            actions.append(a)
            node = node.children[a]
            path.append(node)
        parent = path[-2]                                  # EXPAND: one step of the LEARNED model
        latent, r = net.dynamics(parent.latent, torch.tensor([actions[-1]]))
        node.reward = float(r[0])
        v = expand(node, net, latent)
        for n in reversed(path):                           # BACK UP discounted returns
            n.value_sum += v
            n.N += 1
            mm.update(n.reward + GAMMA * n.value())
            v = n.reward + GAMMA * v
    visits = np.array([root.children[a].N for a in range(net.n_actions)], np.float32)
    return visits, root.value() if root.N else root_v


# ----------------------------------------------------------------------------------------------
# 4. Self-play / acting and target construction
# ----------------------------------------------------------------------------------------------
def play_episode(env, net, temperature=1.0, n_sims=25, start=None):
    obs, done, traj = env.reset(start), False, {"obs": [], "act": [], "rew": [], "pi": [], "v": []}
    terminated = False
    while not done:
        visits, root_value = run_mcts(net, obs, n_sims, add_noise=temperature > 0)
        if temperature > 0:
            p = visits ** (1 / temperature)
            a = int(rng.choice(len(p), p=p / p.sum()))
        else:
            a = int(np.argmax(visits))
        traj["obs"].append(obs); traj["act"].append(a); traj["pi"].append(visits / visits.sum())
        traj["v"].append(root_value)
        obs, r, terminated, truncated = env.step(a)
        traj["rew"].append(r)
        done = terminated or truncated
    with torch.no_grad():   # bootstrap value of the last state: 0 if terminal, else network estimate
        last_v = 0.0 if terminated else float(net.predict(net.represent(torch.as_tensor(obs)[None]))[1][0])
    traj["v"].append(last_v)
    traj["terminated"] = terminated
    return traj


def make_targets(traj, t, K, n_step=10, n_actions=4):
    """Targets for unrolling K steps from position t (see section 4 of the md)."""
    T, rew, V = len(traj["act"]), traj["rew"], traj["v"]
    actions, tv, vm, tr, rm, tp, pm = [], [], [], [], [], [], []
    for k in range(K + 1):
        j = t + k
        # value target: n-step bootstrapped return z_j = sum gamma^i r_{j+i+1} + gamma^(b-j) V[b]
        if j < T:
            b = min(j + n_step, T)
            z = sum(GAMMA ** (i - j) * rew[i] for i in range(j, b)) + GAMMA ** (b - j) * V[b]
            tv.append(z); vm.append(1.0); tp.append(traj["pi"][j]); pm.append(1.0)
        else:   # past the end: absorbing state (value 0) if terminal; unknown (masked) if truncated
            tv.append(0.0); vm.append(1.0 if traj["terminated"] else 0.0)
            tp.append(np.ones(n_actions) / n_actions); pm.append(0.0)
        if k > 0:  # reward target for the transition (j-1 -> j)
            tr.append(rew[j - 1] if j - 1 < T else 0.0)
            rm.append(1.0 if (j - 1 < T or traj["terminated"]) else 0.0)
        if k < K:  # action used for the k-th unroll step (random past the end of the episode)
            actions.append(traj["act"][j] if j < T else int(rng.integers(n_actions)))
    return traj["obs"][t], actions, tv, vm, tr, rm, np.array(tp), pm


def scale_grad(x, s):
    """Forward: identity. Backward: gradient multiplied by s (MuZero halves grads entering dynamics)."""
    return x * s + x.detach() * (1 - s)


def train_step(net, opt, replay, K=5, batch=128):
    samples = []
    for _ in range(batch):
        traj = replay[rng.integers(len(replay))]
        samples.append(make_targets(traj, int(rng.integers(len(traj["act"]))), K))
    obs, acts, tv, vm, tr, rm, tp, pm = [torch.as_tensor(np.array(x), dtype=torch.float32) for x in zip(*samples)]
    acts = acts.long()
    s = net.represent(obs)
    loss = 0.0
    for k in range(K + 1):
        if k > 0:
            s, r_pred = net.dynamics(s, acts[:, k - 1])
            loss = loss + (rm[:, k - 1] * (r_pred - tr[:, k - 1]) ** 2).mean() / K
            s = scale_grad(s, 0.5)
        logits, v = net.predict(s)
        w = 1.0 if k == 0 else 1.0 / K                          # scale unrolled-step losses by 1/K
        loss = loss + w * (vm[:, k] * (v - tv[:, k]) ** 2).mean()
        loss = loss + w * (pm[:, k] * -(tp[:, k] * F.log_softmax(logits, -1)).sum(-1)).mean()
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), 5.0)
    opt.step()
    return loss.item()


def evaluate(env, net, use_search=True):
    succ, ratio = 0, []
    for start in env.free_cells:
        if use_search:
            traj = play_episode(env, net, temperature=0.0, n_sims=25, start=start)
            steps, ok = len(traj["act"]), traj["terminated"]
        else:  # greedy w.r.t. the raw policy head, no planning
            obs, ok, steps = env.reset(start), False, 0
            for steps in range(1, env.max_steps + 1):
                with torch.no_grad():
                    a = int(net.predict(net.represent(torch.as_tensor(obs)[None]))[0].argmax())
                obs, _, ok, trunc = env.step(a)
                if ok or trunc:
                    break
        succ += ok
        if ok:
            ratio.append(steps / env.shortest_path(start))
    return succ / len(env.free_cells), (np.mean(ratio) if ratio else float("nan"))


def main(episodes=300, train_per_episode=8):
    env, net = Maze(), MuZeroNet(25, 4)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    replay, t0, successes = deque(maxlen=400), time.time(), deque(maxlen=50)
    sr, _ = evaluate(env, net)
    print(f"untrained MuZero + MCTS: success {sr:.0%}")
    for ep in range(1, episodes + 1):
        temp = 1.0 if ep < episodes * 0.6 else 0.5               # anneal action-selection temperature
        traj = play_episode(env, net, temperature=temp)
        replay.append(traj)
        successes.append(traj["terminated"])
        for _ in range(train_per_episode):
            loss = train_step(net, opt, replay)
        if ep % 50 == 0:
            print(f"episode {ep:4d} | self-play success (last 50) {np.mean(successes):4.0%} | loss {loss:.4f}"
                  f" | {time.time() - t0:5.1f}s")
        if ep in (25, 50):  # early in training, planning with the learned model beats the raw policy prior
            print(f"   [early check @ep{ep}] success with latent MCTS {evaluate(env, net)[0]:.0%} vs raw policy "
                  f"head {evaluate(env, net, use_search=False)[0]:.0%}")
    sr, ratio = evaluate(env, net, use_search=True)
    sr_raw, ratio_raw = evaluate(env, net, use_search=False)
    print(f"MuZero + latent MCTS : reaches goal from {sr:.0%} of {len(env.free_cells)} starts, "
          f"path length / shortest = {ratio:.2f}")
    print(f"raw policy head only : reaches goal from {sr_raw:.0%} of starts, path length / shortest = {ratio_raw:.2f}")
    # Imagination check: roll the LEARNED model forward open-loop (never touching the env)
    with torch.no_grad():
        for name, plan in [("right,down,down,down,down (reaches G)", [3, 1, 1, 1, 1]),
                           ("down x5 (blocked by the wall)       ", [1, 1, 1, 1, 1])]:
            s = net.represent(torch.as_tensor(env.reset((0, 3)))[None])
            preds = []
            for a in plan:
                s, r = net.dynamics(s, torch.tensor([a]))
                preds.append(float(r[0]))
            print(f"imagined rewards from (0,3), plan {name}: " + " ".join(f"{r:+.2f}" for r in preds))

if __name__ == "__main__":
    main()
