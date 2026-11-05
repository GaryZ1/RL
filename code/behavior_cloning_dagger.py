"""
Behavior Cloning (BC) vs. DAgger (Ross, Gordon & Bagnell, 2011) on a slippery maze.

* The expert knows the shortest path to the goal from EVERY cell (BFS).
* The floor is slippery: with probability SLIP the executed action is random.
* The learner is a small PyTorch MLP classifier on a one-hot cell encoding.

BC trains only on states the expert visited. When the learner slips off the expert's
narrow path it lands in cells it has never seen, acts badly, and errors compound
(it wanders or gets stuck against walls until the time limit).
DAgger rolls out the *learner*, asks the expert to label the states the learner
actually visits, aggregates the data, and retrains.

Run:  python3 code/behavior_cloning_dagger.py    (about 10-40 s on CPU)
"""
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
np.random.seed(0)
torch.manual_seed(0)

W, H, MAX_T, SLIP = 14, 14, 100, 0.05
MOVES = [(1, 0), (0, 1), (-1, 0), (0, -1)]
START, GOAL = (0, 0), (13, 13)


def make_walls():
    """A 'snake' maze: two wall rows with gaps on alternating sides, so the right
    action changes from corridor to corridor (Right, then Left, then Right)."""
    walls = np.zeros((W, H), bool)
    walls[:, 4] = True; walls[W - 1, 4] = False   # gap at the right end
    walls[:, 9] = True; walls[0, 9] = False       # gap at the left end
    return walls


def expert_table(walls):
    """BFS from the goal -> optimal action for every reachable safe cell."""
    dist = -np.ones((W, H), int)
    dist[GOAL] = 0
    q = deque([GOAL])
    while q:
        x, y = q.popleft()
        for dx, dy in MOVES:
            nx, ny = x + dx, y + dy
            if 0 <= nx < W and 0 <= ny < H and not walls[nx, ny] and dist[nx, ny] < 0:
                dist[nx, ny] = dist[x, y] + 1
                q.append((nx, ny))
    if dist[START] < 0:
        return None
    act = np.zeros((W, H), int)
    for x in range(W):
        for y in range(H):
            best = np.inf
            for a, (dx, dy) in enumerate(MOVES):
                nx, ny = min(max(x + dx, 0), W - 1), min(max(y + dy, 0), H - 1)
                if not walls[nx, ny] and 0 <= dist[nx, ny] < best:
                    best, act[x, y] = dist[nx, ny], a
    return act


WALLS = make_walls()
EXPERT = expert_table(WALLS)


class SlipperyMaze:
    def reset(self):
        self.pos, self.t = START, 0
        return self.pos

    def step(self, a):
        if np.random.rand() < SLIP:
            a = np.random.randint(4)
        dx, dy = MOVES[a]
        nxt = (min(max(self.pos[0] + dx, 0), W - 1), min(max(self.pos[1] + dy, 0), H - 1))
        if not WALLS[nxt]:          # bumping into a wall = stay put
            self.pos = nxt
        self.t += 1
        if self.pos == GOAL:
            return self.pos, "goal", True
        return self.pos, None, self.t >= MAX_T


def onehot(cells):
    x = torch.zeros(len(cells), W * H)
    x[torch.arange(len(cells)), torch.tensor([c[0] * H + c[1] for c in cells])] = 1.0
    return x


class Learner(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(W * H, 64), nn.ReLU(), nn.Linear(64, 4))

    def act(self, cell):
        with torch.no_grad():
            return int(self.net(onehot([cell])).argmax())


def rollout(policy, n_episodes):
    """Run `policy(cell) -> action`; return visited cells and outcome counts."""
    env, visited, outcome = SlipperyMaze(), [], {"goal": 0, None: 0}
    for _ in range(n_episodes):
        cell, done = env.reset(), False
        while not done:
            visited.append(cell)
            cell, result, done = env.step(policy(cell))
        outcome[result] += 1
    return visited, outcome


def fit(cells, n_steps=400, lr=3e-3):
    """Supervised learning: cross-entropy to the EXPERT's label for every cell (BC loss)."""
    model = Learner()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X = onehot(cells)
    y = torch.tensor([EXPERT[c] for c in cells])
    for _ in range(n_steps):
        idx = torch.randint(0, len(cells), (min(256, len(cells)),))
        loss = F.cross_entropy(model.net(X[idx]), y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def success(policy, n=300):
    """Fraction of episodes reaching the goal, and mean episode length."""
    visited, out = rollout(policy, n)
    return out["goal"] / n, len(visited) / n


def dagger(n_iters=6, eps_per_iter=5, beta0=1.0):
    """Iteration 0 rolls out the expert (beta=1) - that IS behavior cloning.
    Later iterations roll out the learner (beta=0) and query the expert on visited states."""
    data, model, history = [], None, []
    for i in range(n_iters):
        beta = beta0 if i == 0 else 0.0
        if model is None:
            mixed = lambda c: EXPERT[c]
        else:
            m = model
            mixed = lambda c, m=m: EXPERT[c] if np.random.rand() < beta else m.act(c)
        visited, _ = rollout(mixed, eps_per_iter)
        data += visited                                   # aggregate D <- D U {(s, pi*(s))}
        model = fit(data)
        g, ln = success(model.act)
        history.append((i, len(data), len(set(data)), g, ln))
    return history


if __name__ == "__main__":
    print("Map (S=start, G=goal, #=wall):")
    for y in reversed(range(H)):
        print(" ".join("S" if (x, y) == START else "G" if (x, y) == GOAL else "#" if WALLS[x, y] else "."
                       for x in range(W)))
    g, ln = success(lambda c: EXPERT[c])
    print(f"\nExpert: success {g:.2f}, mean length {ln:.1f} steps (slip prob {SLIP}, limit {MAX_T})")

    # Plain BC with a growing number of expert demos (same total budget as DAgger)
    print("\nBehavior cloning (expert demos only):")
    for n_demo in [5, 15, 30]:
        cells, _ = rollout(lambda c: EXPERT[c], n_demo)
        g, ln = success(fit(cells).act)
        print(f"  {n_demo:3d} demos  -> {len(set(cells)):3d} distinct cells   success {g:.2f}  length {ln:5.1f}")

    print("\nDAgger (5 episodes per iteration; iteration 0 == BC on 5 expert demos):")
    for i, n, uniq, g, ln in dagger():
        print(f"  iter {i}: {n:4d} labelled states, {uniq:3d} distinct cells   success {g:.2f}  length {ln:5.1f}")
    print("\nDAgger labels the states the LEARNER visits, so it learns how to recover from its own mistakes.")
