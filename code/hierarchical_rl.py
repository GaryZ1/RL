"""
Options framework (Sutton, Precup & Singh, 1999) in the classic Four-Rooms domain.

Compares
  * flat Q-learning over the 4 primitive actions, and
  * SMDP Q-learning over primitives + 8 hand-designed "hallway" options
    (each room has two options: "go to hallway A" / "go to hallway B").
Pure NumPy, runs in seconds.

Run:  python3 code/hierarchical_rl.py
"""
from collections import deque

import numpy as np

LAYOUT = """
wwwwwwwwwwwww
w     w     w
w     w     w
w           w
w     w     w
w     w     w
ww wwww     w
w     www www
w     w     w
w     w     w
w           w
w     w     w
wwwwwwwwwwwww
"""
GRID = [list(r) for r in LAYOUT.strip("\n").split("\n")]
H, W = len(GRID), len(GRID[0])
FREE = [(r, c) for r in range(H) for c in range(W) if GRID[r][c] != "w"]
IDX = {cell: i for i, cell in enumerate(FREE)}
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
GOAL = (9, 9)          # inside the bottom-right room
GAMMA, SLIP, MAX_STEPS = 0.99, 1 / 3, 1000


# ----------------------------------------------------------------------------
# Environment (as in Sutton et al. 1999: with prob 1/3 the move goes a random other way)
# ----------------------------------------------------------------------------
def move(cell, a, rng=None):
    if rng is not None and rng.random() < SLIP:
        a = rng.choice([b for b in range(4) if b != a])
    nr, nc = cell[0] + MOVES[a][0], cell[1] + MOVES[a][1]
    return (nr, nc) if GRID[nr][nc] != "w" else cell


# ----------------------------------------------------------------------------
# Rooms, hallways and hallway options
# ----------------------------------------------------------------------------
HALLWAYS = [(3, 6), (6, 2), (7, 9), (10, 6)]


def room_of(cell):
    if cell in HALLWAYS:
        return None
    r, c = cell
    return (0 if r < 7 else 2) + (0 if c < 6 else 1) if not (r == 6 and c > 6) else 1


ROOM_HALLS = {room: [h for h in HALLWAYS
                     if any(room_of(move(h, a)) == room for a in range(4))] for room in range(4)}


class Option:
    """Markov option <I, pi, beta>: initiation set = cells of `room`;
    policy = shortest path (inside the room) to `target` hallway; terminates at the
    target or whenever the agent is outside the room."""

    def __init__(self, room, target):
        self.room, self.target = room, target
        dist = {target: 0}
        q = deque([target])
        while q:  # BFS backwards from the hallway, restricted to this room
            cur = q.popleft()
            for a in range(4):
                nb = move(cur, a)
                if nb not in dist and room_of(nb) == room:
                    dist[nb] = dist[cur] + 1
                    q.append(nb)
        self.policy = {}
        for cell in dist:
            if cell == target:
                continue
            self.policy[cell] = min(range(4), key=lambda a: dist.get(move(cell, a), 1e9))

    def can_start(self, cell):
        return room_of(cell) == self.room

    def terminates(self, cell):
        return cell == self.target or room_of(cell) != self.room


OPTIONS = [Option(room, h) for room in range(4) for h in ROOM_HALLS[room]]
N_PRIM = 4


def available(cell, use_options):
    acts = list(range(N_PRIM))
    if use_options:
        acts += [N_PRIM + i for i, o in enumerate(OPTIONS) if o.can_start(cell)]
    return acts


def execute(cell, choice, rng):
    """Run a primitive or an option to termination.
    Returns next cell, discounted cumulative reward, number of steps k, done flag."""
    if choice < N_PRIM:
        nxt = move(cell, choice, rng)
        return nxt, float(nxt == GOAL), 1, nxt == GOAL
    opt, G, k = OPTIONS[choice - N_PRIM], 0.0, 0
    while True:
        cell = move(cell, opt.policy[cell], rng)
        G += (GAMMA ** k) * float(cell == GOAL)
        k += 1
        if cell == GOAL:
            return cell, G, k, True
        if opt.terminates(cell):
            return cell, G, k, False


# ----------------------------------------------------------------------------
# SMDP Q-learning:  Q(s,o) <- Q(s,o) + alpha [ R + gamma^k max_o' Q(s',o') - Q(s,o) ]
# (with options disabled this is ordinary one-step Q-learning)
# ----------------------------------------------------------------------------
def smdp_q_learning(use_options, episodes=200, alpha=0.125, eps=0.1, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((len(FREE), N_PRIM + len(OPTIONS)))
    steps_per_episode = []
    starts = [c for c in FREE if c != GOAL]
    for _ in range(episodes):
        cell, done, steps = starts[rng.integers(len(starts))], False, 0
        while not done and steps < MAX_STEPS:
            acts = available(cell, use_options)
            q = Q[IDX[cell], acts]
            choice = acts[rng.integers(len(acts))] if rng.random() < eps else acts[int(rng.choice(np.flatnonzero(q == q.max())))]
            nxt, R, k, done = execute(cell, choice, rng)
            target = R if done else R + GAMMA ** k * Q[IDX[nxt], available(nxt, use_options)].max()
            Q[IDX[cell], choice] += alpha * (target - Q[IDX[cell], choice])
            cell, steps = nxt, steps + k
        steps_per_episode.append(steps)
    return np.array(steps_per_episode), Q


if __name__ == "__main__":
    print(f"Four-rooms: {len(FREE)} free cells, {len(OPTIONS)} hallway options, goal at {GOAL}")
    n_seeds = 20
    flat = np.mean([smdp_q_learning(False, seed=s)[0] for s in range(n_seeds)], 0)
    opts = np.mean([smdp_q_learning(True, seed=s)[0] for s in range(n_seeds)], 0)
    print(f"\nMean primitive steps per episode (avg over {n_seeds} seeds)")
    print(f"{'episodes':<12}{'flat Q-learning':>18}{'primitives + options':>24}")
    for lo, hi in [(0, 10), (10, 25), (25, 50), (50, 100), (100, 200)]:
        print(f"{lo:>3}-{hi:<8}{flat[lo:hi].mean():>18.1f}{opts[lo:hi].mean():>24.1f}")
    _, Q = smdp_q_learning(True, seed=0)
    start = (2, 2)
    acts = available(start, True)
    best = acts[int(np.argmax(Q[IDX[start], acts]))]
    name = ["up", "down", "left", "right"][best] if best < N_PRIM else f"option -> hallway {OPTIONS[best - N_PRIM].target}"
    print(f"\nGreedy choice in the top-left room at {start}: {name}")
    print("Options let credit jump across whole rooms in a single backup, so early learning is faster.")
