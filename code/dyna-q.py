"""
Dyna-Q, Dyna-Q+ and prioritized sweeping (Sutton 1990; Moore & Atkeson 1993;
Sutton & Barto, 2nd ed., Ch. 8).

  1. Dyna maze (Example 8.1): steps per episode for n = 0, 5, 50 planning steps.
  2. Shortcut maze (Example 8.3): Dyna-Q vs Dyna-Q+ when a shorter path opens up.
  3. Prioritized sweeping vs Dyna-Q: total value backups until the greedy path is near-optimal.

Pure NumPy (+ heapq from the standard library).   Run:  python3 code/dyna-q.py
"""
import heapq
from collections import deque, defaultdict
import numpy as np

MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]               # up, right, down, left


class Maze:
    """Deterministic grid maze.  Reward +1 on reaching the goal (episode ends), else 0."""

    def __init__(self, rows, cols, start, goal, walls):
        self.rows, self.cols = rows, cols
        self.start, self.goal = start, goal
        self.walls = set(walls)
        self.n_actions = 4

    def step(self, s, a):
        r, c = s[0] + MOVES[a][0], s[1] + MOVES[a][1]
        if not (0 <= r < self.rows and 0 <= c < self.cols) or (r, c) in self.walls:
            r, c = s                                     # blocked: stay put
        s2 = (r, c)
        return s2, (1.0 if s2 == self.goal else 0.0), s2 == self.goal

    def shortest_path(self):
        """BFS length of the optimal path (to check learned policies)."""
        dist, q = {self.start: 0}, deque([self.start])
        while q:
            s = q.popleft()
            for a in range(4):
                s2, _, _ = self.step(s, a)
                if s2 not in dist:
                    dist[s2] = dist[s] + 1
                    q.append(s2)
        return dist[self.goal]


def dyna_maze():
    walls = [(1, 2), (2, 2), (3, 2), (4, 5), (0, 7), (1, 7), (2, 7)]
    return Maze(6, 9, start=(2, 0), goal=(0, 8), walls=walls)


def argmax_tie(q, rng):
    return int(rng.choice(np.flatnonzero(q == q.max())))


class DynaQ:
    """Tabular Dyna-Q.  plus=True gives Dyna-Q+ (exploration bonus kappa*sqrt(tau) in planning)."""

    def __init__(self, maze, rng, n_planning=5, alpha=0.1, gamma=0.95, eps=0.1,
                 plus=False, kappa=1e-3):
        self.m, self.rng = maze, rng
        self.n, self.alpha, self.gamma, self.eps = n_planning, alpha, gamma, eps
        self.plus, self.kappa = plus, kappa
        self.Q = defaultdict(lambda: np.zeros(4))
        self.model = {}                                  # (s, a) -> (r, s')
        self.keys = []                                   # list of model keys, for sampling
        self.last_tried = {}                             # (s, a) -> time step (Dyna-Q+)
        self.t = 0
        self.backups = 0

    def act(self, s):
        if self.rng.random() < self.eps:
            return int(self.rng.integers(4))
        return argmax_tie(self.Q[s], self.rng)

    def q_update(self, s, a, r, s2):
        target = r + self.gamma * self.Q[s2].max() * (s2 != self.m.goal)
        self.Q[s][a] += self.alpha * (target - self.Q[s][a])
        self.backups += 1

    def learn(self, s, a, r, s2):
        self.t += 1
        self.q_update(s, a, r, s2)                       # (d) direct RL
        if (s, a) not in self.model:                     # (e) model learning
            self.keys.append((s, a))
            if self.plus:                                # Dyna-Q+: untried actions from a
                for b in range(4):                       # newly seen state are "known" to
                    if b != a and (s, b) not in self.model:   # lead back to s with r = 0
                        self.model[(s, b)] = (0.0, s)
                        self.keys.append((s, b))
                        self.last_tried[(s, b)] = 0
        self.model[(s, a)] = (r, s2)
        self.last_tried[(s, a)] = self.t
        for _ in range(self.n):                          # (f) planning
            ps, pa = self.keys[self.rng.integers(len(self.keys))]
            pr, ps2 = self.model[(ps, pa)]
            if self.plus:
                pr += self.kappa * np.sqrt(self.t - self.last_tried[(ps, pa)])
            self.q_update(ps, pa, pr, ps2)


def run_episodes(agent, maze, n_episodes):
    lengths = []
    for _ in range(n_episodes):
        s, done, steps = maze.start, False, 0
        while not done:
            a = agent.act(s)
            s2, r, done = maze.step(s, a)
            agent.learn(s, a, r, s2)
            s, steps = s2, steps + 1
        lengths.append(steps)
    return np.array(lengths)


def greedy_length(Q, maze, cap=200):
    s, steps = maze.start, 0
    while s != maze.goal and steps < cap:
        s, _, _ = maze.step(s, int(np.argmax(Q[s])))
        steps += 1
    return steps


# ----------------------------------------------------------------------------
# Changing environment: the shortcut maze
# ----------------------------------------------------------------------------
def shortcut_experiment(plus, rng, total_steps=6000, change_at=3000, n=20):
    early = Maze(6, 9, start=(5, 3), goal=(0, 8), walls=[(3, c) for c in range(1, 9)])
    late = Maze(6, 9, start=(5, 3), goal=(0, 8), walls=[(3, c) for c in range(1, 8)])
    agent = DynaQ(early, rng, n_planning=n, alpha=1.0, eps=0.1, plus=plus, kappa=1e-3)
    cum, total, s = [], 0.0, early.start
    for t in range(total_steps):
        maze = early if t < change_at else late
        agent.m = maze
        a = agent.act(s)
        s2, r, done = maze.step(s, a)
        agent.learn(s, a, r, s2)
        total += r
        cum.append(total)
        s = maze.start if done else s2
    return np.array(cum)


# ----------------------------------------------------------------------------
# Prioritized sweeping (deterministic environment version, S&B Sec. 8.4)
# ----------------------------------------------------------------------------
class PrioritizedSweeping(DynaQ):
    def __init__(self, maze, rng, n_planning=5, theta=1e-4, **kw):
        super().__init__(maze, rng, n_planning, **kw)
        self.theta = theta
        self.pq = []                                     # max-heap via negative priority
        self.predecessors = defaultdict(set)             # s' -> {(s, a) that lead to s'}

    def priority(self, s, a, r, s2):
        target = r + self.gamma * self.Q[s2].max() * (s2 != self.m.goal)
        return abs(target - self.Q[s][a])

    def learn(self, s, a, r, s2):
        self.model[(s, a)] = (r, s2)
        self.predecessors[s2].add((s, a))
        p = self.priority(s, a, r, s2)
        if p > self.theta:
            heapq.heappush(self.pq, (-p, s, a))
        for _ in range(self.n):
            if not self.pq:
                break
            _, ps, pa = heapq.heappop(self.pq)           # most urgent update first
            pr, ps2 = self.model[(ps, pa)]
            self.q_update(ps, pa, pr, ps2)
            for (bs, ba) in self.predecessors[ps]:       # its value changed -> recheck
                br, _ = self.model[(bs, ba)]             # every pair that leads into it
                bp = self.priority(bs, ba, br, ps)
                if bp > self.theta:
                    heapq.heappush(self.pq, (-bp, bs, ba))


def backups_until_near_optimal(make_agent, maze, tol=1.2, max_episodes=200):
    """Count value backups (real + planned) until the greedy path is within `tol` x the
    shortest path.  (Exact optimality can take very long: actions never tried keep
    Q = 0, which is pessimistic here because all rewards are >= 0.)"""
    agent = make_agent()
    target = int(tol * maze.shortest_path())
    for ep in range(max_episodes):
        run_episodes(agent, maze, 1)
        if greedy_length(agent.Q, maze) <= target:
            return agent.backups, ep + 1
    return agent.backups, max_episodes


if __name__ == "__main__":
    maze = dyna_maze()
    print(f"1) Dyna maze (6x9), shortest path = {maze.shortest_path()} steps; "
          "steps per episode, mean of 10 runs")
    print(f"   {'n':>4s} {'ep 1':>7s} {'ep 2':>7s} {'ep 5':>7s} {'ep 10':>7s} {'ep 50':>7s}")
    for n in (0, 5, 50):
        L = np.mean([run_episodes(DynaQ(maze, np.random.default_rng(i), n_planning=n), maze, 50)
                     for i in range(10)], axis=0)
        print(f"   {n:4d} {L[0]:7.0f} {L[1]:7.0f} {L[4]:7.0f} {L[9]:7.0f} {L[49]:7.0f}")

    print("\n2) Shortcut maze: cumulative reward (5 runs, n=20, shortcut opens at step 3000)")
    for plus in (False, True):
        C = np.mean([shortcut_experiment(plus, np.random.default_rng(i)) for i in range(5)], 0)
        print(f"   {'Dyna-Q+' if plus else 'Dyna-Q ':8s}: at 3000 = {C[2999]:5.1f}, at 6000 = {C[-1]:5.1f}, "
              f"reward in steps 3001-6000 = {C[-1] - C[2999]:5.1f}")

    print("\n3) Backups until greedy path <= 1.2 x shortest (Dyna maze, n=5, alpha=0.5, 10 runs)")
    for name, make in [("Dyna-Q", lambda r: DynaQ(maze, r, n_planning=5, alpha=0.5)),
                       ("Prioritized sweeping",
                        lambda r: PrioritizedSweeping(maze, r, n_planning=5, alpha=0.5))]:
        res = np.array([backups_until_near_optimal(lambda: make(np.random.default_rng(i)), maze)
                        for i in range(10)])
        print(f"   {name:22s}: backups = {res[:, 0].mean():8.0f}, episodes = {res[:, 1].mean():5.1f}")
