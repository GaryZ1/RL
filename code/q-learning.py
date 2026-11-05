"""
Q-learning and Double Q-learning (Watkins 1989; Watkins & Dayan 1992; van Hasselt 2010;
Sutton & Barto, 2nd ed., Sec. 6.5 & 6.7).

  1. Convergence demo: Q-learning with a uniformly random behaviour policy and
     Robbins-Monro step sizes recovers Q* (computed by value iteration) on a slippery grid.
  2. Maximization bias (S&B Example 6.7): Q-learning vs Double Q-learning.
  3. Greedy route Q-learning finds on CliffWalking.

Pure NumPy.   Run:  python3 code/q-learning.py
"""
import numpy as np

MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]               # up, right, down, left


class SlipperyGrid:
    """4x4 grid, goal at bottom-right (terminal), -1 per step.  With probability `slip`
    the agent moves in a random perpendicular direction instead of the intended one."""

    def __init__(self, n=4, slip=0.2):
        self.n, self.slip = n, slip
        self.n_states, self.n_actions, self.goal = n * n, 4, n * n - 1

    def _move(self, s, a):
        r, c = divmod(s, self.n)
        nr, nc = r + MOVES[a][0], c + MOVES[a][1]
        return nr * self.n + nc if (0 <= nr < self.n and 0 <= nc < self.n) else s

    def step(self, s, a, rng):
        u = rng.random()
        if u < self.slip / 2:
            a = (a + 1) % 4
        elif u < self.slip:
            a = (a + 3) % 4
        s2 = self._move(s, a)
        return s2, -1.0, s2 == self.goal

    def q_star(self, gamma, iters=2000):
        """Model-based Q* by value iteration (used only to check Q-learning)."""
        Q = np.zeros((self.n_states, 4))
        for _ in range(iters):
            V = Q.max(1)
            V[self.goal] = 0.0
            for s in range(self.n_states):
                for a in range(4):
                    nxt = [(1 - self.slip, a), (self.slip / 2, (a + 1) % 4), (self.slip / 2, (a + 3) % 4)]
                    Q[s, a] = sum(p * (-1.0 + gamma * V[self._move(s, b)]) for p, b in nxt)
        Q[self.goal] = 0.0
        return Q


def q_learning_random_behaviour(env, steps, rng, gamma=0.9, omega=0.7):
    """Off-policy: behaviour is uniform random, target is greedy.
    Step size alpha = 1 / N(s,a)^omega, with 0.5 < omega <= 1 (Robbins-Monro)."""
    Q = np.zeros((env.n_states, env.n_actions))
    N = np.zeros_like(Q)
    s = int(rng.integers(env.n_states - 1))
    for _ in range(steps):
        a = int(rng.integers(env.n_actions))            # explore everything forever
        s2, r, done = env.step(s, a, rng)
        N[s, a] += 1
        alpha = 1.0 / N[s, a] ** omega
        target = r + (0.0 if done else gamma * Q[s2].max())
        Q[s, a] += alpha * (target - Q[s, a])
        s = int(rng.integers(env.n_states - 1)) if done else s2   # random restart
    return Q


# ----------------------------------------------------------------------------
# Maximization bias MDP (S&B Example 6.7)
# ----------------------------------------------------------------------------
A_STATE, B_STATE, TERMINAL = 0, 1, 2
LEFT, RIGHT = 0, 1
N_B_ACTIONS = 10


def mb_step(s, a, rng):
    if s == A_STATE:
        return (B_STATE, 0.0, False) if a == LEFT else (TERMINAL, 0.0, True)
    return TERMINAL, rng.normal(-0.1, 1.0), True        # every action in B: N(-0.1, 1)


def n_actions(s):
    return 2 if s == A_STATE else N_B_ACTIONS


def eps_greedy(q, eps, rng):
    if rng.random() < eps:
        return int(rng.integers(len(q)))
    return int(rng.choice(np.flatnonzero(q == q.max())))


def run_max_bias(double, episodes, rng, alpha=0.1, eps=0.1, gamma=1.0):
    """Returns a 0/1 array: did the agent go LEFT from A in each episode?"""
    Q1 = [np.zeros(2), np.zeros(N_B_ACTIONS)]
    Q2 = [np.zeros(2), np.zeros(N_B_ACTIONS)]
    went_left = np.zeros(episodes)
    for ep in range(episodes):
        s, done = A_STATE, False
        while not done:
            q_behave = Q1[s] + Q2[s] if double else Q1[s]   # Double Q acts on Q1 + Q2
            a = eps_greedy(q_behave, eps, rng)
            if s == A_STATE:
                went_left[ep] = (a == LEFT)
            s2, r, done = mb_step(s, a, rng)
            if not double:
                target = r + (0.0 if done else gamma * Q1[s2].max())
                Q1[s][a] += alpha * (target - Q1[s][a])
            else:
                # flip a coin: update one table, using the OTHER to evaluate its argmax
                QA, QB = (Q1, Q2) if rng.random() < 0.5 else (Q2, Q1)
                if done:
                    target = r
                else:
                    best = eps_greedy(QA[s2], 0.0, rng)  # argmax by QA (random ties)
                    target = r + gamma * QB[s2][best]    # ... evaluated by QB
                QA[s][a] += alpha * (target - QA[s][a])
            s = s2
    return went_left


# ----------------------------------------------------------------------------
# CliffWalking (same layout as S&B Example 6.6)
# ----------------------------------------------------------------------------
def cliff_step(s, a):
    r, c = divmod(s, 12)
    r = min(max(r + MOVES[a][0], 0), 3)
    c = min(max(c + MOVES[a][1], 0), 11)
    if r == 3 and 1 <= c <= 10:
        return 36, -100.0, False                       # back to start
    s2 = r * 12 + c
    return s2, -1.0, s2 == 47


def q_learning_cliff(episodes, rng, alpha=0.5, eps=0.1, gamma=1.0):
    Q = np.zeros((48, 4))
    for _ in range(episodes):
        s, done = 36, False
        while not done:
            a = eps_greedy(Q[s], eps, rng)
            s2, r, done = cliff_step(s, a)
            Q[s, a] += alpha * (r + (0.0 if done else gamma * Q[s2].max()) - Q[s, a])
            s = s2
    return Q


if __name__ == "__main__":
    # 1. convergence to Q*
    env, gamma = SlipperyGrid(), 0.9
    Qs = env.q_star(gamma)
    print("1) Q-learning (random behaviour, alpha=1/N^0.7) vs value-iteration Q*, slippery 4x4 grid")
    for steps in (10_000, 100_000, 400_000):
        Q = q_learning_random_behaviour(env, steps, np.random.default_rng(0), gamma)
        err = np.abs(Q - Qs).max()
        # greedy action is optimal if it attains max_a Q*(s,a) (the grid has many ties)
        a_greedy = Q[:-1].argmax(1)
        ok = Qs[np.arange(env.n_states - 1), a_greedy] >= Qs[:-1].max(1) - 1e-6
        print(f"   {steps:7d} steps: max |Q - Q*| = {err:.3f}, greedy action optimal in "
              f"{100 * ok.mean():.0f}% of states")

    # 2. maximization bias
    runs, episodes = 1000, 300
    print(f"\n2) Maximization bias (Example 6.7): % of episodes going LEFT from A ({runs} runs);"
          " optimal under eps=0.1 is 5%")
    left_q = np.mean([run_max_bias(False, episodes, np.random.default_rng(i)) for i in range(runs)], 0)
    left_dq = np.mean([run_max_bias(True, episodes, np.random.default_rng(i)) for i in range(runs)], 0)
    print(f"   {'episodes':>12s} {'Q-learning':>11s} {'Double Q':>9s}")
    for lo, hi in ((0, 10), (10, 25), (25, 50), (100, 150), (250, 300)):
        print(f"   {f'{lo + 1}-{hi}':>12s} {100 * left_q[lo:hi].mean():10.1f}% {100 * left_dq[lo:hi].mean():8.1f}%")

    # 3. cliff
    Q = q_learning_cliff(500, np.random.default_rng(0))
    s, path = 36, [36]
    while s != 47 and len(path) < 100:
        s, _, _ = cliff_step(s, int(np.argmax(Q[s])))
        path.append(s)
    print(f"\n3) CliffWalking: greedy Q-learning route has {len(path) - 1} steps "
          f"(optimal 13, runs along the cliff edge); Q(start) = {Q[36].max():.1f}")
