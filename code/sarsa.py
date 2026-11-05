"""
SARSA-family control on CliffWalking (Sutton & Barto, 2nd ed., Example 6.6; Ch. 7, 12).

  * SARSA (one-step, on-policy TD control)
  * Expected SARSA
  * n-step SARSA
  * SARSA(lambda) with replacing eligibility traces
  * Q-learning, included only as the classic comparison

Pure NumPy.   Run:  python3 code/sarsa.py
"""
import numpy as np


class CliffWalking:
    """4 x 12 grid.  Start bottom-left, goal bottom-right, the cells between are a cliff.
    Every step costs -1; stepping into the cliff costs -100 and teleports you to start."""
    rows, cols = 4, 12
    n_states, n_actions = rows * cols, 4
    start, goal = 3 * 12 + 0, 3 * 12 + 11
    moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]          # up, right, down, left

    def step(self, s, a):
        r, c = divmod(s, self.cols)
        r = min(max(r + self.moves[a][0], 0), self.rows - 1)
        c = min(max(c + self.moves[a][1], 0), self.cols - 1)
        s2 = r * self.cols + c
        if r == 3 and 1 <= c <= 10:                     # fell off the cliff
            return self.start, -100.0, False
        return s2, -1.0, s2 == self.goal


def eps_greedy(Q, s, eps, rng):
    if rng.random() < eps:
        return int(rng.integers(Q.shape[1]))
    q = Q[s]
    return int(rng.choice(np.flatnonzero(q == q.max())))   # random tie-breaking


def eps_greedy_probs(Q, s, eps):
    """pi(.|s) for an eps-greedy policy (needed by Expected SARSA)."""
    nA = Q.shape[1]
    greedy = np.flatnonzero(Q[s] == Q[s].max())
    p = np.full(nA, eps / nA)
    p[greedy] += (1 - eps) / len(greedy)
    return p


def sarsa(env, episodes, rng, alpha=0.5, eps=0.1, gamma=1.0, expected=False):
    """expected=False: target R + g Q(S',A').   expected=True: R + g sum_a pi(a|S') Q(S',a)."""
    Q = np.zeros((env.n_states, env.n_actions))
    returns = []
    for _ in range(episodes):
        s, G, done = env.start, 0.0, False
        a = eps_greedy(Q, s, eps, rng)
        while not done:
            s2, r, done = env.step(s, a)
            G += r
            a2 = eps_greedy(Q, s2, eps, rng)
            if done:
                target = r                              # Q(terminal, .) = 0
            elif expected:
                target = r + gamma * eps_greedy_probs(Q, s2, eps) @ Q[s2]
            else:
                target = r + gamma * Q[s2, a2]
            Q[s, a] += alpha * (target - Q[s, a])
            s, a = s2, a2
        returns.append(G)
    return Q, np.array(returns)


def q_learning(env, episodes, rng, alpha=0.5, eps=0.1, gamma=1.0):
    Q = np.zeros((env.n_states, env.n_actions))
    returns = []
    for _ in range(episodes):
        s, G, done = env.start, 0.0, False
        while not done:
            a = eps_greedy(Q, s, eps, rng)
            s2, r, done = env.step(s, a)
            G += r
            target = r if done else r + gamma * Q[s2].max()   # off-policy: greedy target
            Q[s, a] += alpha * (target - Q[s, a])
            s = s2
        returns.append(G)
    return Q, np.array(returns)


def n_step_sarsa(env, episodes, rng, n=4, alpha=0.2, eps=0.1, gamma=1.0):
    """S&B Sec. 7.2.  Stores the last n+1 states/actions/rewards of the episode."""
    Q = np.zeros((env.n_states, env.n_actions))
    returns = []
    for _ in range(episodes):
        S, A, R = [env.start], [eps_greedy(Q, env.start, eps, rng)], [0.0]  # R[0] unused
        T, t = np.inf, 0
        while True:
            if t < T:
                s2, r, done = env.step(S[t], A[t])
                S.append(s2)
                R.append(r)
                if done:
                    T = t + 1
                else:
                    A.append(eps_greedy(Q, s2, eps, rng))
            tau = t - n + 1                             # time whose estimate is updated
            if tau >= 0:
                G = sum(gamma ** (i - tau - 1) * R[i] for i in range(tau + 1, int(min(tau + n, T)) + 1))
                if tau + n < T:
                    G += gamma ** n * Q[S[tau + n], A[tau + n]]
                Q[S[tau], A[tau]] += alpha * (G - Q[S[tau], A[tau]])
            if tau == T - 1:
                break
            t += 1
        returns.append(sum(R))
    return Q, np.array(returns)


def sarsa_lambda(env, episodes, rng, lam=0.5, alpha=0.2, eps=0.1, gamma=1.0):
    """Backward-view SARSA(lambda) with replacing traces (S&B Sec. 12.7)."""
    Q = np.zeros((env.n_states, env.n_actions))
    returns = []
    for _ in range(episodes):
        z = np.zeros_like(Q)                            # one trace per (s, a)
        s, G, done = env.start, 0.0, False
        a = eps_greedy(Q, s, eps, rng)
        while not done:
            s2, r, done = env.step(s, a)
            G += r
            a2 = eps_greedy(Q, s2, eps, rng)
            delta = r - Q[s, a] + (0.0 if done else gamma * Q[s2, a2])
            z[s, a] = 1.0                               # replacing trace
            Q += alpha * delta * z                      # update every eligible pair
            z *= gamma * lam
            s, a = s2, a2
        returns.append(G)
    return Q, np.array(returns)


def greedy_path(env, Q, max_steps=100):
    """Render the greedy route from start ('o' = path, 'C' = cliff)."""
    grid = [["." for _ in range(env.cols)] for _ in range(env.rows)]
    for c in range(1, 11):
        grid[3][c] = "C"
    s, steps = env.start, 0
    while s != env.goal and steps < max_steps:
        r, c = divmod(s, env.cols)
        grid[r][c] = "o"
        s, _, _ = env.step(s, int(np.argmax(Q[s])))
        steps += 1
    grid[3][11] = "G"
    return steps, "\n".join("      " + "".join(row) for row in grid)


if __name__ == "__main__":
    env = CliffWalking()
    runs, episodes = 20, 500
    algos = {
        "SARSA            a=0.5": lambda rng: sarsa(env, episodes, rng, alpha=0.5),
        "Expected SARSA   a=0.5": lambda rng: sarsa(env, episodes, rng, alpha=0.5, expected=True),
        "Expected SARSA   a=1.0": lambda rng: sarsa(env, episodes, rng, alpha=1.0, expected=True),
        "4-step SARSA     a=0.2": lambda rng: n_step_sarsa(env, episodes, rng, n=4, alpha=0.2),
        "SARSA(0.5)       a=0.2": lambda rng: sarsa_lambda(env, episodes, rng, lam=0.5, alpha=0.2),
        "Q-learning       a=0.5": lambda rng: q_learning(env, episodes, rng, alpha=0.5),
    }
    print(f"CliffWalking, eps=0.1, {runs} runs x {episodes} episodes")
    print(f"   {'algorithm':24s} {'online return (eps 1-100)':>26s} {'(eps 101-500)':>14s} {'greedy steps':>13s}")
    final_Q = {}
    for name, fn in algos.items():
        results = [fn(np.random.default_rng(seed)) for seed in range(runs)]
        R = np.mean([r for _, r in results], axis=0)
        steps, _ = greedy_path(env, results[0][0])
        final_Q[name] = results[0][0]
        print(f"   {name:24s} {R[:100].mean():26.1f} {R[100:].mean():14.1f} {steps:13d}")

    for name in ("SARSA            a=0.5", "Q-learning       a=0.5"):
        steps, pic = greedy_path(env, final_Q[name])
        print(f"\n   Greedy route learned by {name.split()[0]} ({steps} steps):\n{pic}")
