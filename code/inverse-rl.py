"""
Maximum-Entropy Inverse RL (Ziebart, Maas, Bagnell & Dey, 2008) on a tiny gridworld,
in pure NumPy, plus a transfer test that shows WHY one would recover a reward instead
of just cloning actions.

Each cell has a terrain type: road, grass, lava, or goal. The (hidden) true reward is
linear in one-hot terrain features:  r(s) = w_true . phi(s).
The expert behaves (soft-)optimally for the true reward on MAP A; we observe only its
trajectories. MaxEnt IRL recovers w by matching expected feature counts.
Then we plan with the recovered reward on a NEW map B, which behavior cloning cannot
transfer to (the states are different), but a reward over features can.

Run:  python3 code/inverse-rl.py     (a few seconds)
"""
import numpy as np

rng = np.random.default_rng(0)
N, T = 8, 20                                   # grid size, horizon
MOVES = [(1, 0), (0, 1), (-1, 0), (0, -1), (0, 0)]  # R, U, L, D, stay
TYPES = ["road", "grass", "lava", "goal"]
W_TRUE = np.array([0.0, -0.3, -2.0, 1.0])      # hidden from the learner


def make_map(seed):
    r = np.random.default_rng(seed)
    terrain = r.choice(3, size=(N, N), p=[0.5, 0.35, 0.15])  # road / grass / lava
    terrain[N - 1, N - 1] = 3                                   # goal in the corner
    terrain[0, 0] = 0
    return terrain


def features(terrain):
    """phi: [S, F] one-hot terrain features, S = N*N."""
    return np.eye(len(TYPES))[terrain.reshape(-1)]


def transition_table():
    """Deterministic next-state index P[s, a]."""
    P = np.zeros((N * N, len(MOVES)), int)
    for x in range(N):
        for y in range(N):
            for a, (dx, dy) in enumerate(MOVES):
                nx, ny = min(max(x + dx, 0), N - 1), min(max(y + dy, 0), N - 1)
                P[x * N + y, a] = nx * N + ny
    return P


P = transition_table()


def soft_value_iteration(r, T=T):
    """Finite-horizon soft (MaxEnt) Bellman backup.
    Q_t(s,a) = r(s) + V_{t+1}(s'),   V_t(s) = log sum_a exp Q_t(s,a)
    Returns time-indexed stochastic policies pi_t(a|s) = exp(Q_t - V_t)."""
    V = np.zeros(N * N)
    policies = []
    for _ in range(T):
        Q = r[:, None] + V[P]                       # [S, A]
        V = np.logaddexp.reduce(Q, axis=1)
        policies.append(np.exp(Q - V[:, None]))
    return policies[::-1]                           # pi_0 ... pi_{T-1}


def expected_svf(policies, p0):
    """Forward pass: expected state-visitation counts D(s) = sum_t P(s_t = s)."""
    d, D = p0.copy(), np.zeros(N * N)
    for pi in policies:
        D += d
        d_next = np.zeros(N * N)
        for a in range(len(MOVES)):
            np.add.at(d_next, P[:, a], d * pi[:, a])
        d = d_next
    return D


def sample_demos(policies, starts, n):
    trajs = []
    for _ in range(n):
        s = starts[rng.integers(len(starts))]
        traj = [s]
        for pi in policies[:-1]:
            a = rng.choice(len(MOVES), p=pi[s])
            s = P[s, a]
            traj.append(s)
        trajs.append(traj)
    return trajs


def maxent_irl(phi, trajs, n_iters=300, lr=0.05):
    """Gradient ascent on the log-likelihood of demos under the MaxEnt model:
       grad L(w) = mu_expert - E_{pi_w}[ sum_t phi(s_t) ]."""
    mu_E = np.mean([phi[t].sum(0) for t in trajs], axis=0)
    p0 = np.bincount([t[0] for t in trajs], minlength=N * N) / len(trajs)
    w = np.zeros(phi.shape[1])
    for it in range(n_iters):
        policies = soft_value_iteration(phi @ w)
        mu_w = expected_svf(policies, p0) @ phi
        grad = mu_E - mu_w
        w += lr * grad
        if it % 100 == 0 or it == n_iters - 1:
            print(f"  iter {it:3d}  |feature-count gap| = {np.abs(grad).sum():.3f}")
    return w


def true_return(terrain, policies, starts):
    """Expected TRUE return of a (time-indexed) policy from uniform starts."""
    phi = features(terrain)
    p0 = np.zeros(N * N); p0[starts] = 1 / len(starts)
    return float(expected_svf(policies, p0) @ (phi @ W_TRUE))


def greedy(policies):
    """Turn soft policies into deterministic argmax policies (planning with the reward)."""
    return [np.eye(len(MOVES))[pi.argmax(1)] for pi in policies]


if __name__ == "__main__":
    A, B = make_map(1), make_map(7)
    starts = [x * N + y for x in range(3) for y in range(3)]  # bottom-left 3x3 block
    phiA = features(A)

    expert_pols = soft_value_iteration(phiA @ W_TRUE)
    demos = sample_demos(expert_pols, starts, n=300)
    print(f"Collected {len(demos)} expert demonstrations of length {T} on map A")

    print("\nRunning MaxEnt IRL ...")
    w = maxent_irl(phiA, demos)
    # Rewards are only identifiable up to shift (and scale, given temperature): compare shifted
    print("\n terrain   true w   learned w   (learned - learned[road])")
    for i, name in enumerate(TYPES):
        print(f" {name:<7}{W_TRUE[i]:>8.2f}{w[i]:>11.2f}{w[i] - w[0]:>12.2f}")

    print("\nExpected TRUE return from the same start states")
    print("  (soft = Boltzmann policy like the demonstrator; greedy = deterministic planner)")
    print(f"  {'map':<14}{'soft expert':>12}{'soft IRL':>10}{'greedy true-r':>15}{'greedy IRL-r':>14}{'random':>8}")
    for name, terrain in [("A (train)", A), ("B (new)", B)]:
        phi = features(terrain)
        row = [true_return(terrain, soft_value_iteration(phi @ W_TRUE), starts),
               true_return(terrain, soft_value_iteration(phi @ w), starts),
               true_return(terrain, greedy(soft_value_iteration(phi @ W_TRUE)), starts),
               true_return(terrain, greedy(soft_value_iteration(phi @ w)), starts),
               true_return(terrain, [np.full((N * N, 5), 0.2)] * T, starts)]
        print(f"  {name:<14}" + "".join(f"{v:>{wd}.2f}" for v, wd in zip(row, [12, 10, 15, 14, 8])))
    print("\nThe recovered reward transfers to a new layout because it is a function of terrain\n"
          "features, not of particular cells: that is the main selling point of IRL over BC.")
