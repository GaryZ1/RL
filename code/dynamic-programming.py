"""
Dynamic programming for finite MDPs (Sutton & Barto, 2nd ed., Ch. 3-4).

  * iterative policy evaluation   (Bellman expectation backup)
  * exact policy evaluation       (linear solve, for checking)
  * policy iteration              (evaluate -> greedy improve -> repeat)
  * value iteration               (Bellman optimality backup)

The MDP is given as dense arrays P[s, a, s'] and R[s, a] (expected reward),
built from a tiny GridWorld.  Pure NumPy.   Run:  python3 code/dynamic-programming.py
"""
import numpy as np

ACTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]   # up, right, down, left
ARROWS = "^>v<"


class GridWorld:
    """rows x cols grid; -1 reward per step; terminal cells are absorbing (reward 0).

    slip: probability that the move goes to one of the two perpendicular directions
    instead (split equally).  slip=0 gives Example 4.1 of Sutton & Barto.
    """

    def __init__(self, rows=4, cols=4, terminals=(0, 15), step_reward=-1.0, slip=0.0):
        self.rows, self.cols = rows, cols
        self.nS, self.nA = rows * cols, 4
        self.terminals = set(terminals)
        self.P = np.zeros((self.nS, self.nA, self.nS))
        self.R = np.zeros((self.nS, self.nA))
        for s in range(self.nS):
            for a in range(self.nA):
                if s in self.terminals:                 # absorbing, no more reward
                    self.P[s, a, s] = 1.0
                    continue
                perp = [(a + 1) % 4, (a + 3) % 4]
                for move, p in [(a, 1 - slip), (perp[0], slip / 2), (perp[1], slip / 2)]:
                    if p > 0:
                        self.P[s, a, self._next(s, move)] += p
                self.R[s, a] = step_reward

    def _next(self, s, a):
        r, c = divmod(s, self.cols)
        dr, dc = ACTIONS[a]
        nr, nc = r + dr, c + dc
        if 0 <= nr < self.rows and 0 <= nc < self.cols:
            return nr * self.cols + nc
        return s                                         # bumping a wall: stay put


def q_from_v(P, R, V, gamma):
    """One-step lookahead: Q(s,a) = R(s,a) + gamma * sum_s' P(s'|s,a) V(s')."""
    return R + gamma * P @ V                             # (S,A,S') @ (S',) -> (S,A)


def policy_evaluation(P, R, pi, gamma=1.0, theta=1e-8, in_place=True):
    """Iterative policy evaluation.  pi[s, a] = probability of action a in state s.

    in_place=True updates V(s) immediately (Gauss-Seidel), which usually converges
    in fewer sweeps than the two-array (Jacobi) version.
    """
    nS = P.shape[0]
    V = np.zeros(nS)
    sweeps = 0
    while True:
        sweeps += 1
        if in_place:
            delta = 0.0
            for s in range(nS):
                v_old = V[s]
                V[s] = pi[s] @ (R[s] + gamma * P[s] @ V)
                delta = max(delta, abs(v_old - V[s]))
        else:
            V_new = np.sum(pi * q_from_v(P, R, V, gamma), axis=1)
            delta = np.max(np.abs(V_new - V))
            V = V_new
        if delta < theta:
            return V, sweeps


def exact_policy_value(P, R, pi, gamma, nonterminal):
    """Solve the Bellman expectation equation V = R_pi + gamma P_pi V on non-terminal states."""
    P_pi = np.einsum("sa,sat->st", pi, P)
    R_pi = np.sum(pi * R, axis=1)
    idx = np.array(sorted(nonterminal))
    V = np.zeros(P.shape[0])
    A = np.eye(len(idx)) - gamma * P_pi[np.ix_(idx, idx)]
    V[idx] = np.linalg.solve(A, R_pi[idx])
    return V


def greedy(P, R, V, gamma, old_policy=None, tol=1e-9):
    """Greedy policy w.r.t. V.  Keep the old action on (near-)ties to avoid flip-flopping."""
    Q = q_from_v(P, R, V, gamma)
    new = Q.argmax(axis=1)
    if old_policy is not None:
        keep = Q[np.arange(len(new)), old_policy] >= Q.max(axis=1) - tol
        new = np.where(keep, old_policy, new)
    return new


def policy_iteration(P, R, gamma=1.0, theta=1e-8):
    """Policy iteration.  We start from the equiprobable random policy (rather than an
    arbitrary deterministic one) because with gamma=1 a deterministic policy that never
    reaches a terminal state has value -infinity and its evaluation would never converge.
    Every greedy policy after that is "proper" (it improves on a proper policy)."""
    nS, nA = R.shape
    pi = np.full((nS, nA), 1.0 / nA)                     # pi[s, a], stochastic start
    policy = None
    for it in range(1, 1000):
        V, _ = policy_evaluation(P, R, pi, gamma, theta)        # 1. evaluation
        new_policy = greedy(P, R, V, gamma, old_policy=policy)  # 2. improvement
        if policy is not None and np.array_equal(new_policy, policy):
            return policy, V, it                         # policy stable -> optimal
        policy = new_policy
        pi = np.eye(nA)[policy]                          # one-hot deterministic policy
    raise RuntimeError("policy iteration did not converge")


def value_iteration(P, R, gamma=1.0, theta=1e-8):
    V = np.zeros(P.shape[0])
    for sweep in range(1, 100000):
        V_new = q_from_v(P, R, V, gamma).max(axis=1)     # Bellman optimality backup
        delta = np.max(np.abs(V_new - V))
        V = V_new
        if delta < theta:
            return greedy(P, R, V, gamma), V, sweep
    raise RuntimeError("value iteration did not converge")


def show(env, V=None, policy=None):
    for r in range(env.rows):
        cells = []
        for c in range(env.cols):
            s = r * env.cols + c
            if policy is not None:
                cells.append(" T" if s in env.terminals else " " + ARROWS[policy[s]])
            else:
                cells.append(f"{V[s]:7.2f}")
        print("   " + " ".join(cells))


if __name__ == "__main__":
    # --- Example 4.1: equiprobable random policy on the 4x4 grid, gamma = 1 -----
    env = GridWorld()
    nonterm = set(range(env.nS)) - env.terminals
    pi_random = np.full((env.nS, env.nA), 0.25)
    V_inplace, sw_in = policy_evaluation(env.P, env.R, pi_random, gamma=1.0, in_place=True)
    V_twoarr, sw_two = policy_evaluation(env.P, env.R, pi_random, gamma=1.0, in_place=False)
    V_exact = exact_policy_value(env.P, env.R, pi_random, 1.0, nonterm)
    print("Example 4.1 - v_pi for the random policy (expect 0 -14 -20 -22 / ...):")
    show(env, V_inplace)
    print(f"   sweeps: in-place={sw_in}, two-array={sw_two};  "
          f"max |iterative - exact| = {np.abs(V_inplace - V_exact).max():.2e}")

    # --- Policy iteration and value iteration on the same grid ----------------
    pol_pi, V_pi, n_pi = policy_iteration(env.P, env.R, gamma=1.0)
    pol_vi, V_vi, n_vi = value_iteration(env.P, env.R, gamma=1.0)
    print(f"\nDeterministic grid: policy iteration {n_pi} iterations, "
          f"value iteration {n_vi} sweeps; same V*? {np.allclose(V_pi, V_vi)}")
    show(env, V_vi)
    show(env, policy=pol_vi)

    # --- Slippery grid (20% slip), gamma = 0.95 -------------------------------
    slip = GridWorld(slip=0.2)
    pol_pi, V_pi, n_pi = policy_iteration(slip.P, slip.R, gamma=0.95)
    pol_vi, V_vi, n_vi = value_iteration(slip.P, slip.R, gamma=0.95)
    print(f"\nSlippery grid (slip=0.2, gamma=0.95): PI {n_pi} iterations, VI {n_vi} sweeps, "
          f"max |V_PI - V_VI| = {np.abs(V_pi - V_vi).max():.1e}")
    show(slip, V_vi)
    show(slip, policy=pol_pi)
    # Bellman optimality residual of the result (should be ~0)
    resid = np.abs(q_from_v(slip.P, slip.R, V_vi, 0.95).max(1) - V_vi).max()
    print(f"   Bellman optimality residual: {resid:.1e}")
