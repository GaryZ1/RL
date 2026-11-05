"""
Temporal-difference prediction (Sutton & Barto, 2nd ed., Ch. 6, 7, 12).

  * TD(0) vs constant-alpha MC on the 5-state random walk (Example 6.2)
  * n-step TD on the 19-state random walk (Sec. 7.1)
  * TD(lambda), backward view with eligibility traces (Sec. 12.2)
  * check: offline TD(lambda) == offline lambda-return (forward view), exactly

Pure NumPy.   Run:  python3 code/td-learning.py
"""
import numpy as np


class RandomWalk:
    """n non-terminal states 1..n, terminals 0 and n+1, start in the middle.
    Each step goes left/right with prob 1/2.  Reward on the right exit is +1;
    on the left exit it is `left_reward` (0 for the 5-state task, -1 for 19 states)."""

    def __init__(self, n=5, left_reward=0.0):
        self.n, self.left_reward = n, left_reward
        self.start = (n + 1) // 2
        # true values: linear interpolation between the two exit rewards (gamma = 1)
        idx = np.arange(1, n + 1)
        self.true_v = left_reward + (1 - left_reward) * idx / (n + 1)

    def episode(self, rng):
        """Return visited states S_0..S_T and rewards R_1..R_T."""
        s, states, rewards = self.start, [self.start], []
        while 0 < s <= self.n:
            s += 1 if rng.random() < 0.5 else -1
            rewards.append(1.0 if s == self.n + 1 else (self.left_reward if s == 0 else 0.0))
            states.append(s)
        return states, rewards


def rms(V, env):
    return np.sqrt(np.mean((V[1:env.n + 1] - env.true_v) ** 2))


def td0(env, n_episodes, alpha, rng, gamma=1.0, v0=0.5):
    V = np.full(env.n + 2, v0)
    V[0] = V[-1] = 0.0                                  # terminal values are 0
    errors = []
    for _ in range(n_episodes):
        S, R = env.episode(rng)
        for t in range(len(R)):
            td_error = R[t] + gamma * V[S[t + 1]] - V[S[t]]   # delta_t
            V[S[t]] += alpha * td_error
        errors.append(rms(V, env))
    return V, np.array(errors)


def constant_alpha_mc(env, n_episodes, alpha, rng, gamma=1.0, v0=0.5):
    V = np.full(env.n + 2, v0)
    V[0] = V[-1] = 0.0
    errors = []
    for _ in range(n_episodes):
        S, R = env.episode(rng)
        G = 0.0
        for t in reversed(range(len(R))):               # every-visit, wait until the end
            G = gamma * G + R[t]
            V[S[t]] += alpha * (G - V[S[t]])
        errors.append(rms(V, env))
    return V, np.array(errors)


def n_step_td(env, n_episodes, alpha, n, rng, gamma=1.0):
    """Online n-step TD (S&B Sec. 7.1 pseudocode).  n=1 is TD(0); n >= T is MC."""
    V = np.zeros(env.n + 2)
    errors = []
    for _ in range(n_episodes):
        S, R = env.episode(rng)                         # pre-sampled: policy has no choice
        T = len(R)
        for tau in range(T):                            # tau = time whose estimate is updated
            end = min(tau + n, T)
            G = sum(gamma ** (i - tau) * R[i] for i in range(tau, end))
            if tau + n < T:
                G += gamma ** n * V[S[tau + n]]         # bootstrap from n steps ahead
            V[S[tau]] += alpha * (G - V[S[tau]])
        errors.append(rms(V, env))
    return V, np.array(errors)


def td_lambda(env, n_episodes, alpha, lam, rng, gamma=1.0, trace="accumulating"):
    """Online TD(lambda), backward view: every state keeps an eligibility trace z(s)."""
    V = np.zeros(env.n + 2)
    errors = []
    for _ in range(n_episodes):
        S, R = env.episode(rng)
        z = np.zeros_like(V)                            # eligibility traces, reset per episode
        for t in range(len(R)):
            z *= gamma * lam                            # decay all traces
            if trace == "accumulating":
                z[S[t]] += 1.0
            else:                                       # replacing traces
                z[S[t]] = 1.0
            delta = R[t] + gamma * V[S[t + 1]] - V[S[t]]
            V += alpha * delta * z                      # credit all recently visited states
            V[0] = V[-1] = 0.0
        errors.append(rms(V, env))
    return V, np.array(errors)


def lambda_return(S, R, V, t, lam, gamma=1.0):
    """Forward view: G_t^lambda = (1-lam) sum_n lam^(n-1) G_{t:t+n}  +  lam^(T-t-1) G_t."""
    T = len(R)
    G_lam, G_n = 0.0, 0.0
    for n in range(1, T - t + 1):
        G_n += gamma ** (n - 1) * R[t + n - 1]          # running sum of n-step rewards
        if t + n < T:
            G_lam += (1 - lam) * lam ** (n - 1) * (G_n + gamma ** n * V[S[t + n]])
        else:                                           # n-step return reached the end: MC return
            G_lam += lam ** (n - 1) * G_n
    return G_lam


def offline_equivalence_check(env, lam, rng, gamma=1.0):
    """With V held fixed during an episode, the summed TD(lambda) updates equal the
    summed lambda-return updates (S&B, 1st ed. Sec. 7.4; 2nd ed. Sec. 12.1-12.2)."""
    V = rng.normal(0, 0.3, env.n + 2)
    V[0] = V[-1] = 0.0
    S, R = env.episode(rng)
    forward = np.zeros_like(V)
    backward = np.zeros_like(V)
    z = np.zeros_like(V)
    for t in range(len(R)):
        forward[S[t]] += lambda_return(S, R, V, t, lam, gamma) - V[S[t]]
        z *= gamma * lam
        z[S[t]] += 1.0
        backward += (R[t] + gamma * V[S[t + 1]] - V[S[t]]) * z
    return np.max(np.abs(forward - backward))


if __name__ == "__main__":
    # --- 1. TD(0) vs MC on the 5-state random walk ---------------------------
    env5 = RandomWalk(5)
    runs, eps = 100, 100
    td_err = np.mean([td0(env5, eps, 0.1, np.random.default_rng(i))[1] for i in range(runs)], 0)
    mc_err = np.mean([constant_alpha_mc(env5, eps, 0.02, np.random.default_rng(i))[1]
                      for i in range(runs)], 0)
    V, _ = td0(env5, 3000, 0.01, np.random.default_rng(0))
    print("5-state random walk (100 runs x 100 episodes), RMS error vs episodes")
    print(f"   {'episode':>8s} {'TD(0) a=0.1':>12s} {'MC a=0.02':>10s}")
    for e in (1, 10, 50, 100):
        print(f"   {e:8d} {td_err[e - 1]:12.3f} {mc_err[e - 1]:10.3f}")
    print(f"   TD(0) a=0.01 V after 3000 episodes: {np.round(V[1:6], 3)}  (true {np.round(env5.true_v, 3)})")

    # --- 2. n-step TD on the 19-state random walk -----------------------------
    env19 = RandomWalk(19, left_reward=-1.0)
    alphas = [0.05, 0.1, 0.2, 0.4, 0.6]
    runs = 30
    print("\n19-state random walk: mean RMS over first 10 episodes (30 runs), best alpha")
    for n in (1, 2, 4, 8, 16, 64):
        res = [np.mean([n_step_td(env19, 10, a, n, np.random.default_rng(i))[1].mean()
                        for i in range(runs)]) for a in alphas]
        b = int(np.argmin(res))
        print(f"   n-step TD  n={n:<3d} best alpha={alphas[b]:.2f}  RMS={res[b]:.3f}")

    # --- 3. TD(lambda), backward view ----------------------------------------
    # Accumulating traces with lambda ~ 1 and a large alpha can blow up (traces > 1
    # multiply the step size); we count such alphas as "diverged" instead of crashing.
    for trace in ("accumulating", "replacing"):
        for lam in (0.0, 0.4, 0.8, 0.9, 1.0):
            res = []
            for a in alphas:
                with np.errstate(all="ignore"):
                    r = np.mean([td_lambda(env19, 10, a, lam, np.random.default_rng(i),
                                           trace=trace)[1].mean() for i in range(runs)])
                res.append(r if np.isfinite(r) and r < 10 else np.inf)
            b = int(np.argmin(res))
            n_div = int(np.sum(np.isinf(res)))
            print(f"   TD(lambda) {trace:12s} lam={lam:<4.1f} best alpha={alphas[b]:.2f}  "
                  f"RMS={res[b]:.3f}" + (f"  ({n_div} alpha(s) diverged)" if n_div else ""))

    # --- 4. forward view == backward view (offline) ---------------------------
    rng = np.random.default_rng(0)
    gap = max(offline_equivalence_check(env19, lam, rng) for lam in (0.3, 0.7, 0.95))
    print(f"\nOffline lambda-return vs offline TD(lambda): max |difference| = {gap:.2e}")
