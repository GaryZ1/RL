"""
Multi-armed bandits on the 10-armed testbed (Sutton & Barto, 2nd ed., Ch. 2),
plus disjoint LinUCB on a toy contextual bandit (Li et al., 2010).

Agents: epsilon-greedy (sample average), optimistic greedy, UCB1,
Gaussian Thompson sampling, gradient bandit (softmax preferences + baseline).

Pure NumPy, no other dependencies.   Run:  python3 code/multi-armed-bandits.py
"""
import numpy as np


# ----------------------------------------------------------------------------
# Environment: the k-armed Gaussian testbed
# ----------------------------------------------------------------------------
class Testbed:
    """True values q*(a) ~ N(mean, 1); each pull returns R ~ N(q*(a), 1)."""

    def __init__(self, k, rng, mean=0.0):
        self.rng = rng
        self.q_star = rng.normal(mean, 1.0, size=k)
        self.best = int(np.argmax(self.q_star))

    def pull(self, a):
        return self.rng.normal(self.q_star[a], 1.0)


def argmax_tie(x, rng):
    """argmax with random tie-breaking (important when all estimates start equal)."""
    return int(rng.choice(np.flatnonzero(x == x.max())))


# ----------------------------------------------------------------------------
# Agents: each has act(t) -> action and update(a, r)
# ----------------------------------------------------------------------------
class EpsilonGreedy:
    """Sample-average (alpha=None) or constant-step-size epsilon-greedy."""

    def __init__(self, k, rng, eps=0.1, q0=0.0, alpha=None):
        self.k, self.rng, self.eps, self.alpha = k, rng, eps, alpha
        self.Q = np.full(k, q0, dtype=float)   # action-value estimates
        self.N = np.zeros(k)                   # pull counts

    def act(self, t):
        if self.rng.random() < self.eps:
            return int(self.rng.integers(self.k))       # explore
        return argmax_tie(self.Q, self.rng)             # exploit

    def update(self, a, r):
        self.N[a] += 1
        step = self.alpha if self.alpha is not None else 1.0 / self.N[a]
        self.Q[a] += step * (r - self.Q[a])             # NewEst = Old + step*(Target-Old)


class UCB1:
    """Upper-confidence-bound action selection: Q(a) + c*sqrt(ln t / N(a))."""

    def __init__(self, k, rng, c=2.0):
        self.k, self.rng, self.c = k, rng, c
        self.Q = np.zeros(k)
        self.N = np.zeros(k)

    def act(self, t):
        untried = np.flatnonzero(self.N == 0)
        if untried.size:                                # pull every arm once first
            return int(untried[0])
        bonus = self.c * np.sqrt(np.log(t) / self.N)
        return argmax_tie(self.Q + bonus, self.rng)

    def update(self, a, r):
        self.N[a] += 1
        self.Q[a] += (r - self.Q[a]) / self.N[a]


class GaussianThompson:
    """Thompson sampling with a N(mu0, 1/tau0) prior and known noise variance 1.

    Posterior for each arm stays Gaussian: precision tau grows by 1 per pull,
    mean is the precision-weighted average of prior mean and observed rewards.
    """

    def __init__(self, k, rng, mu0=0.0, tau0=1.0):
        self.rng = rng
        self.mu = np.full(k, mu0, dtype=float)   # posterior means
        self.tau = np.full(k, tau0, dtype=float)  # posterior precisions

    def act(self, t):
        theta = self.rng.normal(self.mu, 1.0 / np.sqrt(self.tau))  # one sample per arm
        return int(np.argmax(theta))

    def update(self, a, r):
        self.mu[a] = (self.tau[a] * self.mu[a] + r) / (self.tau[a] + 1.0)
        self.tau[a] += 1.0


class GradientBandit:
    """Softmax over preferences H, stochastic gradient ascent on E[R]."""

    def __init__(self, k, rng, alpha=0.1, baseline=True):
        self.k, self.rng, self.alpha, self.baseline = k, rng, alpha, baseline
        self.H = np.zeros(k)
        self.r_bar, self.n = 0.0, 0
        self.pi = np.full(k, 1.0 / k)

    def act(self, t):
        z = np.exp(self.H - self.H.max())               # numerically stable softmax
        self.pi = z / z.sum()
        return int(self.rng.choice(self.k, p=self.pi))

    def update(self, a, r):
        b = self.r_bar if self.baseline else 0.0
        one_hot = np.zeros(self.k)
        one_hot[a] = 1.0
        self.H += self.alpha * (r - b) * (one_hot - self.pi)
        self.n += 1
        self.r_bar += (r - self.r_bar) / self.n        # running-average baseline


def run_testbed(make_agent, runs=200, steps=1000, k=10, mean=0.0, seed=0):
    """Average reward and %-optimal-action curves over many random testbeds."""
    rewards = np.zeros((runs, steps))
    optimal = np.zeros((runs, steps))
    for i in range(runs):
        rng = np.random.default_rng(seed + i)
        env = Testbed(k, rng, mean)
        agent = make_agent(k, rng)
        for t in range(steps):
            a = agent.act(t + 1)
            r = env.pull(a)
            agent.update(a, r)
            rewards[i, t] = r
            optimal[i, t] = (a == env.best)
    return rewards.mean(0), optimal.mean(0)


# ----------------------------------------------------------------------------
# Contextual bandit: disjoint LinUCB
# ----------------------------------------------------------------------------
class LinUCB:
    """One ridge regression per arm; pick argmax_a  x.theta_a + alpha*sqrt(x^T A_a^-1 x)."""

    def __init__(self, n_arms, d, alpha=1.0):
        self.alpha = alpha
        self.A = np.stack([np.eye(d) for _ in range(n_arms)])   # d x d per arm
        self.b = np.zeros((n_arms, d))

    def act(self, x):
        scores = []
        for A, b in zip(self.A, self.b):
            A_inv = np.linalg.inv(A)
            theta = A_inv @ b
            scores.append(x @ theta + self.alpha * np.sqrt(x @ A_inv @ x))
        return int(np.argmax(scores))

    def update(self, a, x, r):
        self.A[a] += np.outer(x, x)
        self.b[a] += r * x


def run_contextual(T=2000, n_arms=5, d=6, seed=0):
    rng = np.random.default_rng(seed)
    theta_true = rng.normal(0, 1, size=(n_arms, d))       # hidden per-arm weights
    linucb = LinUCB(n_arms, d, alpha=1.0)
    eps = EpsilonGreedy(n_arms, rng, eps=0.1)              # ignores the context
    regret = {"LinUCB": 0.0, "eps-greedy (no context)": 0.0}
    for _ in range(T):
        x = rng.normal(0, 1, size=d) / np.sqrt(d)          # context (e.g. user features)
        means = theta_true @ x
        best = means.max()
        a = linucb.act(x)
        linucb.update(a, x, means[a] + rng.normal(0, 0.1))
        regret["LinUCB"] += best - means[a]
        a = eps.act(0)
        eps.update(a, means[a] + rng.normal(0, 0.1))
        regret["eps-greedy (no context)"] += best - means[a]
    return regret


if __name__ == "__main__":
    agents = {
        "greedy (eps=0)":            lambda k, rng: EpsilonGreedy(k, rng, eps=0.0),
        "eps-greedy eps=0.1":        lambda k, rng: EpsilonGreedy(k, rng, eps=0.1),
        "eps-greedy eps=0.01":       lambda k, rng: EpsilonGreedy(k, rng, eps=0.01),
        "optimistic Q0=5, a=0.1":    lambda k, rng: EpsilonGreedy(k, rng, eps=0.0, q0=5.0, alpha=0.1),
        "UCB1 c=2":                  lambda k, rng: UCB1(k, rng, c=2.0),
        "Thompson (Gaussian)":       lambda k, rng: GaussianThompson(k, rng),
        "gradient a=0.1 +baseline":  lambda k, rng: GradientBandit(k, rng, alpha=0.1),
    }
    print("10-armed testbed, 200 runs x 1000 steps")
    print(f"{'agent':28s} {'avg reward':>10s} {'%opt (last 100)':>16s}")
    for name, make in agents.items():
        r, opt = run_testbed(make)
        print(f"{name:28s} {r.mean():10.3f} {100 * opt[-100:].mean():15.1f}%")

    # Baseline matters when rewards are far from 0 (S&B Fig. 2.5 uses mean +4).
    print("\nGradient bandit on testbed with q* mean = +4 (100 runs)")
    for base in (True, False):
        _, opt = run_testbed(lambda k, rng: GradientBandit(k, rng, 0.1, baseline=base),
                             runs=100, mean=4.0)
        print(f"  baseline={str(base):5s}  %opt (last 100) = {100 * opt[-100:].mean():.1f}%")

    print("\nContextual bandit (5 arms, 6-dim context, 2000 rounds): cumulative regret")
    for name, reg in run_contextual().items():
        print(f"  {name:26s} {reg:8.1f}")
