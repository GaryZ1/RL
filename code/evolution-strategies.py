"""
Black-box policy search on a NumPy CartPole: Cross-Entropy Method (CEM),
OpenAI-style Evolution Strategies (Salimans et al., 2017) and CMA-ES (Hansen).

The policy is a tiny MLP (4 -> 8 -> 1, 49 parameters). None of the methods uses
gradients of the policy or the environment - only episode returns ("fitness").

Run:  python3 code/evolution-strategies.py     (roughly 10-40 s on CPU)
"""
import numpy as np

# ----------------------------------------------------------------------------
# 1) Vectorised CartPole (same physics constants as the classic Barto et al. task)
# ----------------------------------------------------------------------------
G, MC, MP, L, F_MAG, TAU = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02
X_LIM, TH_LIM, MAX_STEPS = 2.4, 12 * 2 * np.pi / 360, 500


def cartpole_returns(params, n_episodes=3, seed=0):
    """Evaluate a batch of parameter vectors [P, D]; returns mean episode length [P].
    All candidates use the SAME start states (common random numbers -> fair ranking)."""
    P = params.shape[0]
    rng = np.random.default_rng(seed)
    s = np.repeat(rng.uniform(-0.05, 0.05, (1, n_episodes, 4)), P, 0)  # [P, E, 4]
    alive = np.ones((P, n_episodes), bool)
    ret = np.zeros((P, n_episodes))
    W1, b1, W2, b2 = unpack(params)
    for _ in range(MAX_STEPS):
        h = np.tanh(np.einsum("ped,phd->peh", s, W1) + b1[:, None, :])
        out = np.einsum("peh,ph->pe", h, W2) + b2[:, None]
        force = np.where(out > 0, F_MAG, -F_MAG)
        x, xd, th, thd = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
        ct, st = np.cos(th), np.sin(th)
        temp = (force + MP * L * thd ** 2 * st) / (MC + MP)
        thacc = (G * st - ct * temp) / (L * (4.0 / 3.0 - MP * ct ** 2 / (MC + MP)))
        xacc = temp - MP * L * thacc * ct / (MC + MP)
        s = np.stack([x + TAU * xd, xd + TAU * xacc, th + TAU * thd, thd + TAU * thacc], -1)
        ret += alive
        alive &= (np.abs(s[..., 0]) < X_LIM) & (np.abs(s[..., 2]) < TH_LIM)
        if not alive.any():
            break
    return ret.mean(1)


HID = 8
D = 4 * HID + HID + HID + 1  # 49 parameters


def unpack(p):
    P = p.shape[0]
    i = 0
    W1 = p[:, i:i + 4 * HID].reshape(P, HID, 4); i += 4 * HID
    b1 = p[:, i:i + HID]; i += HID
    W2 = p[:, i:i + HID]; i += HID
    b2 = p[:, i]
    return W1, b1, W2, b2


# ----------------------------------------------------------------------------
# 2) Cross-Entropy Method: sample, keep the elite, refit a diagonal Gaussian
# ----------------------------------------------------------------------------
def cem(gens=30, pop=50, elite_frac=0.2, init_std=1.0, extra_std=0.05, seed=0):
    rng = np.random.default_rng(seed)
    mu, std = np.zeros(D), np.full(D, init_std)
    n_elite, hist = int(pop * elite_frac), []
    for g in range(gens):
        X = mu + std * rng.standard_normal((pop, D))
        f = cartpole_returns(X, seed=g)
        elite = X[np.argsort(-f)[:n_elite]]
        mu, std = elite.mean(0), elite.std(0) + extra_std  # extra noise avoids premature collapse
        hist.append(f.max())
    return mu, hist


# ----------------------------------------------------------------------------
# 3) OpenAI-ES: antithetic Gaussian perturbations, centred ranks, Adam on the mean
# ----------------------------------------------------------------------------
def centered_ranks(f):
    r = np.empty_like(f)
    r[np.argsort(f)] = np.arange(len(f))
    return r / (len(f) - 1) - 0.5


def openai_es(gens=30, half_pop=25, sigma=0.1, lr=0.05, wd=0.005, seed=0):
    rng = np.random.default_rng(seed)
    theta, m, v, hist = np.zeros(D), np.zeros(D), np.zeros(D), []
    for g in range(gens):
        eps = rng.standard_normal((half_pop, D))
        eps = np.concatenate([eps, -eps])                    # mirrored sampling
        f = cartpole_returns(theta + sigma * eps, seed=g)
        grad = (centered_ranks(f) @ eps) / (len(f) * sigma)  # ES gradient estimate of E[F(theta + sigma eps)]
        grad -= wd * theta
        m = 0.9 * m + 0.1 * grad                             # Adam (ascent)
        v = 0.999 * v + 0.001 * grad ** 2
        theta += lr * (m / (1 - 0.9 ** (g + 1))) / (np.sqrt(v / (1 - 0.999 ** (g + 1))) + 1e-8)
        hist.append(f.max())
    return theta, hist


# ----------------------------------------------------------------------------
# 4) CMA-ES (compact (mu/mu_w, lambda) version following Hansen's tutorial)
# ----------------------------------------------------------------------------
def cma_es(gens=60, sigma=0.5, seed=0):
    rng = np.random.default_rng(seed)
    n = D
    lam = 4 + int(3 * np.log(n))
    mu_ = lam // 2
    w = np.log(mu_ + 0.5) - np.log(np.arange(1, mu_ + 1))
    w /= w.sum()
    mueff = 1 / (w ** 2).sum()
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    cs = (mueff + 2) / (n + mueff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
    mean, pc, ps, C = np.zeros(n), np.zeros(n), np.zeros(n), np.eye(n)
    hist = []
    for g in range(gens):
        evals, B = np.linalg.eigh(C)
        Dg = np.sqrt(np.maximum(evals, 1e-20))
        z = rng.standard_normal((lam, n))
        y = z * Dg @ B.T                                  # y ~ N(0, C)
        X = mean + sigma * y
        f = cartpole_returns(X, seed=g)
        idx = np.argsort(-f)[:mu_]                        # maximise fitness
        y_w = w @ y[idx]
        mean = mean + sigma * y_w
        C_inv_sqrt = B @ np.diag(1 / Dg) @ B.T
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * C_inv_sqrt @ y_w
        hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (g + 1))) / chiN < 1.4 + 2 / (n + 1)
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * y_w
        C = ((1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
             + cmu * (y[idx].T * w) @ y[idx])
        sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
        hist.append(f.max())
    return mean, hist, lam


if __name__ == "__main__":
    print(f"CartPole (max {MAX_STEPS} steps), MLP policy with {D} parameters\n")
    results = {}
    th, h = cem()
    results["CEM (pop 50, 30 gens)"] = (th, h, 50 * 30)
    th, h = openai_es()
    results["OpenAI-ES (pop 50, 30 gens)"] = (th, h, 50 * 30)
    th, h, lam = cma_es()
    results[f"CMA-ES (pop {lam}, 60 gens)"] = (th, h, lam * 60)
    print(f"{'method':<30}{'candidates':>11}{'best fit @10%':>15}{'@50%':>7}{'@100%':>7}{'final mean (20 eps)':>21}")
    for name, (th, h, n_eval) in results.items():
        final = cartpole_returns(th[None], n_episodes=20, seed=999)[0]
        q = lambda frac: h[max(0, int(len(h) * frac) - 1)]
        print(f"{name:<30}{n_eval:>11}{q(0.1):>15.0f}{q(0.5):>7.0f}{q(1.0):>7.0f}{final:>21.1f}")
    print("\nFitness = mean episode length over 3 start states (500 = balanced the whole time).")
