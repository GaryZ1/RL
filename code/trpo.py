"""
TRPO (Schulman et al. 2015) on a self-contained NumPy CartPole.

Key pieces implemented from scratch:
  * Fisher-vector product via double back-prop through the mean KL (no explicit Fisher matrix)
  * conjugate gradient to solve F x = g
  * step scaling to the KL trust-region boundary + backtracking line search
  * GAE advantages and a separately-trained value function
Run:  python3 code/trpo.py      (about 30-60 s on CPU)
"""
import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class CartPole:
    """Classic cart-pole physics (same constants as Gym CartPole), capped at max_steps."""
    def __init__(self, max_steps=200, seed=0):
        self.gravity, self.m_cart, self.m_pole = 9.8, 1.0, 0.1
        self.length, self.force_mag, self.tau = 0.5, 10.0, 0.02
        self.x_limit, self.theta_limit = 2.4, 12 * 2 * math.pi / 360
        self.max_steps, self.rng = max_steps, np.random.default_rng(seed)
        self.obs_dim, self.n_actions = 4, 2

    def reset(self):
        self.state, self.t = self.rng.uniform(-0.05, 0.05, size=4), 0
        return self.state.astype(np.float32)

    def step(self, action):
        x, x_dot, th, th_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        cos, sin = math.cos(th), math.sin(th)
        total_m, pm_l = self.m_cart + self.m_pole, self.m_pole * self.length
        temp = (force + pm_l * th_dot ** 2 * sin) / total_m
        th_acc = (self.gravity * sin - cos * temp) / (
            self.length * (4.0 / 3.0 - self.m_pole * cos ** 2 / total_m))
        x_acc = temp - pm_l * th_acc * cos / total_m
        x, x_dot = x + self.tau * x_dot, x_dot + self.tau * x_acc
        th, th_dot = th + self.tau * th_dot, th_dot + self.tau * th_acc
        self.state, self.t = np.array([x, x_dot, th, th_dot]), self.t + 1
        terminated = bool(abs(x) > self.x_limit or abs(th) > self.theta_limit)
        return self.state.astype(np.float32), 1.0, terminated, self.t >= self.max_steps


class VecCartPole:
    """N CartPoles stepped in lock-step with auto-reset; also returns the true final obs."""
    def __init__(self, n, seed=0):
        self.envs = [CartPole(seed=seed + i) for i in range(n)]
        self.ep_ret, self.finished = np.zeros(n), []

    def reset(self):
        return np.stack([e.reset() for e in self.envs])

    def step(self, actions):
        obs, rew, term, trunc, final_obs = [], [], [], [], []
        for i, (env, a) in enumerate(zip(self.envs, actions)):
            o, r, te, tr = env.step(int(a))
            self.ep_ret[i] += r
            final_obs.append(o)
            if te or tr:
                self.finished.append(self.ep_ret[i]); self.ep_ret[i] = 0.0
                o = env.reset()
            obs.append(o); rew.append(r); term.append(te); trunc.append(tr)
        return (np.stack(obs), np.array(rew, np.float32), np.array(term), np.array(trunc),
                np.stack(final_obs))


def compute_gae(rewards, values, dones, last_values, gamma=0.99, lam=0.97):
    T, adv = rewards.shape[0], torch.zeros_like(rewards)
    gae = torch.zeros_like(last_values)
    for t in reversed(range(T)):
        next_v = last_values if t == T - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        adv[t] = gae
    return adv, adv + values


# ----------------------------------------------------------------------------
# The TRPO-specific numerics
# ----------------------------------------------------------------------------
def conjugate_gradient(Avp, b, n_iters=10, residual_tol=1e-10):
    """Solve A x = b for symmetric positive-definite A, given only v -> A v (Hestenes-Stiefel)."""
    x = torch.zeros_like(b)
    r = b.clone()              # residual b - A x  (x = 0)
    p = b.clone()              # search direction
    rr = r @ r
    for _ in range(n_iters):
        Ap = Avp(p)
        alpha = rr / (p @ Ap)
        x += alpha * p
        r -= alpha * Ap
        rr_new = r @ r
        if rr_new < residual_tol:
            break
        p = r + (rr_new / rr) * p
        rr = rr_new
    return x


def mean_kl(policy, obs, old_logits):
    """KL(pi_old || pi_theta) averaged over states, for categorical policies."""
    old_logp = torch.log_softmax(old_logits, -1)
    new_logp = torch.log_softmax(policy(obs), -1)
    return (old_logp.exp() * (old_logp - new_logp)).sum(-1).mean()


def fisher_vector_product(policy, obs, old_logits, v, damping=0.1):
    """F v = d/dtheta [ (d KL / d theta) . v ]  evaluated at theta_old  (+ damping * v)."""
    kl = mean_kl(policy, obs, old_logits)
    grads = torch.autograd.grad(kl, list(policy.parameters()), create_graph=True)
    flat_grad = torch.cat([g.reshape(-1) for g in grads])
    hvp = torch.autograd.grad(flat_grad @ v, list(policy.parameters()))
    return torch.cat([g.reshape(-1) for g in hvp]) + damping * v


def surrogate(policy, obs, acts, adv, old_logp):
    """L(theta) = E[ pi_theta(a|s) / pi_old(a|s) * A ]."""
    logp = torch.distributions.Categorical(logits=policy(obs)).log_prob(acts)
    return (torch.exp(logp - old_logp) * adv).mean()


def trpo_step(policy, obs, acts, adv, max_kl=0.01, damping=0.1, cg_iters=10,
              backtrack_iters=10, backtrack_coeff=0.5):
    with torch.no_grad():
        old_logits = policy(obs)
        old_logp = torch.distributions.Categorical(logits=old_logits).log_prob(acts)
    # 1) policy gradient g of the surrogate at theta_old
    loss = surrogate(policy, obs, acts, adv, old_logp)
    g = torch.cat([x.reshape(-1) for x in torch.autograd.grad(loss, list(policy.parameters()))])
    # 2) natural-gradient direction s = F^{-1} g by conjugate gradient
    Fvp = lambda v: fisher_vector_product(policy, obs, old_logits, v, damping)
    s = conjugate_gradient(Fvp, g, cg_iters)
    # 3) largest step on the quadratic KL model: 0.5 * beta^2 s^T F s = max_kl
    sFs = s @ Fvp(s)
    full_step = torch.sqrt(2 * max_kl / (sFs + 1e-8)) * s
    expected_improve = g @ full_step
    # 4) backtracking line search: accept if KL <= max_kl AND surrogate improves
    theta_old = parameters_to_vector(policy.parameters()).detach().clone()
    with torch.no_grad():
        for i in range(backtrack_iters):
            frac = backtrack_coeff ** i
            vector_to_parameters(theta_old + frac * full_step, policy.parameters())
            new_loss = surrogate(policy, obs, acts, adv, old_logp)
            kl = mean_kl(policy, obs, old_logits)
            if kl <= max_kl and new_loss - loss > 0:
                return {"kl": kl.item(), "improve": (new_loss - loss).item(),
                        "expected": (frac * expected_improve).item(), "backtracks": i}
        vector_to_parameters(theta_old, policy.parameters())     # no acceptable step: reject
    return {"kl": 0.0, "improve": 0.0, "expected": expected_improve.item(), "backtracks": -1}


def mlp(n_in, n_out, hidden=64):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                         nn.Linear(hidden, n_out))


def evaluate(policy, n_episodes=10, seed=123):
    env, total = CartPole(seed=seed), 0.0
    for _ in range(n_episodes):
        o, done = env.reset(), False
        while not done:
            with torch.no_grad():
                a = int(policy(torch.as_tensor(o)).argmax())
            o, r, term, trunc = env.step(a)
            total, done = total + r, term or trunc
    return total / n_episodes


def train(n_iters=40, n_envs=8, n_steps=256, gamma=0.99, lam=0.97, max_kl=0.01, vf_lr=1e-3,
          vf_epochs=5, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    envs = VecCartPole(n_envs, seed=seed)
    policy, critic = mlp(4, 2), mlp(4, 1)
    v_opt = torch.optim.Adam(critic.parameters(), lr=vf_lr)
    before = evaluate(policy)
    obs = torch.as_tensor(envs.reset())
    for it in range(1, n_iters + 1):
        b_obs, b_act, b_rew, b_done = [], [], [], []
        for _ in range(n_steps):
            with torch.no_grad():
                a = torch.distributions.Categorical(logits=policy(obs)).sample()
            nobs, r, term, trunc, final_obs = envs.step(a.numpy())
            r = torch.as_tensor(r)
            if trunc.any():   # bootstrap through time-limit truncation
                with torch.no_grad():
                    r = r + gamma * critic(torch.as_tensor(final_obs)).squeeze(-1) * \
                        torch.as_tensor(trunc & ~term, dtype=torch.float32)
            b_obs.append(obs); b_act.append(a); b_rew.append(r)
            b_done.append(torch.as_tensor(term | trunc, dtype=torch.float32))
            obs = torch.as_tensor(nobs)
        b_obs, b_act = torch.stack(b_obs), torch.stack(b_act)
        b_rew, b_done = torch.stack(b_rew), torch.stack(b_done)
        with torch.no_grad():
            adv, ret = compute_gae(b_rew, critic(b_obs).squeeze(-1), b_done,
                                   critic(obs).squeeze(-1), gamma, lam)
        flat_obs, flat_act = b_obs.reshape(-1, 4), b_act.reshape(-1)
        flat_adv, flat_ret = adv.reshape(-1), ret.reshape(-1)
        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)

        info = trpo_step(policy, flat_obs, flat_act, flat_adv, max_kl=max_kl)

        # value function: plain regression on GAE returns (minibatch Adam)
        for _ in range(vf_epochs):
            for idx in torch.randperm(len(flat_ret)).split(256):
                v_loss = ((critic(flat_obs[idx]).squeeze(-1) - flat_ret[idx]) ** 2).mean()
                v_opt.zero_grad(); v_loss.backward(); v_opt.step()
        if it % 5 == 0:
            print(f"iter {it:3d}  avg return (last 20 eps) {np.mean(envs.finished[-20:]):6.1f}  "
                  f"KL {info['kl']:.4f}  surrogate gain {info['improve']:.4f} "
                  f"(predicted {info['expected']:.4f})  backtracks {info['backtracks']}")
    return before, np.mean(envs.finished[-20:]), evaluate(policy)


def _sanity_checks():
    # conjugate gradient solves a random SPD system using only matrix-vector products
    torch.manual_seed(0)
    M = torch.randn(20, 20, dtype=torch.float64)
    A = M @ M.T + 20 * torch.eye(20, dtype=torch.float64)
    b = torch.randn(20, dtype=torch.float64)
    x = conjugate_gradient(lambda v: A @ v, b, n_iters=50, residual_tol=1e-20)
    assert torch.allclose(x, torch.linalg.solve(A, b), atol=1e-8)
    # Fisher-vector product matches an explicit Hessian of the KL for a tiny policy
    pol = nn.Linear(3, 2)
    obs = torch.randn(16, 3)
    with torch.no_grad():
        old_logits = pol(obs)
    v = torch.randn(sum(p.numel() for p in pol.parameters()))
    def kl_of(flat):
        W, bb = flat[:6].view(2, 3), flat[6:]
        old_logp = torch.log_softmax(old_logits, -1)
        return (old_logp.exp() * (old_logp - torch.log_softmax(obs @ W.T + bb, -1))).sum(-1).mean()
    H = torch.autograd.functional.hessian(kl_of, parameters_to_vector(pol.parameters()).detach())
    assert torch.allclose(fisher_vector_product(pol, obs, old_logits, v, damping=0.0), H @ v, atol=1e-5)
    print("sanity checks passed (CG == linalg.solve, Fisher-vector product == KL Hessian @ v)")


if __name__ == "__main__":
    torch.set_num_threads(1)
    _sanity_checks()
    before, train_ret, after = train()
    print(f"\nTRPO greedy return before training: {before:.1f}")
    print(f"TRPO avg training-episode return at the end: {train_ret:.1f}")
    print(f"TRPO greedy return after training : {after:.1f}   (max 200)")
