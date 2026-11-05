"""
PPO-Clip (Schulman et al. 2017) on a self-contained NumPy CartPole, following the
CleanRL-style "implementation details" (Huang et al., ICLR Blog Track 2022):
  vectorized envs, GAE, advantage normalization (per minibatch), clipped surrogate,
  clipped value loss, entropy bonus, several epochs of shuffled minibatches, global
  gradient clipping, Adam with eps=1e-5 and linear LR annealing, orthogonal init,
  correct bootstrapping on time-limit truncation, approx-KL / clip-fraction diagnostics.
Run:  python3 code/ppo.py      (about 30-60 s on CPU)
"""
import math

import numpy as np
import torch
import torch.nn as nn


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


def layer_init(layer, std=math.sqrt(2), bias=0.0):
    nn.init.orthogonal_(layer.weight, std)        # detail: orthogonal init, small policy-head scale
    nn.init.constant_(layer.bias, bias)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, obs_dim=4, n_actions=2, hidden=64):
        super().__init__()
        self.critic = nn.Sequential(layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
                                    layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
                                    layer_init(nn.Linear(hidden, 1), std=1.0))
        self.actor = nn.Sequential(layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
                                   layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
                                   layer_init(nn.Linear(hidden, n_actions), std=0.01))

    def value(self, x):
        return self.critic(x).squeeze(-1)

    def act(self, x, action=None):
        dist = torch.distributions.Categorical(logits=self.actor(x))
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.value(x)


def compute_gae(rewards, values, dones, last_values, gamma=0.99, lam=0.95):
    T, adv = rewards.shape[0], torch.zeros_like(rewards)
    gae = torch.zeros_like(last_values)
    for t in reversed(range(T)):
        next_v = last_values if t == T - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        adv[t] = gae
    return adv, adv + values


def ppo_loss(new_logp, old_logp, adv, new_values, old_values, returns, entropy,
             clip_coef=0.2, vf_coef=0.5, ent_coef=0.01, clip_vloss=True):
    """The PPO objective on a minibatch of flat tensors. Returns (loss, diagnostics dict)."""
    log_ratio = new_logp - old_logp
    ratio = log_ratio.exp()                                       # r_t(theta)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)                 # per-minibatch normalization
    pg_loss1 = -adv * ratio
    pg_loss2 = -adv * ratio.clamp(1 - clip_coef, 1 + clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()                # = -min(r A, clip(r) A)
    if clip_vloss:                                                # value clipping (PPO2 style)
        v_clipped = old_values + (new_values - old_values).clamp(-clip_coef, clip_coef)
        v_loss = 0.5 * torch.max((new_values - returns) ** 2, (v_clipped - returns) ** 2).mean()
    else:
        v_loss = 0.5 * ((new_values - returns) ** 2).mean()
    ent = entropy.mean()
    loss = pg_loss + vf_coef * v_loss - ent_coef * ent
    with torch.no_grad():
        approx_kl = ((ratio - 1) - log_ratio).mean()              # low-variance KL estimator
        clipfrac = ((ratio - 1).abs() > clip_coef).float().mean()
    return loss, {"pg": pg_loss.item(), "v": v_loss.item(), "ent": ent.item(),
                  "kl": approx_kl.item(), "clipfrac": clipfrac.item()}


def evaluate(agent, n_episodes=10, seed=123):
    env, total = CartPole(seed=seed), 0.0
    for _ in range(n_episodes):
        o, done = env.reset(), False
        while not done:
            with torch.no_grad():
                a = int(agent.actor(torch.as_tensor(o)).argmax())
            o, r, term, trunc = env.step(a)
            total, done = total + r, term or trunc
    return total / n_episodes


def train(total_steps=100_000, n_envs=8, n_steps=128, n_epochs=4, n_minibatches=4,
          lr=1e-3, gamma=0.99, lam=0.95, clip_coef=0.2, ent_coef=0.01, vf_coef=0.5,
          max_grad_norm=0.5, target_kl=None, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    envs, agent = VecCartPole(n_envs, seed=seed), ActorCritic()
    opt = torch.optim.Adam(agent.parameters(), lr=lr, eps=1e-5)
    batch_size = n_envs * n_steps
    mb_size, n_updates = batch_size // n_minibatches, total_steps // batch_size
    before = evaluate(agent)
    obs = torch.as_tensor(envs.reset())
    for update in range(1, n_updates + 1):
        opt.param_groups[0]["lr"] = lr * (1.0 - (update - 1) / n_updates)  # linear LR annealing
        # ------------------------------ rollout ------------------------------
        b_obs, b_act, b_logp, b_rew, b_done, b_val = [], [], [], [], [], []
        for _ in range(n_steps):
            with torch.no_grad():
                a, logp, _, v = agent.act(obs)
            nobs, r, term, trunc, final_obs = envs.step(a.numpy())
            r = torch.as_tensor(r)
            if trunc.any():   # time-limit truncation is not death: bootstrap with V(final obs)
                with torch.no_grad():
                    r = r + gamma * agent.value(torch.as_tensor(final_obs)) * \
                        torch.as_tensor(trunc & ~term, dtype=torch.float32)
            b_obs.append(obs); b_act.append(a); b_logp.append(logp); b_val.append(v)
            b_rew.append(r); b_done.append(torch.as_tensor(term | trunc, dtype=torch.float32))
            obs = torch.as_tensor(nobs)
        stack = lambda xs: torch.stack(xs)
        b_obs, b_act, b_logp, b_val = stack(b_obs), stack(b_act), stack(b_logp), stack(b_val)
        b_rew, b_done = stack(b_rew), stack(b_done)
        with torch.no_grad():
            adv, ret = compute_gae(b_rew, b_val, b_done, agent.value(obs), gamma, lam)
        # flatten [T, N, ...] -> [T*N, ...]
        f_obs, f_act, f_logp = b_obs.reshape(-1, 4), b_act.reshape(-1), b_logp.reshape(-1)
        f_adv, f_ret, f_val = adv.reshape(-1), ret.reshape(-1), b_val.reshape(-1)

        # ------------------- K epochs of shuffled minibatch SGD -------------------
        for epoch in range(n_epochs):
            for idx in torch.randperm(batch_size).split(mb_size):
                _, new_logp, entropy, new_v = agent.act(f_obs[idx], f_act[idx])
                loss, info = ppo_loss(new_logp, f_logp[idx], f_adv[idx], new_v, f_val[idx],
                                      f_ret[idx], entropy, clip_coef, vf_coef, ent_coef)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                opt.step()
            if target_kl is not None and info["kl"] > target_kl:   # optional early stopping
                break
        if update % 10 == 0:
            print(f"update {update:3d}  steps {update * batch_size:6d}  "
                  f"avg return (last 20 eps) {np.mean(envs.finished[-20:]):6.1f}  "
                  f"approx_kl {info['kl']:.4f}  clipfrac {info['clipfrac']:.3f}  "
                  f"entropy {info['ent']:.3f}")
    return before, np.mean(envs.finished[-20:]), evaluate(agent)


def _sanity_checks():
    # The clipped objective has zero gradient once the ratio leaves [1-eps, 1+eps] in the
    # direction the advantage favours (that is what stops the policy from moving too far).
    old_logp = torch.zeros(2)
    new_logp = torch.log(torch.tensor([1.5, 0.5])).requires_grad_()   # ratios 1.5 and 0.5
    adv = torch.tensor([1.0, -1.0])     # (normalization keeps the signs for this pair)
    zeros = torch.zeros(2)
    loss, _ = ppo_loss(new_logp, old_logp, adv, zeros, zeros, zeros, zeros, ent_coef=0.0)
    loss.backward()
    assert torch.allclose(new_logp.grad, torch.zeros(2)), new_logp.grad
    # inside the trust region the gradient is the ordinary importance-weighted PG
    new_logp2 = torch.zeros(2, requires_grad=True)
    loss2, _ = ppo_loss(new_logp2, old_logp, adv, zeros, zeros, zeros, zeros, ent_coef=0.0)
    loss2.backward()
    assert (new_logp2.grad.abs() > 0).all()
    print("sanity checks passed (clipped surrogate kills gradients outside the clip range)")


if __name__ == "__main__":
    torch.set_num_threads(1)
    _sanity_checks()
    before, train_ret, after = train()
    print(f"\nPPO greedy return before training: {before:.1f}")
    print(f"PPO avg training-episode return at the end: {train_ret:.1f}")
    print(f"PPO greedy return after training : {after:.1f}   (max 200)")
