"""
Generative Adversarial Imitation Learning (GAIL, Ho & Ermon 2016) in miniature.

Task: goal-conditioned 2-D reaching. Observation o = (x, y, gx-x, gy-y); action = velocity in
[-1,1]^2.  The learner NEVER sees the true reward; it only gets 20 expert demonstrations.

  * Discriminator D(s,a) is trained to tell expert (s,a) pairs from policy pairs.
  * The policy is trained with PPO on the surrogate reward r(s,a) = -log(1 - D(s,a)).
  * For evaluation only, we measure the true task cost: distance to the goal.

Also compares with behavior cloning (BC) on the same demos and with a random policy.

Run:  python3 code/gail.py     (roughly 40-100 s on CPU)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
np.random.seed(0)
torch.manual_seed(0)

STEP, HORIZON, N_ENVS = 0.1, 25, 64


# ----------------------------------------------------------------------------
# 1) Vectorised NumPy environment + scripted expert
# ----------------------------------------------------------------------------
class ReachEnv:
    def __init__(self, n):
        self.n = n

    def reset(self):
        self.pos = np.random.uniform(0, 1, (self.n, 2))
        self.goal = np.random.uniform(0, 1, (self.n, 2))
        self.t = 0
        return self.obs()

    def obs(self):
        # position + goal offset (relative coordinates make the problem easier for both D and pi)
        return np.concatenate([self.pos, self.goal - self.pos], 1).astype(np.float32)

    def step(self, a):
        self.pos = np.clip(self.pos + STEP * np.clip(a, -1, 1), 0, 1)
        self.t += 1
        true_r = -np.linalg.norm(self.pos - self.goal, axis=1)  # only used for evaluation
        return self.obs(), true_r, self.t >= HORIZON


def expert_policy(o):
    d = o[:, 2:]                     # goal offset
    dist = np.linalg.norm(d, axis=1, keepdims=True) + 1e-8
    return d / dist * np.minimum(1.0, dist / STEP)  # full speed, then slow down exactly onto goal


def collect_expert(n_episodes=20):
    env, S, A = ReachEnv(n_episodes), [], []
    o, done = env.reset(), False
    while not done:
        a = expert_policy(o)
        S.append(o); A.append(a)
        o, _, done = env.step(a)
    return torch.tensor(np.concatenate(S)), torch.tensor(np.concatenate(A), dtype=torch.float32)


# ----------------------------------------------------------------------------
# 2) Networks
# ----------------------------------------------------------------------------
def mlp(i, o, h=64):
    return nn.Sequential(nn.Linear(i, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, o))


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.mu, self.v = mlp(4, 2), mlp(4, 1)
        self.log_std = nn.Parameter(torch.full((2,), -0.5))

    def dist(self, o):
        return torch.distributions.Normal(self.mu(o), self.log_std.exp())


# ----------------------------------------------------------------------------
# 3) GAIL pieces: discriminator loss and surrogate reward
# ----------------------------------------------------------------------------
def discriminator_loss(D, s_exp, a_exp, s_pol, a_pol):
    """Binary cross-entropy: expert pairs -> label 1, policy pairs -> label 0."""
    logit_e = D(torch.cat([s_exp, a_exp], 1))
    logit_p = D(torch.cat([s_pol, a_pol], 1))
    return (F.binary_cross_entropy_with_logits(logit_e, torch.ones_like(logit_e)) +
            F.binary_cross_entropy_with_logits(logit_p, torch.zeros_like(logit_p)))


@torch.no_grad()
def gail_reward(D, s, a):
    """r = -log(1 - D(s,a)) = softplus(logit).  High when the pair 'looks expert'."""
    return F.softplus(D(torch.cat([s, a], 1))).squeeze(-1)


def ppo_update(pi, opt, obs, act, logp_old, adv, ret, epochs=4, mb=512, clip=0.2):
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    n = obs.shape[0]
    for _ in range(epochs):
        for idx in torch.randperm(n).split(mb):
            d = pi.dist(obs[idx])
            logp = d.log_prob(act[idx]).sum(-1)
            ratio = (logp - logp_old[idx]).exp()
            pg = -torch.min(ratio * adv[idx], ratio.clamp(1 - clip, 1 + clip) * adv[idx]).mean()
            v_loss = F.mse_loss(pi.v(obs[idx]).squeeze(-1), ret[idx])
            loss = pg + 0.5 * v_loss          # (an entropy bonus is optional here)
            opt.zero_grad(); loss.backward(); opt.step()


def gae(rew, val, last_val, gamma=0.99, lam=0.95):
    T = rew.shape[0]
    adv, last = torch.zeros_like(rew), torch.zeros(rew.shape[1])
    for t in reversed(range(T)):
        nxt = last_val if t == T - 1 else val[t + 1]
        delta = rew[t] + gamma * nxt - val[t]   # fixed horizon: bootstrap the time-limit
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + val


def train_gail(s_exp, a_exp, iters=300):
    pi, D = Policy(), mlp(6, 1)
    opt_pi = torch.optim.Adam(pi.parameters(), lr=1e-3)
    opt_d = torch.optim.Adam(D.parameters(), lr=1e-3)
    env = ReachEnv(N_ENVS)
    for it in range(iters):
        O, A, LP, V = [], [], [], []
        o, done = env.reset(), False
        while not done:  # roll out the current policy
            ot = torch.tensor(o)
            with torch.no_grad():
                d = pi.dist(ot); a = d.sample()
                O.append(ot); A.append(a); LP.append(d.log_prob(a).sum(-1)); V.append(pi.v(ot).squeeze(-1))
            o, _, done = env.step(a.numpy())
        O, A, LP, V = map(torch.stack, (O, A, LP, V))       # [T, N, ...]
        flatO, flatA = O.reshape(-1, 4), A.reshape(-1, 2)
        # (1) a few discriminator steps (a weak D gives an uninformative reward)
        for _ in range(5):
            idx = torch.randint(0, s_exp.shape[0], (flatO.shape[0],))
            ld = discriminator_loss(D, s_exp[idx], a_exp[idx], flatO, flatA.clamp(-1, 1))
            opt_d.zero_grad(); ld.backward(); opt_d.step()
        # (2) PPO step on the discriminator reward
        R = gail_reward(D, flatO, flatA.clamp(-1, 1)).reshape(HORIZON, N_ENVS)
        with torch.no_grad():
            last_v = pi.v(torch.tensor(o)).squeeze(-1)
        adv, ret = gae(R, V, last_v)
        ppo_update(pi, opt_pi, flatO, flatA, LP.reshape(-1), adv.reshape(-1), ret.reshape(-1))
        if it % 50 == 0 or it == iters - 1:
            print(f"  iter {it:3d}  D loss {ld.item():.3f}   mean GAIL reward {R.mean():.3f}   "
                  f"true final dist {evaluate(lambda x: pi.mu(torch.tensor(x)).detach().numpy())[1]:.3f}")
    return pi


def train_bc(s_exp, a_exp, steps=2000):
    pi = Policy()
    opt = torch.optim.Adam(pi.parameters(), lr=1e-3)
    for _ in range(steps):
        idx = torch.randint(0, s_exp.shape[0], (256,))
        loss = F.mse_loss(pi.mu(s_exp[idx]), a_exp[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    return pi


def evaluate(policy_fn, n=500):
    np_state = np.random.get_state()
    np.random.seed(123)  # same evaluation goals for every method
    env = ReachEnv(n)
    o, done, G = env.reset(), False, 0.0
    while not done:
        o, r, done = env.step(policy_fn(o))
        G += r
    np.random.set_state(np_state)
    return float(G.mean()), float(np.linalg.norm(env.pos - env.goal, axis=1).mean())


if __name__ == "__main__":
    s_exp, a_exp = collect_expert()
    print(f"Expert demos: {s_exp.shape[0]} (s,a) pairs from 20 episodes\n")
    print("Training GAIL ...")
    pi_gail = train_gail(s_exp, a_exp)
    pi_bc = train_bc(s_exp, a_exp)
    rows = [("expert", expert_policy),
            ("random", lambda o: np.random.uniform(-1, 1, (o.shape[0], 2))),
            ("behavior cloning", lambda o: pi_bc.mu(torch.tensor(o)).detach().numpy()),
            ("GAIL (PPO + discriminator)", lambda o: pi_gail.mu(torch.tensor(o)).detach().numpy())]
    print(f"\n{'policy':<28}{'true return':>12}{'final dist':>12}")
    for name, fn in rows:
        G, dist = evaluate(fn)
        print(f"{name:<28}{G:>12.2f}{dist:>12.3f}")
    print("\nGAIL never saw the true reward (distance); it learned purely by trying to make its\n"
          "(state, action) pairs indistinguishable from the expert's.")
