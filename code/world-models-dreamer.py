"""A simplified Dreamer (Hafner et al.): RSSM world model + actor-critic trained purely in imagination.

Toy env: a 2-D point mass that must be driven to the origin (continuous actions).
What is kept from Dreamer V1/V2:
  * RSSM: deterministic GRU path h_t + stochastic Gaussian latent z_t (prior p(z|h), posterior q(z|h,o))
  * world-model loss = reconstruction + reward prediction + KL(posterior || prior) with free nats
    and KL balancing (V2)
  * actor-critic trained on H-step latent "dreams", lambda-returns, actor gradients backpropagated
    through the learned dynamics (V1 style, continuous actions), slow target critic
Simplified: vector observations (no CNN), Gaussian (not categorical) latents, no continue head
(the env never terminates), tiny networks.  numpy + torch, ~1-2 minutes on CPU.
    python3 code/world-models-dreamer.py
"""
import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

torch.set_num_threads(1)
np.random.seed(0)
torch.manual_seed(0)


# ----------------------------------------------------------------------------------------------
# 1. Environment: point mass in 2-D, reward = -distance to origin
# ----------------------------------------------------------------------------------------------
class PointMass:
    obs_dim, act_dim, horizon = 4, 2, 50

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.p = self.rng.uniform(-1.5, 1.5, size=2)
        self.v = np.zeros(2)
        self.t = 0
        return self._obs()

    def _obs(self):  # slightly noisy observation of position & velocity
        return (np.concatenate([self.p, self.v]) + 0.01 * self.rng.standard_normal(4)).astype(np.float32)

    def step(self, a):
        a = np.clip(a, -1, 1)
        self.v = 0.9 * self.v + 0.1 * a                       # damped acceleration
        self.p = np.clip(self.p + 0.2 * self.v, -2, 2)
        self.t += 1
        reward = -np.linalg.norm(self.p) - 0.01 * np.sum(a ** 2)
        return self._obs(), float(reward), False, self.t >= self.horizon


# ----------------------------------------------------------------------------------------------
# 2. RSSM world model
# ----------------------------------------------------------------------------------------------
H_DIM, Z_DIM, HID = 64, 8, 128


def mlp(inp, out, hid=HID):
    return nn.Sequential(nn.Linear(inp, hid), nn.ELU(), nn.Linear(hid, hid), nn.ELU(), nn.Linear(hid, out))


class RSSM(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.encoder = mlp(obs_dim, HID)
        self.inp = nn.Linear(Z_DIM + act_dim, HID)
        self.gru = nn.GRUCell(HID, H_DIM)                     # deterministic path  h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
        self.prior_net = mlp(H_DIM, 2 * Z_DIM)                # p(z_t | h_t)          -- used when dreaming
        self.post_net = mlp(H_DIM + HID, 2 * Z_DIM)           # q(z_t | h_t, o_t)     -- used when observing
        self.decoder = mlp(H_DIM + Z_DIM, obs_dim)            # p(o_t | h_t, z_t)
        self.reward = mlp(H_DIM + Z_DIM, 1)                   # p(r_t | h_t, z_t)

    @staticmethod
    def _dist(x):
        mean, std = x.chunk(2, -1)
        return Normal(mean, F.softplus(std) + 0.1)

    def initial(self, batch):
        return torch.zeros(batch, H_DIM), torch.zeros(batch, Z_DIM)

    def img_step(self, h, z, a):
        """One step of the latent dynamics WITHOUT an observation (the 'dream' step)."""
        h = self.gru(F.elu(self.inp(torch.cat([z, a], -1))), h)
        prior = self._dist(self.prior_net(h))
        return h, prior

    def obs_step(self, h, z, a, obs):
        """One filtering step WITH an observation: prior from the dynamics, posterior corrected by o_t."""
        h, prior = self.img_step(h, z, a)
        post = self._dist(self.post_net(torch.cat([h, self.encoder(obs)], -1)))
        return h, prior, post

    def observe(self, obs, act):
        """obs: (B, L, obs_dim), act: (B, L, act_dim) where act[:, t] is the action taken BEFORE obs[:, t]."""
        B, L, _ = obs.shape
        h, z = self.initial(B)
        hs, zs, priors_m, priors_s, posts_m, posts_s = [], [], [], [], [], []
        for t in range(L):
            h, prior, post = self.obs_step(h, z, act[:, t], obs[:, t])
            z = post.rsample()
            hs.append(h); zs.append(z)
            priors_m.append(prior.mean); priors_s.append(prior.stddev)
            posts_m.append(post.mean); posts_s.append(post.stddev)
        st = lambda xs: torch.stack(xs, 1)
        return st(hs), st(zs), Normal(st(priors_m), st(priors_s)), Normal(st(posts_m), st(posts_s))


def world_model_loss(wm, obs, act, rew, free_nats=1.0, kl_balance=0.8, kl_scale=1.0):
    h, z, prior, post = wm.observe(obs, act)
    feat = torch.cat([h, z], -1)
    recon = F.mse_loss(wm.decoder(feat), obs, reduction="none").sum(-1).mean()
    reward = F.mse_loss(wm.reward(feat).squeeze(-1), rew)
    # KL balancing (DreamerV2): train the prior toward the posterior faster than the reverse
    sg = lambda d: Normal(d.mean.detach(), d.stddev.detach())
    kl_prior = kl_divergence(sg(post), prior).sum(-1).mean()
    kl_post = kl_divergence(post, sg(prior)).sum(-1).mean()
    kl = kl_balance * torch.clamp(kl_prior, min=free_nats) + (1 - kl_balance) * torch.clamp(kl_post, min=free_nats)
    return recon + reward + kl_scale * kl, (h.detach(), z.detach()), dict(recon=recon.item(), rew=reward.item(),
                                                                           kl=kl_prior.item())


# ----------------------------------------------------------------------------------------------
# 3. Actor and critic operating on latent features [h, z]
# ----------------------------------------------------------------------------------------------
class Actor(nn.Module):
    def __init__(self, act_dim):
        super().__init__()
        self.net = mlp(H_DIM + Z_DIM, 2 * act_dim)

    def forward(self, feat, deterministic=False):
        mean, std = self.net(feat).chunk(2, -1)
        std = F.softplus(std) + 0.05
        u = mean if deterministic else mean + std * torch.randn_like(mean)   # reparameterized
        ent = Normal(mean, std).entropy().sum(-1)                            # pre-tanh entropy (bonus)
        return torch.tanh(u), ent


def lambda_returns(rewards, values, gamma=0.95, lam=0.95):
    """rewards: (H, N) for steps 1..H; values: (H+1, N) for states 0..H.
    R^lambda_t = r_{t+1} + gamma * ((1 - lam) * V(s_{t+1}) + lam * R^lambda_{t+1}),  R^lambda_H = V(s_H)."""
    H = rewards.shape[0]
    ret, out = values[-1], []
    for t in reversed(range(H)):
        ret = rewards[t] + gamma * ((1 - lam) * values[t + 1] + lam * ret)
        out.append(ret)
    return torch.stack(out[::-1])           # (H, N): returns for states 0..H-1


def imagine_and_learn(wm, actor, critic, critic_t, opt_actor, opt_critic, start, horizon=15, ent_coef=1e-3):
    h, z = [x.reshape(-1, x.shape[-1]) for x in start]       # every posterior state is a dream start
    feats, rews = [torch.cat([h, z], -1)], []
    for _ in range(horizon):
        a, _ = actor(feats[-1])
        h, prior = wm.img_step(h, z, a)
        z = prior.rsample()                                  # dream with the PRIOR (no observations)
        f = torch.cat([h, z], -1)
        feats.append(f)
        rews.append(wm.reward(f).squeeze(-1))
    feats, rews = torch.stack(feats), torch.stack(rews)      # (H+1, N, F), (H, N)
    values = critic_t(feats).squeeze(-1)
    returns = lambda_returns(rews, values)
    _, ent = actor(feats[:-1].detach())
    # actor: maximize lambda-returns by backpropagating THROUGH the learned dynamics and reward model
    actor_loss = -returns.mean() - ent_coef * ent.mean()
    opt_actor.zero_grad()
    actor_loss.backward()
    nn.utils.clip_grad_norm_(actor.parameters(), 100.0)
    opt_actor.step()
    # critic: regress V(s_t) on the (detached) lambda-returns
    critic_loss = 0.5 * (critic(feats[:-1].detach()).squeeze(-1) - returns.detach()).pow(2).mean()
    opt_critic.zero_grad()
    critic_loss.backward()
    opt_critic.step()
    return actor_loss.item(), critic_loss.item(), rews.mean().item()


# ----------------------------------------------------------------------------------------------
# 4. Acting in the real env: filter the latent state online with the posterior
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def run_episode(env, wm, actor, mode="explore"):
    obs, ep = env.reset(), {"obs": [], "act": [], "rew": []}
    h, z = wm.initial(1)
    a_prev = torch.zeros(1, env.act_dim)
    ret = 0.0
    while True:
        h, _, post = wm.obs_step(h, z, a_prev, torch.as_tensor(obs)[None])
        z = post.mean if mode == "eval" else post.sample()
        if mode == "random":
            a = torch.as_tensor(np.random.uniform(-1, 1, size=(1, env.act_dim)), dtype=torch.float32)
        else:
            a, _ = actor(torch.cat([h, z], -1), deterministic=(mode == "eval"))
        # store (o_t, a_{t-1}, r_t) aligned so that act[t] is the action that LED to obs[t]
        ep["obs"].append(obs); ep["act"].append(a_prev[0].numpy()); ep["rew"].append(0.0 if not ep["rew"] else r)
        obs, r, term, trunc = env.step(a[0].numpy())
        ret += r
        a_prev = a
        if term or trunc:
            break
    ep = {k: np.array(v, np.float32) for k, v in ep.items()}
    return ep, ret


def sample_batch(episodes, B=32, L=20):
    o, a, r = [], [], []
    for _ in range(B):
        ep = episodes[np.random.randint(len(episodes))]
        t = np.random.randint(0, len(ep["obs"]) - L + 1)
        o.append(ep["obs"][t:t + L]); a.append(ep["act"][t:t + L]); r.append(ep["rew"][t:t + L])
    return [torch.as_tensor(np.array(x)) for x in (o, a, r)]


def main(seed_episodes=5, train_episodes=25, updates_per_episode=40):
    env = PointMass(0)
    wm, actor = RSSM(env.obs_dim, env.act_dim), Actor(env.act_dim)
    critic = mlp(H_DIM + Z_DIM, 1)
    critic_t = copy.deepcopy(critic)
    opt_wm = torch.optim.Adam(wm.parameters(), lr=1e-3)
    opt_actor = torch.optim.Adam(actor.parameters(), lr=3e-4)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=3e-4)

    episodes = [run_episode(env, wm, actor, mode="random")[0] for _ in range(seed_episodes)]
    rand_ret = np.mean([run_episode(env, wm, actor, mode="random")[1] for _ in range(10)])
    pd_env = PointMass(123)                    # hand-designed PD controller as a reference ceiling
    pd_rets = []
    for _ in range(10):
        o, ret = pd_env.reset(), 0.0
        for _ in range(pd_env.horizon):
            o, r, _, _ = pd_env.step(np.clip(-2.0 * o[:2] - 3.0 * o[2:], -1, 1))
            ret += r
        pd_rets.append(ret)
    print(f"random policy return {rand_ret:7.1f} | hand-tuned PD controller {np.mean(pd_rets):7.1f}")

    t0, it = time.time(), 0
    for ep_i in range(1, train_episodes + 1):
        for _ in range(updates_per_episode):
            obs, act, rew = sample_batch(episodes)
            loss, start, info = world_model_loss(wm, obs, act, rew)
            opt_wm.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(wm.parameters(), 100.0)
            opt_wm.step()
            _, _, dream_r = imagine_and_learn(wm, actor, critic, critic_t, opt_actor, opt_critic, start)
            it += 1
            if it % 20 == 0:   # hard-ish target critic update (Dreamer V2 uses periodic copies)
                critic_t.load_state_dict(critic.state_dict())
        ep, ret = run_episode(env, wm, actor, mode="explore")
        episodes.append(ep)
        if ep_i % 5 == 0:
            print(f"episode {ep_i:3d} | real return {ret:7.1f} | recon {info['recon']:.4f} rew-err {info['rew']:.4f} "
                  f"KL {info['kl']:.2f} | mean dreamed reward {dream_r:6.3f} | {time.time() - t0:5.1f}s")
    eval_ret = np.mean([run_episode(env, wm, actor, mode="eval")[1] for _ in range(10)])
    print(f"Dreamer-style agent (eval) return {eval_ret:7.1f}   vs random {rand_ret:.1f}, PD {np.mean(pd_rets):.1f}")

    # open-loop imagination check: observe 5 real steps, then dream 15 steps and compare predicted rewards
    ep, _ = run_episode(env, wm, actor, mode="eval")
    with torch.no_grad():
        o, a = torch.as_tensor(ep["obs"][None, :5]), torch.as_tensor(ep["act"][None, :5])
        h, z, _, post = wm.observe(o, a)
        h, z = h[:, -1], post.mean[:, -1]
        pred = []
        for t in range(5, 20):
            h, prior = wm.img_step(h, z, torch.as_tensor(ep["act"][None, t]))
            z = prior.mean
            pred.append(wm.reward(torch.cat([h, z], -1)).item())
    print("open-loop dream vs reality (rewards, steps 5..19):")
    print("  dreamed: " + " ".join(f"{x:5.2f}" for x in pred))
    print("  real   : " + " ".join(f"{x:5.2f}" for x in ep["rew"][5:20]))


if __name__ == "__main__":
    main()
