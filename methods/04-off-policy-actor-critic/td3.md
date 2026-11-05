# Twin Delayed Deep Deterministic Policy Gradient (TD3)
> DDPG plus three surgical fixes against critic overestimation: clipped double-Q targets, delayed actor/target updates, and target policy smoothing. **Family:** Model-free → Actor-critic → Off-policy deterministic actor-critic  ·  **Year / origin:** Fujimoto, van Hoof & Meger, ICML 2018 (McGill University)  ·  **On/Off-policy:** Off-policy  ·  **Action space:** Continuous

## 1. What is it?

TD3 is the "make DDPG actually work" algorithm. Fujimoto et al. showed that the overestimation bias well known from Q-learning (and fixed in the discrete case by Double Q-learning / Double DQN) also afflicts actor-critic methods: the actor is trained to maximize the critic, so it actively seeks out regions where the critic's errors are positive, and TD bootstrapping propagates those errors through time. The result is inflated value estimates, a policy that chases phantoms, and unstable learning.

TD3 keeps DDPG's structure — deterministic actor, Q critic, replay buffer, Polyak-averaged target networks, Gaussian exploration noise — and adds three changes:

1. **Clipped double-Q learning** — learn two critics $Q_{\phi_1}, Q_{\phi_2}$ and use the *minimum* of their target values in the Bellman target. This biases the target toward underestimation, which is far less harmful than overestimation because the actor does not seek out underestimated actions.
2. **Delayed policy updates** — update the actor (and the target networks) only every $d$ critic updates (typically $d=2$), so the actor follows a critic that has had time to reduce its error.
3. **Target policy smoothing** — add small clipped noise to the target action $\mu_{\theta'}(s')$ when computing the target. This is a regularizer: it says "similar actions should have similar values," removing narrow spikes in $Q$ that the actor could exploit.

Where it sits: model-free, off-policy, deterministic actor-critic for continuous control. On the MuJoCo benchmarks of its time it substantially outperformed DDPG and was competitive with SAC (published the same year). It remains one of the strongest, simplest baselines, and is the backbone of several later methods (TD3+BC for offline RL, TD7).

## 2. Family & lineage

- **Parents:** DDPG (Lillicrap et al., 2015); Double Q-learning (van Hasselt, 2010) and Double DQN (van Hasselt, Guez & Silver, 2016) for the overestimation insight.
- **Siblings:** SAC (Haarnoja et al., 2018) — independently adopted the twin-critic min trick, adds entropy; D4PG — distributional variant of DDPG.
- **Descendants:** **TD3+BC** (Fujimoto & Gu, 2021, offline RL with a behavior-cloning term); **TD7** (Fujimoto et al., 2023, adds learned state-action embeddings "SALE", prioritized replay and checkpoints); REDQ (Chen et al., 2021, ensemble of critics + random subset min) generalizes clipped double-Q; many robotics and model-based pipelines use TD3 as their inner learner.

```mermaid
graph LR
  DQ["Double Q-learning (2010)"] --> DDQN["Double DQN (2016)"]
  DDPG["DDPG (2015)"] --> TD3["TD3 (2018)"]
  DDQN --> TD3
  TD3 --> TD3BC["TD3+BC (2021)"]
  TD3 --> TD7["TD7 (2023)"]
  TD3 --> REDQ["REDQ (2021)"]
  TD3 -.twin critics.-> SAC["SAC v2 (2018)"]
```

## 3. Core idea & intuition

**Why overestimation happens.** Suppose the critic's estimates are true values plus zero-mean noise: $\hat Q(s,a) = Q(s,a) + \epsilon(s,a)$. Picking the action that maximizes $\hat Q$ preferentially picks actions where $\epsilon$ happens to be positive, so $\mathbb E[\max_a \hat Q] \ge \max_a Q$. In DDPG the actor *is* that maximizer (via gradient ascent), and the target $r + \gamma \hat Q(s', \mu(s'))$ feeds the inflated value back into the critic. Errors compound over the effective horizon $1/(1-\gamma)$.

**Why the min of two critics helps.** Two critics trained on the same targets but with different initializations make partly independent errors. Taking $\min(Q_1, Q_2)$ cancels a lot of the upward bias. It can underestimate — but an underestimated action is simply not chosen by the actor, so the error does not get amplified the way overestimation does.

Why not just use Double DQN's trick (evaluate the target actor's action with the *online* critic)? Fujimoto et al. found that in actor-critic the online and target networks change so slowly that they are too correlated to decorrelate the error; an explicit second critic works much better.

**Why delay the actor.** The actor's gradient is only as good as the critic's slope. If you update the actor every step, it chases a critic that is still noisy and moving. Updating the actor less often (and the targets with it) lets the critic's error shrink first: "update the value estimate until it is good, then improve the policy" — a soft version of policy iteration.

**Why smooth the target.** A deterministic actor trained against a neural critic can find sharp, narrow peaks in $Q$ (artifacts of function approximation). Averaging the target over a small neighborhood of actions, $\mathbb E_\epsilon[Q(s', \mu'(s') + \epsilon)]$, removes those peaks, akin to SARSA-style expected updates with a smoothed policy.

## 4. Math / formal details

Symbols: state $s$, action $a\in[a_{\text{low}}, a_{\text{high}}]^m$, reward $r$, discount $\gamma$, terminal flag $d$; actor $\mu_\theta$; two critics $Q_{\phi_1}, Q_{\phi_2}$; target networks $\theta', \phi'_1, \phi'_2$; replay buffer $\mathcal D$; Polyak rate $\tau$; target-noise std $\tilde\sigma$ and clip $c$; policy delay $d_\pi$.

**Overestimation (actor-critic version).** Let $\theta_{\text{approx}}$ be the actor updated by ascending $Q_\phi$ and $\theta_{\text{true}}$ the actor updated by ascending the true $Q^\pi$. Fujimoto et al. show that for sufficiently small step sizes, $\mathbb E[Q_\phi(s, \mu_{\theta_{\text{approx}}}(s))] \ge \mathbb E[Q^\pi(s,\mu_{\theta_{\text{approx}}}(s))]$ under mild assumptions — i.e., the updated policy's value is overestimated — and that this error accumulates through TD bootstrapping.

**Target policy smoothing.**
$$\tilde a' = \text{clip}\Big(\mu_{\theta'}(s') + \text{clip}(\epsilon, -c, c),\ a_{\text{low}}, a_{\text{high}}\Big), \qquad \epsilon\sim\mathcal N(0, \tilde\sigma^2 I).$$

**Clipped double-Q target** (one shared target for both critics):
$$y = r + \gamma (1-d)\, \min_{j=1,2} Q_{\phi'_j}(s', \tilde a').$$

**Critic losses:**
$$\mathcal L(\phi_j) = \mathbb E_{(s,a,r,s',d)\sim\mathcal D}\big[(Q_{\phi_j}(s,a) - y)^2\big], \quad j=1,2.$$

**Delayed actor update** (every $d_\pi$ critic steps), deterministic policy gradient through $Q_{\phi_1}$ only:
$$\nabla_\theta J(\theta) = \mathbb E_{s\sim\mathcal D}\big[\nabla_a Q_{\phi_1}(s,a)\big|_{a=\mu_\theta(s)}\;\nabla_\theta\mu_\theta(s)\big].$$

**Targets** (also every $d_\pi$ steps):
$$\phi'_j \leftarrow \tau\phi_j + (1-\tau)\phi'_j, \qquad \theta' \leftarrow \tau\theta + (1-\tau)\theta'.$$

**Exploration** when acting: $a = \text{clip}(\mu_\theta(s) + \mathcal N(0,\sigma^2_{\text{expl}}),\ a_{\text{low}}, a_{\text{high}})$, typically $\sigma_{\text{expl}} = 0.1\, a_{\max}$.

**Bias intuition for the min.** If $Q_1 = Q + \epsilon_1$, $Q_2 = Q + \epsilon_2$ with i.i.d. zero-mean noise of std $\sigma_\epsilon$, then $\mathbb E[\min(Q_1,Q_2)] = Q - \sigma_\epsilon/\sqrt{\pi}$ for Gaussian noise — a bounded downward bias in place of the unbounded upward drift caused by maximization feedback.

## 5. Algorithm (pseudocode)

```text
TD3
Input: actor mu_theta, critics Q_phi1, Q_phi2, replay D, tau, gamma,
       exploration std sigma, target noise std sigma~, noise clip c, policy delay d
 1. Initialize theta, phi1, phi2; targets theta' <- theta, phi1' <- phi1, phi2' <- phi2
 2. for t = 1 .. T do
 3.     a = clip(mu_theta(s) + N(0, sigma^2), a_low, a_high)   (uniform random during warm-up)
 4.     step env -> r, s', d (d = 1 only for true termination); store (s, a, r, s', d) in D
 5.     sample minibatch B from D
 6.     eps = clip(N(0, sigma~^2), -c, c)
 7.     a~' = clip(mu_theta'(s') + eps, a_low, a_high)                 # target policy smoothing
 8.     y = r + gamma * (1 - d) * min(Q_phi1'(s', a~'), Q_phi2'(s', a~'))   # clipped double-Q
 9.     update phi1, phi2 by minimizing sum_j mean_B (Q_phij(s,a) - y)^2
10.     if t mod d == 0:                                                # delayed updates
11.         update theta by gradient ascent on mean_B Q_phi1(s, mu_theta(s))
12.         phi_j' <- tau*phi_j + (1-tau)*phi_j'   (j = 1, 2)
13.         theta' <- tau*theta + (1-tau)*theta'
14. return mu_theta
```

## 6. Python implementation

Runnable version: [`code/td3.py`](../../code/td3.py)

Same NumPy Pendulum-v1 as the DDPG page so you can compare directly. After training, the script also measures critic bias: it compares $Q_1(s, \mu(s))$ along greedy rollouts with the actual discounted Monte-Carlo return. Tested output (≈55 s CPU):

```text
step   3000 | last-5 train return  -1534.3 |   6.4s
step   6000 | last-5 train return   -273.8 |  16.7s
step   9000 | last-5 train return   -126.1 |  27.3s
step  12000 | last-5 train return   -201.8 |  40.8s
step  15000 | last-5 train return   -112.1 |  52.7s
TD3 (greedy)   : avg return   -179.0   (~-150 is near-optimal)
critic bias    : mean Q1 prediction   -27.4 vs true discounted return   -28.3  (clipped double-Q keeps these close)
```

@@CODE@@

What to try:
- Set `policy_delay=1`, `policy_noise=0.0` and replace `torch.min(q1_t, q2_t)` with `q1_t` → you get (twin-headed) DDPG; rerun the bias measurement and compare.
- On Pendulum the differences are small (the task is easy); on harder MuJoCo-style tasks the TD3 paper reports large gains from clipped double-Q especially.

## 7. Practical tips & hyperparameters

| Hyperparameter | TD3 paper / author repo (typical) | Notes |
|---|---|---|
| Actor & critic LR | 1e-3 in the paper; 3e-4 in the author's later repo | Adam |
| Batch size | 100 in the paper; 256 in the later repo | |
| $\gamma$ | 0.99 | |
| $\tau$ | 0.005 | |
| Policy delay $d$ | 2 | update actor every 2 critic steps |
| Target noise $\tilde\sigma$, clip $c$ | 0.2, 0.5 (× max action) | |
| Exploration noise | $\mathcal N(0, 0.1)$ (× max action) | |
| Warm-up random steps | 1k in the paper, 25k in the later repo | |
| Network | 2 hidden layers, 256 (or 400–300), ReLU | |

Tips:
- **Both critics regress on the same target**; do not give each critic its own min target.
- **Actor uses $Q_1$ only** (cheaper; using the min or mean of both also works).
- **Scale noise to the action range.** If actions are in $[-2, 2]$ (Pendulum), noise of 0.2 means $0.2 \times 2 = 0.4$.
- **Underestimation is not free**: in very sparse-reward or optimistic-exploration settings, the pessimism of the min can slow learning. Some methods (e.g. tuned ensembles, REDQ, "optimistic" variants) trade off pessimism.
- **Debugging signal**: plot mean $Q_1(s,\mu(s))$ vs Monte-Carlo returns (as the script does). Large positive gaps = overestimation (actor exploiting critic); large negative = excessive pessimism.
- **Delayed updates also delay the targets** — that is part of what stabilizes training.
- **Offline**: TD3+BC adds $\lambda$-normalized behavior cloning to the actor loss: maximize $\lambda Q(s,\mu(s)) - (\mu(s) - a)^2$. A one-line change with strong results on D4RL.

## 8. Pros and cons

**Pros**
- Much more stable and better-performing than DDPG with only a few lines of change.
- Simple, deterministic, cheap at inference; easy to reason about.
- Strong baseline on continuous control; few hyperparameters; robust defaults.
- Excellent foundation for offline RL (TD3+BC) and representation-learning extensions (TD7).

**Cons**
- Still needs hand-tuned exploration noise; deterministic policy can be poor at exploration.
- Pessimistic bias can under-explore or slow learning in some tasks.
- Two critics ≈ 2× critic compute.
- Continuous actions only.
- SAC is often slightly more robust across tasks thanks to entropy-driven exploration.

## 9. When to use / when not to use

**Use when**
- You need a reliable, simple off-policy baseline for continuous control (robotics sims, locomotion, manipulation).
- You want a deterministic policy (e.g. for deployment or as a base for offline RL with TD3+BC).
- You care about sample efficiency more than wall-clock parallelism.

**Avoid when**
- Exploration is the bottleneck (sparse rewards) → SAC (entropy), intrinsic motivation, or model-based methods.
- Actions are discrete → DQN family, PPO.
- You have massively parallel cheap simulators → PPO can be simpler and faster in wall-clock.
- Your problem is a fixed dataset with no interaction → use offline variants (TD3+BC, IQL, CQL), not plain TD3.

## 10. Key papers, books & repos

- S. Fujimoto, H. van Hoof, D. Meger. *Addressing Function Approximation Error in Actor-Critic Methods.* ICML 2018, arXiv:1802.09477.
- H. van Hasselt. *Double Q-learning.* NeurIPS 2010.
- H. van Hasselt, A. Guez, D. Silver. *Deep Reinforcement Learning with Double Q-learning.* AAAI 2016.
- S. Thrun, A. Schwartz. *Issues in Using Function Approximation for Reinforcement Learning.* 1993 (early analysis of overestimation).
- T. P. Lillicrap et al. *Continuous control with deep reinforcement learning (DDPG).* arXiv:1509.02971.
- S. Fujimoto, S. S. Gu. *A Minimalist Approach to Offline Reinforcement Learning (TD3+BC).* NeurIPS 2021.
- S. Fujimoto et al. *For SALE: State-Action Representation Learning for Deep Reinforcement Learning (TD7).* NeurIPS 2023.
- X. Chen, C. Wang, Z. Zhou, K. Ross. *Randomized Ensembled Double Q-Learning: Learning Fast Without a Model (REDQ).* ICLR 2021.
- R. S. Sutton & A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed., 2018 — §6.7 (maximization bias and Double Learning), Ch. 13 (policy gradient).
- Repos: [sfujim/TD3](https://github.com/sfujim/TD3) (author's reference), [openai/spinningup](https://github.com/openai/spinningup) (TD3 docs + code), [DLR-RM/stable-baselines3](https://github.com/DLR-RM/stable-baselines3) (`TD3`), [vwxyzjn/cleanrl](https://github.com/vwxyzjn/cleanrl) (`td3_continuous_action.py`).

## 11. For business users

**What it does.** TD3 learns a controller for continuous "dials" (motor torques, flow rates, set-points) from experience, like DDPG, but with built-in skepticism: it keeps two independent "judges" of how good an action is and always believes the more pessimistic one. It also lets the judges settle before the "driver" changes its behavior.

**Business analogy.** A purchasing manager gets two independent quotes for how much a project will return and plans on the lower one; and she waits for the analysts to finish revising their numbers before changing strategy. That habit avoids chasing overly rosy forecasts.

**Real-world use cases.** Robot locomotion and manipulation in simulation (then transferred to hardware), industrial control research, energy-system control, and as the core learner inside offline-RL systems that learn from historical logs (TD3+BC).

**Value / ROI.** More reliable than DDPG at essentially the same cost; fewer failed training runs means less engineering time and compute wasted. Produces a fast, deterministic controller that is easy to deploy and test.

**Risks.** Still requires a good simulator or safe exploration; can be overly cautious in some tasks; like all RL, it optimizes exactly the reward you define — mis-specified rewards produce mis-specified behavior.

**Costs.** Small-to-moderate compute (often a single GPU); main costs are simulator fidelity, reward design, and validation.

**Questions to ask your team.**
- Are we comparing against SAC and a simple non-RL controller (e.g. PID / MPC)?
- How do learned value estimates compare with actual measured returns?
- How stable are results over 5+ random seeds?
- What is the plan for safe deployment and monitoring on the real system?

## 12. For kids (ages ~8–12)

**Two weather forecasters.** You want to know whether to bring a jacket to the park. You ask two friends who each guess the weather. One says "super sunny, 30 degrees!" and the other says "kind of cool, 18 degrees." A careful kid plans for the cooler guess and brings a light jacket — you won't freeze if the happy guess was wrong. TD3 does the same: it asks two "guessers" how good a move is and trusts the less excited one. It also doesn't change its plan every second — it waits until the guessers have thought a bit more. And it checks that moves *near* a good move are also good, so it doesn't fall for a lucky fluke.

**Try it yourself.** Ask two family members to guess how many candies are in a jar. Write down the smaller guess. Count the candies. Was the smaller guess closer more often than the bigger one? Try it with several jars!

**Recap.**
1. Ask two guessers, believe the more careful one.
2. Don't change your plan too often — let the guessers learn first.
3. Good moves should have good neighbors, not just one lucky spot.

## 13. Related methods

- [DDPG — the parent algorithm](./ddpg.md)
- [SAC — stochastic maximum-entropy cousin that also uses twin critics](./sac.md)
- [DQN / Double DQN — origins of overestimation fixes](../02-deep-value-based/dqn.md)
- [Rainbow DQN — combining many DQN fixes](../02-deep-value-based/rainbow-dqn.md)
- [Q-learning — maximization bias in the tabular case](../01-foundations-tabular/q-learning.md)
- [CQL — pessimism taken further, for offline RL](../06-offline-rl/cql.md)
- [IQL — offline RL alternative to TD3+BC](../06-offline-rl/iql.md)
- [MBPO / PETS — model-based RL that pairs a learned model with an off-policy learner](../05-model-based/mbpo-pets.md)

[← Back to overview](../../RL_Methods_Overview.md)
