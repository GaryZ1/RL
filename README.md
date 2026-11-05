# RL Knowledge Base: Every Major Reinforcement Learning Method, Explained Three Ways

From Sutton & Barto's tabular methods to the RL that trains today's LLMs (RLHF, DPO, GRPO, RLVR, DAPO, GSPO, VAPO,
on-policy distillation, agentic RL). Every method is explained **for engineers** (math, pseudocode, Python),
**for business users** (value, risks, questions to ask), and **for kids** (stories and activities).

## Start here

| Document | What's inside |
|---|---|
| [**RL_Methods_Overview.md**](RL_Methods_Overview.md) | Every method: what it is, family, pros/cons, master comparison table, **mind map**, LLM pipeline, decision guide, timeline |
| [**RL_Future_Directions.md**](RL_Future_Directions.md) | Where RL is heading: scaling, rewards beyond verifiers, dense signals, agentic RL, safety, world models, predictions |
| [`methods/`](methods) | 49 deep-dive files, one per method or method family |
| [`code/`](code) | Runnable Python for every deep-dive (NumPy / PyTorch, CPU-friendly, no gym needed) |

## Method deep dives

| Family | Files |
|---|---|
| 01 Foundations & tabular | [Bandits](methods/01-foundations-tabular/multi-armed-bandits.md) · [Dynamic programming](methods/01-foundations-tabular/dynamic-programming.md) · [Monte Carlo](methods/01-foundations-tabular/monte-carlo.md) · [TD learning](methods/01-foundations-tabular/td-learning.md) · [SARSA](methods/01-foundations-tabular/sarsa.md) · [Q-learning](methods/01-foundations-tabular/q-learning.md) · [Dyna-Q](methods/01-foundations-tabular/dyna-q.md) |
| 02 Deep value-based | [DQN](methods/02-deep-value-based/dqn.md) · [Rainbow](methods/02-deep-value-based/rainbow-dqn.md) · [Distributional RL](methods/02-deep-value-based/distributional-rl.md) |
| 03 Policy gradient | [REINFORCE](methods/03-policy-gradient/reinforce.md) · [Actor-critic / A2C / A3C / GAE](methods/03-policy-gradient/actor-critic-a2c-a3c.md) · [TRPO](methods/03-policy-gradient/trpo.md) · [PPO](methods/03-policy-gradient/ppo.md) |
| 04 Off-policy actor-critic | [DDPG](methods/04-off-policy-actor-critic/ddpg.md) · [TD3](methods/04-off-policy-actor-critic/td3.md) · [SAC](methods/04-off-policy-actor-critic/sac.md) |
| 05 Model-based | [MCTS / AlphaZero](methods/05-model-based/mcts-alphazero.md) · [MuZero](methods/05-model-based/muzero.md) · [World models / Dreamer](methods/05-model-based/world-models-dreamer.md) · [PETS / MBPO](methods/05-model-based/mbpo-pets.md) |
| 06 Offline RL | [CQL](methods/06-offline-rl/cql.md) · [IQL](methods/06-offline-rl/iql.md) · [Decision Transformer](methods/06-offline-rl/decision-transformer.md) |
| 07 Imitation & inverse RL | [Behavior cloning / DAgger](methods/07-imitation-inverse-rl/behavior-cloning-dagger.md) · [Inverse RL](methods/07-imitation-inverse-rl/inverse-rl.md) · [GAIL](methods/07-imitation-inverse-rl/gail.md) |
| 08 Exploration, hierarchy, multi-agent | [Curiosity / RND](methods/08-exploration-hierarchical-multiagent/curiosity-icm-rnd.md) · [Hierarchical RL](methods/08-exploration-hierarchical-multiagent/hierarchical-rl.md) · [Multi-agent RL](methods/08-exploration-hierarchical-multiagent/multi-agent-rl.md) · [Evolution strategies](methods/08-exploration-hierarchical-multiagent/evolution-strategies.md) |
| 09 LLM: preference & alignment | [RLHF-PPO](methods/09-llm-preference-alignment/rlhf-ppo.md) · [Reward modeling](methods/09-llm-preference-alignment/reward-modeling.md) · [RLAIF / Constitutional AI](methods/09-llm-preference-alignment/rlaif-constitutional-ai.md) · [DPO](methods/09-llm-preference-alignment/dpo.md) · [DPO variants](methods/09-llm-preference-alignment/dpo-variants.md) · [Online / self-play PO](methods/09-llm-preference-alignment/online-iterative-self-play-po.md) · [Rubric & generative rewards](methods/09-llm-preference-alignment/rubric-generative-rewards.md) · [Safety RL & reward hacking](methods/09-llm-preference-alignment/safety-rl-reward-hacking.md) |
| 10 LLM: reasoning & agents | [RLVR / DeepSeek-R1](methods/10-llm-reasoning-rl/rlvr.md) · [GRPO](methods/10-llm-reasoning-rl/grpo.md) · [RLOO / ReMax / REINFORCE++](methods/10-llm-reasoning-rl/rloo-remax-reinforcepp.md) · [DAPO / Dr. GRPO](methods/10-llm-reasoning-rl/dapo-dr-grpo.md) · [GSPO / SAPO / GMPO / CISPO](methods/10-llm-reasoning-rl/gspo-sapo-gmpo-cispo.md) · [VAPO](methods/10-llm-reasoning-rl/value-based-llm-rl-vapo.md) · [PRM / PRIME](methods/10-llm-reasoning-rl/process-reward-prime.md) · [On-policy distillation](methods/10-llm-reasoning-rl/on-policy-distillation.md) · [Agentic multi-turn RL](methods/10-llm-reasoning-rl/agentic-multi-turn-rl.md) · [Self-rewarding / TTRL](methods/10-llm-reasoning-rl/self-rewarding-ttrl.md) |

## Every deep-dive file has the same 13 sections

1. What is it? · 2. Family & lineage · 3. Core idea & intuition · 4. Math · 5. Pseudocode · 6. **Python implementation** ·
7. Practical tips · 8. **Pros and cons** · 9. When to use · 10. Papers & repos · 11. **For business users** ·
12. **For kids** · 13. Related methods

## Running the code

```bash
pip install numpy torch
```

```bash
python3 code/q-learning.py
```

All scripts are self-contained (tiny built-in environments, no gym), run on CPU, and finish in about two minutes or
less. LLM-RL scripts train a tiny toy "language model" on a verifiable task so you can watch the loss functions
(GRPO, DAPO, GSPO, DPO, …) work end-to-end; each file also shows how to run the same method with a real library
(TRL, verl, OpenRLHF), which needs a GPU.

## Main sources

- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed., 2018)
- Papers and reports from OpenAI, Anthropic, DeepSeek, Google DeepMind, Qwen, ByteDance Seed, MiniMax, Moonshot AI,
  Meta, AI2, Thinking Machines Lab, and university labs (full citations in each file)
- Repositories: `openai/spinningup`, `anthropics/hh-rlhf`, `deepseek-ai/DeepSeek-R1`, `huggingface/trl`,
  `volcengine/verl`, `OpenRLHF/OpenRLHF`, `vwxyzjn/cleanrl`, `DLR-RM/stable-baselines3`
