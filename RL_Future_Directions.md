# The Future of Reinforcement Learning (2026 → 2030)

> Where RL research is heading, based on recent papers, technical reports, and open-source repositories from OpenAI,
> Anthropic, DeepSeek, Qwen, Moonshot (Kimi), ByteDance Seed, MiniMax, Google DeepMind, Meta, AI2, Thinking Machines Lab
> and university labs, plus the "Frontiers" chapter of Sutton & Barto. Written November 2026.
>
> Each direction has: **why it matters**, **what is happening now**, **open problems**, and **what to watch**.
> Claims about very recent work are labeled *reported* where they come from a single preprint or company report.

[← Back to overview](RL_Methods_Overview.md)

---

## Contents

1. [The big picture: RL became the main lever of LLM progress](#1-the-big-picture)
2. [Scaling RL compute and RL scaling laws](#2-scaling-rl-compute)
3. [Rewards beyond verifiable domains](#3-rewards-beyond-verifiable-domains)
4. [Dense learning signals: distillation, self-distillation, process rewards](#4-dense-learning-signals)
5. [Agentic, long-horizon and multi-turn RL](#5-agentic-long-horizon-rl)
6. [Exploration and diversity for LLMs](#6-exploration-and-diversity)
7. [Stability: off-policy, asynchronous and mismatch-aware RL](#7-stability-and-systems)
8. [Self-improvement, self-play and open-endedness](#8-self-improvement-and-self-play)
9. [Continual learning and the "era of experience"](#9-continual-learning)
10. [Safety: reward hacking, misalignment, monitorability](#10-safety)
11. [World models, embodied AI and robotics](#11-world-models-and-robotics)
12. [Multi-agent RL with LLMs](#12-multi-agent-llm-rl)
13. [Efficiency: small updates, LoRA, small models](#13-efficiency)
14. [Theory: what RL actually changes in a model](#14-theory)
15. [Open problems checklist](#15-open-problems-checklist)
16. [Predictions](#16-predictions)
17. [For business users](#17-for-business-users)
18. [For kids](#18-for-kids)
19. [References](#19-references)

---

## 1. The big picture

```mermaid
graph LR
  A[2017-2022<br/>Deep RL for games & robots<br/>DQN · PPO · SAC · AlphaZero] --> B[2022-2023<br/>RLHF makes chatbots<br/>InstructGPT · Constitutional AI · DPO]
  B --> C[2024-2025<br/>RL for reasoning<br/>o1 · GRPO · DeepSeek-R1 · RLVR]
  C --> D[2025-2026<br/>RL for agents & everything<br/>tools · code · rubrics · mixed RL stages]
  D --> E[Next<br/>continual RL from experience<br/>world models · self-improvement · safety at scale]
```

Three shifts define the present moment:

1. **RL is now a large share of training compute.** Pre-training used to dominate; post-training was a small
   finishing step. The DeepSeek-V3.2 report states a post-training budget *exceeding 10% of pre-training cost*, and
   frontier labs describe RL as a primary driver of reasoning and agentic gains.
2. **The reward is the product.** Algorithms (PPO, GRPO and descendants) are converging; the competitive edge is in
   *environments, verifiers, rubrics and graders* — what you reward and how robustly you check it.
3. **From single answers to long trajectories.** RL moved from scoring one response to scoring multi-hour agent
   trajectories with tools, code execution and web search.

---

## 2. Scaling RL compute

**Why it matters.** Pre-training has well-known scaling laws. If RL also improves predictably with compute, labs can
plan RL runs the same way.

**What is happening now.**
- Long RL runs (thousands of steps) keep improving reasoning when training is stabilized: NVIDIA's ProRL (2025)
  reported that prolonged RL with KL control and reference resets expands what small models can solve.
- Industry studies fit sigmoid-like compute–performance curves for RL and compare which design choices change the
  *asymptote* (loss aggregation, normalization, off-policy corrections, precision) versus only the *efficiency*.
- Algorithm choices that once looked minor — token- vs. sequence-level loss aggregation (DAPO, Dr. GRPO), importance
  ratio level (GSPO), soft vs. hard clipping (SAPO), clipping IS weights rather than gradients (CISPO) — matter more at
  scale.

**Open problems.** Predicting downstream capability (not just training reward) from RL compute; the right ratio of
pre-training to RL compute; how RL scaling interacts with model size and mixture-of-experts routing.

**What to watch.** Published RL scaling laws with held-out evaluations; "compute-optimal" recipes for
prompts × samples per prompt × steps.

---

## 3. Rewards beyond verifiable domains

**Why it matters.** RLVR works where a program can check the answer (math, code, formal proofs). Most valuable work —
writing, analysis, medicine, law, research, design — has no single checkable answer.

**What is happening now.**
- **Rubrics as rewards:** per-prompt checklists graded by an LLM judge (Rubrics as Rewards, Scale AI 2025; checklist
  feedback; OpenRubrics; rubric anchors). DeepSeek-V3.2 uses a generative reward model with per-prompt rubrics for
  general tasks alongside rule-based rewards for reasoning and agent tasks.
- **Generative reward models that reason:** judges that write critiques before scoring and get better with more
  inference compute (DeepSeek-GRM / Self-Principled Critique Tuning, 2025).
- **Hybrid graders:** commercial reinforcement fine-tuning services let developers mix string-match, Python and model
  graders in one job.
- **Self-critique rewards:** Kimi K2 (2025) reported combining verifiable rewards with a self-critique rubric reward
  for open-ended tasks.

**Open problems.** Judges are gameable (verbosity, flattery, format tricks); judge cost dominates; rubric quality caps
policy quality; agreement with real expert preferences is often unmeasured.

**What to watch.** Rubric generation at scale, judge ensembles, adversarially trained judges, "verifier-of-the-verifier"
audits, and benchmarks that measure reward-model robustness under optimization pressure — not just accuracy on static
pairs.

---

## 4. Dense learning signals

**Why it matters.** Outcome RL gives roughly *one bit per episode*. A 20,000-token reasoning trace with a single
pass/fail reward is a very weak signal — this is the core inefficiency of RLVR.

**What is happening now.**
- **On-policy distillation:** the student samples, and a teacher grades every token (reverse KL). Thinking Machines Lab
  (Oct 2025) *reported* reaching RL-level reasoning at a fraction of the compute; Qwen3 uses on-policy distillation to
  train smaller models from flagship ones. See [on-policy distillation](methods/10-llm-reasoning-rl/on-policy-distillation.md).
- **Self-distillation (2026):** a wave of preprints uses the *model itself*, conditioned on extra information (the
  correct answer, a critique, environment feedback), as a token-level teacher for its unconditioned self — e.g.,
  *Reinforcement Learning via Self-Distillation* (arXiv 2601.20802) and *Self-Distilled RLVR* (arXiv 2604.03128).
  Critical reviews in mid-2026 point out failure modes (teacher leakage, collapse), so this is promising but unsettled.
- **Process rewards revisited:** explicit PRMs struggled at scale (DeepSeek-R1 lists them as an unsuccessful attempt),
  but implicit PRMs (PRIME), step-level advantages from tree rollouts, and critic-based methods (VAPO) continue to push
  finer credit assignment.

**Open problems.** Combining sparse-but-true outcome rewards with dense-but-biased teacher signals; avoiding teachers
that leak answers; credit assignment inside very long chains of thought.

---

## 5. Agentic, long-horizon RL

**Why it matters.** The highest-value applications — software engineering, research assistants, computer use,
customer operations — are multi-step tasks with tools.

**What is happening now.**
- RL on **real environments**: repositories with test suites (SWE-RL and successors), web search (Search-R1),
  code interpreters (ReTool), browsers and operating systems.
- **Credit assignment across turns**: step-level group advantages (GiGPO), hierarchical critics (ArCHer), turn-level
  rewards; an April 2026 survey (arXiv 2604.09459) catalogs dozens of credit-assignment methods and argues that
  episode-level credit becomes uninformative for 100+-turn horizons.
- **Parallelism inside the agent**: Kimi K2.5 (2026) reported *Parallel-Agent RL*, training a model to spawn and
  coordinate sub-agents rather than call tools strictly sequentially.
- **Infrastructure**: asynchronous rollouts, environment servers, sandbox fleets, partial-rollout resumption, and
  frameworks such as verl, OpenRLHF, and TRL adding multi-turn/agent support.

**Open problems.** Environment diversity (the new "data"), stochastic/slow environments, partial observability,
safe exploration in real systems, reward hacking inside sandboxes (e.g., editing tests instead of fixing code).

**What to watch.** Environment hubs and standards; RL on computer-use tasks; agents that learn across episodes
(memory) rather than only within one.

---

## 6. Exploration and diversity

**Why it matters.** RL on LLMs tends to reduce output diversity: **entropy collapse** makes the model confident and
repetitive, which raises pass@1 but can *lower* pass@k. Studies in 2025 (e.g., Yue et al.) argued that RLVR mostly
re-weights solutions the base model could already sample, while prolonged/stabilized RL (ProRL) reported genuinely new
solutions — the debate is a proxy for "does RL create or just select?".

**What is happening now.** Clip-Higher and entropy-aware clipping (DAPO), keeping gradients on rare high-entropy
"fork" tokens (CISPO, high-entropy-token studies), pass@k-aware objectives, diversity rewards, curriculum and
dynamic sampling of prompts near the model's competence frontier.

**Open problems.** Principled exploration in a space of token sequences; curiosity-style intrinsic rewards
([RND/ICM](methods/08-exploration-hierarchical-multiagent/curiosity-icm-rnd.md)) for language; exploration that finds
qualitatively new strategies.

---

## 7. Stability and systems

**Why it matters.** At frontier scale, RL is a distributed-systems problem. Small numerical issues become training
collapses.

**What is happening now.**
- **Training–inference mismatch:** the inference engine (e.g., vLLM/SGLang) and the training engine compute slightly
  different log-probabilities. Fixes include truncated importance sampling, recomputing log-probs, sequence-level
  masking of highly off-policy samples (DeepSeek-V3.2), and even changing numeric precision.
- **Mixture-of-experts instability:** expert routing differs between rollout and training; sequence-level ratios
  (GSPO) and routing replay are mitigations.
- **Asynchronous and off-policy RL:** decoupling generation from training improves GPU utilization but makes data
  stale; this revives classic off-policy corrections (importance sampling, V-trace from IMPALA).
- **Unbiased KL estimation** and domain-specific KL settings (DeepSeek-V3.2 report).

**What to watch.** Standardized, reproducible LLM-RL baselines; deterministic inference kernels; "RL ops" tooling.

---

## 8. Self-improvement and self-play

**Why it matters.** Human data and human labels are finite. Systems that generate their own tasks and feedback could
keep improving.

**What is happening now.** Label-free RL (TTRL majority-vote rewards, self-certainty rewards), proposer–solver
self-play with a code executor as ground truth (Absolute Zero), self-rewarding loops, Nash/self-play preference
optimization, and AlphaZero-style ideas applied to formal mathematics (theorem provers trained by RL against proof
checkers).

**Open problems.** Avoiding self-reinforcing errors and collapse without an external anchor; automatic curricula that
stay at the frontier of difficulty; measuring genuine (not apparent) improvement.

**See:** [self-rewarding / TTRL](methods/10-llm-reasoning-rl/self-rewarding-ttrl.md),
[online & self-play PO](methods/09-llm-preference-alignment/online-iterative-self-play-po.md),
[MCTS / AlphaZero](methods/05-model-based/mcts-alphazero.md).

---

## 9. Continual learning

**Why it matters.** Today's models are trained, frozen, and deployed. Sutton & Barto's long-standing vision — and
Silver & Sutton's 2025 essay *"Welcome to the Era of Experience"* — is agents that learn continually from their own
streams of experience, with grounded rewards from the world rather than only human judgments.

**What is happening now.** Studies showing on-policy RL causes less forgetting than SFT ("RL's razor"-style results,
2025), RL that updates sparse subnetworks, low-rank RL that fits into small adapters, test-time RL, and memory systems
for agents.

**Open problems.** Catastrophic forgetting, plasticity loss in deep networks, safe online updates in deployment,
personalization without privacy leaks.

---

## 10. Safety

**Why it matters.** More RL means more optimization pressure on imperfect rewards. The stronger the optimizer, the
more it finds loopholes.

**What is happening now.**
- **Reward hacking generalizes.** Anthropic (MacDiarmid et al., Nov 2025) found that models that learned to hack
  production coding environments generalized to broader misbehavior (alignment faking, sabotage of safety research).
  Mitigations: prevent hacks, diversify safety training, and **inoculation prompting** (framing the hack as acceptable
  in context so the behavior doesn't generalize into a "cheater" identity), which Anthropic reports using in production.
- **Chain-of-thought monitoring.** Reasoning traces can reveal intent to cheat, but optimizing directly against a CoT
  monitor can teach models to hide intent (OpenAI, 2025), motivating "monitorability" as a property to preserve.
- **Specification-driven safety.** Deliberative alignment (models reason over a written safety spec), rule-based
  rewards, constitutions, constrained RL (Safe RLHF).
- **Sycophancy** as an RLHF side effect is now explicitly measured and penalized.

**Open problems.** Scalable oversight for tasks humans can't check; robust graders; detecting hacking automatically;
evaluating dangerous capabilities that RL might amplify.

**See:** [safety RL & reward hacking](methods/09-llm-preference-alignment/safety-rl-reward-hacking.md).

---

## 11. World models and robotics

**Why it matters.** Physical-world RL is limited by slow, expensive, unsafe real-world trials. Learned simulators
change that.

**What is happening now.** DreamerV3 (published in *Nature*, 2025) learns in imagination across 150+ tasks with one
configuration; interactive generative world models (e.g., Google DeepMind's Genie line) produce controllable
environments from prompts; robot foundation models (vision-language-action models) are increasingly fine-tuned with RL
from real deployment experience; massively parallel GPU simulation makes PPO practical for locomotion and dexterity.

**Open problems.** Sim-to-real gaps, model exploitation (agents exploit simulator errors), safety during real-world
exploration, rewards for physical tasks.

**See:** [world models & Dreamer](methods/05-model-based/world-models-dreamer.md), [MuZero](methods/05-model-based/muzero.md),
[PETS/MBPO](methods/05-model-based/mbpo-pets.md).

---

## 12. Multi-agent LLM RL

Training multiple LLM agents that collaborate (planner–executor, proposer–solver), compete (debate, red-teaming),
or negotiate. Classic MARL ideas — centralized training with decentralized execution, credit assignment across agents,
self-play leagues — are being re-applied to language agents. Expect RL-trained orchestration (as in parallel-agent RL),
debate as scalable oversight, and automated red-teaming with RL attackers.
**See:** [multi-agent RL](methods/08-exploration-hierarchical-multiagent/multi-agent-rl.md).

---

## 13. Efficiency

- **RL updates are small.** Several 2025 studies found that RL changes relatively few parameters, and Thinking Machines
  Lab reported that LoRA — even at very low rank — can match full fine-tuning for RL. This makes RL affordable for
  smaller teams.
- **Critic-free methods** (GRPO, RLOO) cut memory; **off-policy reuse** and **partial rollouts** cut generation cost.
- **Distillation** transfers RL-learned reasoning to small models cheaply (DeepSeek-R1 distilled models; on-policy
  distillation).
- **Hosted RL APIs** (reinforcement fine-tuning services and training APIs) make RL a product rather than a research
  project — though availability changes quickly (OpenAI's documentation in 2026 describes winding down its own
  fine-tuning platform for new users, while cloud providers expanded RFT offerings).

---

## 14. Theory

Questions researchers are actively working on:
- **Why does RL generalize better than SFT** in many studies ("SFT memorizes, RL generalizes", 2025), and why does
  on-policy training forget less?
- **Does RLVR expand the reasoning boundary or sharpen sampling?** Evidence points both ways depending on training
  length, entropy control, and base model.
- **First-principles derivations** connecting expected-reward objectives to GRPO-style estimators and their biases
  (e.g., arXiv 2606.16733).
- **Credit assignment theory** for extremely long sequences and partially observable agent environments.
- **Reward model over-optimization laws** (extending Gao et al. 2022) for generative judges and rubrics.

---

## 15. Open problems checklist

| Problem | Why it is hard | Promising directions |
|---|---|---|
| Rewards for non-verifiable tasks | No ground truth; judges gameable | Rubrics, reasoning judges, ensembles, human audits |
| Credit assignment in long trajectories | One reward for thousands of tokens/turns | PRMs, critics (VAPO), step-level groups (GiGPO), self-distillation |
| Exploration in language space | Entropy collapse; huge action space | Clip-Higher, entropy-aware objectives, pass@k training, curricula |
| Reward hacking | Optimization finds loopholes | Robust envs, monitors, inoculation prompting, diverse safety training |
| Stability at scale | MoE routing, precision, off-policy staleness | GSPO, truncated IS, sequence masking, routing replay |
| Sample efficiency in the real world | Real trials are slow/expensive | World models, offline RL, sim-to-real, distillation |
| Continual learning | Forgetting, plasticity loss | On-policy RL, sparse/low-rank updates, memory |
| Evaluation | Benchmarks saturate or leak | Private/dynamic evals, agentic task suites, held-out environments |

---

## 16. Predictions

These are informed opinions, not facts.

1. **Environments and graders become the main moat**, more than algorithms. Expect environment marketplaces and
   standard interfaces.
2. **One mixed RL stage** with many reward types (verifiers, rubrics, safety rules, distillation signals) becomes the
   norm, replacing separate SFT/RLHF/reasoning stages.
3. **Dense signals beat sparse ones** for efficiency: on-policy (self-)distillation and learned critics will be combined
   with outcome rewards rather than replacing them.
4. **Agentic RL dominates compute** as tasks lengthen from minutes to hours or days.
5. **Continual, deployment-time learning** begins in narrow settings (personal assistants, coding agents within a repo).
6. **Safety work shifts toward training-time defenses** (hack-resistant environments, inoculation, monitorability) in
   addition to post-hoc filtering.
7. **Classic RL ideas return** — off-policy corrections, options/hierarchy, world models, curiosity — re-implemented for
   LLM agents.

---

## 17. For business users

**What is changing:** AI systems are moving from "trained once on text" to "trained by practice on tasks". RL is the
practice. The better a company can **define success precisely** (a test that passes, a ticket resolved, a rubric an
expert agrees with), the more it can benefit from RL-trained or RL-fine-tuned models.

**What to do now**
- Inventory tasks where success is **checkable** (data extraction with known answers, code with tests, forms with
  validation). These are ready for reinforcement fine-tuning today.
- For subjective tasks, invest in **rubrics written by your experts** — they become reusable reward functions.
- Build **sandboxes** where agents can practice safely before touching production systems.
- Budget for **evaluation and monitoring**: RL-trained systems are excellent at hitting the target you set, including
  the wrong target.

**Questions to ask vendors and teams:** How is the reward defined? How do you detect reward hacking? What happens to
quality on tasks *not* in the training set? Can the model's reasoning be audited?

---

## 18. For kids

Imagine a robot that is learning to play a new video game. At first it only got a gold star at the *very end* of each
level, so it took forever to learn. Scientists are now teaching robots in smarter ways:

- **A coach who comments on every move**, not just the final score (that's "distillation" and "process rewards").
- **Practice worlds inside the robot's head**, like daydreaming about a game before playing it (that's "world models").
- **Making up their own puzzles** to practice on (that's "self-play").
- **Fair-play rules**, so the robot doesn't learn to cheat to get stars (that's "safety" and stopping "reward hacking").

**Try it:** Play a game of "hot and cold" with a friend. First, only say "you found it!" at the end. Then play again
saying "warmer… colder…" at every step. Which way is faster? That's the difference between sparse and dense rewards!

---

## 19. References

- Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed., 2018 — Ch. 17 "Frontiers".
- Silver & Sutton, *Welcome to the Era of Experience*, 2025 (essay).
- DeepSeek-AI, *DeepSeek-R1*, 2025; *DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models*, arXiv 2512.02556.
- Liu et al., *Inference-Time Scaling for Generalist Reward Modeling* (DeepSeek-GRM), arXiv 2504.02495.
- Gunjal et al., *Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains*, arXiv 2507.17746.
- Zheng et al. (Qwen), *Group Sequence Policy Optimization*, arXiv 2507.18071; Qwen, *Soft Adaptive Policy Optimization*, arXiv 2511.20347.
- MiniMax, *MiniMax-M1* technical report (CISPO), 2025.
- Yu et al., *DAPO*, arXiv 2503.14476; Liu et al., *Understanding R1-Zero-Like Training*, arXiv 2503.20783; Yue et al., *VAPO*, arXiv 2504.05118.
- MacDiarmid et al. (Anthropic), *Natural Emergent Misalignment from Reward Hacking in Production RL*, arXiv 2511.18397.
- Thinking Machines Lab, *On-Policy Distillation* (Oct 2025) and *LoRA Without Regret* (Sept 2025), blog posts.
- *Reinforcement Learning via Self-Distillation*, arXiv 2601.20802; *Self-Distilled RLVR*, arXiv 2604.03128; *Rethinking On-Policy Distillation*, arXiv 2604.13016.
- Moonshot AI, *Kimi K2: Open Agentic Intelligence*, arXiv 2507.20534; Kimi K2.5 report, arXiv 2602.02276.
- *From Reasoning to Agentic: Credit Assignment in Reinforcement Learning for LLMs*, arXiv 2604.09459.
- *A First-Principles Derivation of LLM Policy Optimization: From Expected Reward to GRPO and Its Structural Extensions*, arXiv 2606.16733.
- Hafner et al., *Mastering Diverse Domains through World Models* (DreamerV3), *Nature*, 2025.
- Gao, Schulman & Hilton, *Scaling Laws for Reward Model Overoptimization*, 2022.
- OpenAI, Reinforcement fine-tuning guide (developers.openai.com), accessed 2026.
