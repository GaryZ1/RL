"""Agentic / multi-turn tool-use RL in miniature.

Environment ("two-hop lookup"): a hidden table maps keys -> entries. The question names key q.
table[q] is a pointer KEY(k2); table[k2] holds VAL(v). The agent must SEARCH(q), read the pointer,
SEARCH(k2), read the value, then ANSWER(v). Up to 4 turns. Reward 1 for the right answer, minus a
small cost per search. Without using the tool the agent can only guess (1/6).

(a) Reusable pieces:
    - build_token_sequence_and_mask : flatten a multi-turn trajectory into one token stream with a
      loss mask that is 1 on agent-generated tokens and 0 on prompt / tool-output tokens
      (Search-R1's "retrieved token masking"; every agent-RL framework does this).
    - episode_group_advantages      : GRPO-style trajectory-level advantages.
    - gigpo_advantages              : GiGPO-style episode advantages + step-level advantages from
      'anchor states' (identical states encountered by different rollouts of the same task).
    - masked_pg_loss                : PG loss over agent tokens only.
(b) Train the same tiny policy with episode-level GRPO vs. GiGPO-style advantages.

Run:  python3 code/agentic-multi-turn-rl.py     (CPU, ~1 min)
"""
import copy
import random
from collections import defaultdict

import torch
import torch.nn as nn

torch.set_num_threads(1)

N_KEYS = 4
MAX_TURNS = 4
SEARCH_COST = 0.01
# action tokens: SEARCH_0..5, ANSWER_0..5 ; observation tokens: KEY_0..5, VAL_0..5 ; NONE
N_ACT = 2 * N_KEYS
OBS_NONE = 2 * N_KEYS
VOCAB_NAMES = ([f"SEARCH_{i}" for i in range(N_KEYS)] + [f"ANSWER_{i}" for i in range(N_KEYS)] +
               [f"<obs>KEY_{i}</obs>" for i in range(N_KEYS)] + [f"<obs>VAL_{i}</obs>" for i in range(N_KEYS)])


class TwoHopLookupEnv:
    def __init__(self, seed, hops=2):
        rng = random.Random(seed)
        self.q = rng.randrange(N_KEYS)
        self.k2 = rng.choice([k for k in range(N_KEYS) if k != self.q])
        self.v = rng.randrange(N_KEYS)
        self.table = {}
        for k in range(N_KEYS):                         # distractor entries
            self.table[k] = ("KEY", rng.randrange(N_KEYS)) if rng.random() < 0.5 else ("VAL", rng.randrange(N_KEYS))
        if hops == 2:
            self.table[self.q] = ("KEY", self.k2)
            self.table[self.k2] = ("VAL", self.v)
        else:                                           # one-hop task: the entry holds the value
            self.table[self.q] = ("VAL", self.v)

    def step(self, action):
        """Returns (obs_token or None, reward, done)."""
        if action < N_KEYS:                             # SEARCH(key)
            kind, x = self.table[action]
            obs = x if kind == "KEY" else N_KEYS + x    # obs id in 0..11
            return obs, -SEARCH_COST, False
        answer = action - N_KEYS
        return None, float(answer == self.v), True


def state_features(q, last_obs, turn):
    return torch.cat([nn.functional.one_hot(torch.tensor(q), N_KEYS),
                      nn.functional.one_hot(torch.tensor(last_obs), 2 * N_KEYS + 1),
                      nn.functional.one_hot(torch.tensor(turn), MAX_TURNS)]).float()


N_FEAT = N_KEYS + 2 * N_KEYS + 1 + MAX_TURNS


@torch.no_grad()
def rollout(policy, env):
    """One episode. Returns list of turns: dict(state_key, feat, action, obs, reward)."""
    turns, last_obs = [], OBS_NONE
    for turn in range(MAX_TURNS):
        f = state_features(env.q, last_obs, turn)
        a = torch.distributions.Categorical(logits=policy(f)).sample().item()
        if turn == MAX_TURNS - 1 and a < N_KEYS:       # forced to stop: searching on last turn = fail
            turns.append(dict(state=(env.q, last_obs, turn), feat=f, action=a, obs=None, reward=-SEARCH_COST))
            break
        obs, r, done = env.step(a)
        turns.append(dict(state=(env.q, last_obs, turn), feat=f, action=a, obs=obs, reward=r))
        if done:
            break
        last_obs = obs
    return turns


# ----------------------------------------------------------------------------------------
# (a) Masking and advantages
# ----------------------------------------------------------------------------------------
def build_token_sequence_and_mask(prompt_tokens, turns, obs_offset=N_ACT):
    """Flatten [prompt][act_1][obs_1][act_2][obs_2]... into token ids + loss mask.
    In a real LLM each act/obs is many tokens; the rule is the same: mask = 1 only for tokens the
    policy generated, 0 for prompt and tool/environment output (which the policy did not choose)."""
    tokens, mask = list(prompt_tokens), [0] * len(prompt_tokens)
    for t in turns:
        tokens.append(t["action"])
        mask.append(1)
        if t["obs"] is not None:
            tokens.append(obs_offset + t["obs"])
            mask.append(0)
    return tokens, mask


def discounted_returns(rewards, gamma):
    out, g = [], 0.0
    for r in reversed(rewards):
        g = r + gamma * g
        out.append(g)
    return out[::-1]


def episode_group_advantages(episodes, G, eps=1e-6):
    """GRPO for agents: one scalar per trajectory = (R - mean_group)/std_group, broadcast to all turns."""
    R = torch.tensor([sum(t["reward"] for t in ep) for ep in episodes]).view(-1, G)
    A = ((R - R.mean(1, keepdim=True)) / (R.std(1, keepdim=True) + eps)).view(-1)
    return [[A[i].item()] * len(ep) for i, ep in enumerate(episodes)]


def gigpo_advantages(episodes, G, gamma=0.95, omega=1.0, eps=1e-6, step_norm="mean"):
    """GiGPO-style (Feng et al. 2025): A = A_episode + omega * A_step.
    Step level: within each group (same task), turns that start from the SAME environment state
    (an 'anchor state') form a step-group; their discounted returns-from-here are normalised
    against each other. Anchor groups of size 1 get A_step = 0.
    step_norm="mean" only centres the returns; "std" also divides by the std -- beware: then tiny
    return differences (e.g. one extra search costing 0.01) are blown up to +-1."""
    A_ep = episode_group_advantages(episodes, G, eps)
    A = [list(a) for a in A_ep]
    for g0 in range(0, len(episodes), G):
        buckets = defaultdict(list)                    # anchor state -> [(ep idx, turn idx, return)]
        for i in range(g0, g0 + G):
            rets = discounted_returns([t["reward"] for t in episodes[i]], gamma)
            for j, t in enumerate(episodes[i]):
                buckets[t["state"]].append((i, j, rets[j]))
        for items in buckets.values():
            if len(items) < 2:
                continue
            r = torch.tensor([x[2] for x in items])
            a_step = r - r.mean()
            if step_norm == "std":
                a_step = a_step / (r.std() + eps)
            for (i, j, _), a in zip(items, a_step.tolist()):
                A[i][j] += omega * a
    return A


def masked_pg_loss(logp, adv, mask):
    """logp/adv/mask: [N_turns_total] for agent tokens; tool tokens never enter (mask=0)."""
    return -(logp * adv * mask).sum() / mask.sum().clamp(min=1)


# ----------------------------------------------------------------------------------------
# (b) Training
# ----------------------------------------------------------------------------------------
def sft_warm_start(policy, steps=40, lr=3e-3):
    """'Cold-start' SFT on ONE-HOP demonstrations only (search q -> read value -> answer).
    The agent learns the tool format but has never seen a pointer (KEY) observation."""
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    for _ in range(steps):
        feats, acts = [], []
        for _ in range(64):
            env = TwoHopLookupEnv(random.randrange(10 ** 9), hops=1)
            feats += [state_features(env.q, OBS_NONE, 0), state_features(env.q, N_KEYS + env.v, 1)]
            acts += [env.q, N_KEYS + env.v]
        loss = nn.functional.cross_entropy(policy(torch.stack(feats)), torch.tensor(acts))
        opt.zero_grad()
        loss.backward()
        opt.step()


def train(adv_mode, iters=300, n_tasks=16, G=8, lr=5e-3, seed=0, kl_coef=0.0):
    torch.manual_seed(seed)
    random.seed(seed)
    policy = nn.Sequential(nn.Linear(N_FEAT, 64), nn.Tanh(), nn.Linear(64, N_ACT))
    sft_warm_start(policy)
    ref = copy.deepcopy(policy)                         # SFT policy as KL anchor
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    log = []
    for it in range(iters):
        episodes = []
        for _ in range(n_tasks):
            task_seed = random.randrange(10 ** 9)
            for _ in range(G):                          # G rollouts of the SAME task -> a group
                episodes.append(rollout(policy, TwoHopLookupEnv(task_seed, hops=2)))
        if adv_mode == "grpo":
            adv = episode_group_advantages(episodes, G)
        else:
            adv = gigpo_advantages(episodes, G, step_norm="std" if adv_mode == "gigpo-std" else "mean")
        feats = torch.stack([t["feat"] for ep in episodes for t in ep])
        acts = torch.tensor([t["action"] for ep in episodes for t in ep])
        A = torch.tensor([a for ep_a in adv for a in ep_a])
        logp = torch.log_softmax(policy(feats), -1).gather(1, acts[:, None]).squeeze(1)
        loss = masked_pg_loss(logp, A, torch.ones_like(A))
        if kl_coef > 0:   # exact KL(pi || pi_sft) over the action vocabulary at visited states
            lp_all = torch.log_softmax(policy(feats), -1)
            with torch.no_grad():
                lr_all = torch.log_softmax(ref(feats), -1)
            loss = loss + kl_coef * (lp_all.exp() * (lp_all - lr_all)).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        success = sum(ep[-1]["reward"] > 0.5 for ep in episodes) / len(episodes)
        log.append((success, sum(len(ep) for ep in episodes) / len(episodes)))
    return policy, log


if __name__ == "__main__":
    env = TwoHopLookupEnv(seed=3)
    demo = [dict(action=env.q, obs=env.k2), dict(action=env.k2, obs=N_KEYS + env.v),
            dict(action=N_KEYS + env.v, obs=None)]
    toks, mask = build_token_sequence_and_mask([], demo)
    print("ideal trajectory  :", " ".join(VOCAB_NAMES[t] for t in toks))
    print("loss mask         :", mask, "(tool outputs are masked out)\n")
    print("one-hop SFT warm start, then RL on two-hop tasks (pointer observations never seen in SFT)")
    print("success rate (mean turns) by training phase, mean of 2 seeds; ideal = 1.0 with 3 turns")
    trained = None
    for mode in ["grpo", "gigpo", "gigpo-std"]:
        res = []
        for s in range(2):
            pol, lg = train(mode, seed=s)
            res.append(lg)
            if mode == "gigpo" and s == 0:
                trained = pol
        cells = []
        for sl in [slice(0, 20), slice(100, 150), slice(-20, None)]:
            succ = sum(sum(x[0] for x in r[sl]) / len(r[sl]) for r in res) / 2
            turns = sum(sum(x[1] for x in r[sl]) / len(r[sl]) for r in res) / 2
            cells.append(f"{succ:.3f} ({turns:.2f})")
        print(f"  {mode:9s} its 0-20: {cells[0]} | its 100-150: {cells[1]} | last 20: {cells[2]}")
    policy = trained
    with torch.no_grad():
        probs = [torch.softmax(policy(state_features(q, OBS_NONE, 0)), -1) for q in range(N_KEYS)]
    print("P(first action = SEARCH(q)) per question key:", [round(p_[q].item(), 2) for q, p_ in enumerate(probs)])
    for task in [12345, 777, 2026]:
        env = TwoHopLookupEnv(seed=task)
        ep = rollout(policy, env)
        toks, mask = build_token_sequence_and_mask([], ep)
        print(f"trained agent, new task (q={env.q}, k2={env.k2}, answer={env.v}):",
              " ".join(VOCAB_NAMES[t] for t in toks), "| reward", round(sum(t["reward"] for t in ep), 2))
