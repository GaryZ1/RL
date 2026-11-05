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
import random
from collections import defaultdict

import torch
import torch.nn as nn

torch.set_num_threads(1)

N_KEYS = 6
MAX_TURNS = 4
SEARCH_COST = 0.05
# action tokens: SEARCH_0..5, ANSWER_0..5 ; observation tokens: KEY_0..5, VAL_0..5 ; NONE
N_ACT = 2 * N_KEYS
OBS_NONE = 2 * N_KEYS
VOCAB_NAMES = ([f"SEARCH_{i}" for i in range(N_KEYS)] + [f"ANSWER_{i}" for i in range(N_KEYS)] +
               [f"<obs>KEY_{i}</obs>" for i in range(N_KEYS)] + [f"<obs>VAL_{i}</obs>" for i in range(N_KEYS)])


class TwoHopLookupEnv:
    def __init__(self, seed):
        rng = random.Random(seed)
        self.q = rng.randrange(N_KEYS)
        self.k2 = rng.choice([k for k in range(N_KEYS) if k != self.q])
        self.v = rng.randrange(N_KEYS)
        self.table = {}
        for k in range(N_KEYS):                         # distractor entries
            self.table[k] = ("KEY", rng.randrange(N_KEYS)) if rng.random() < 0.5 else ("VAL", rng.randrange(N_KEYS))
        self.table[self.q] = ("KEY", self.k2)
        self.table[self.k2] = ("VAL", self.v)

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


def gigpo_advantages(episodes, G, gamma=0.95, omega=1.0, eps=1e-6):
    """GiGPO-style (Feng et al. 2025): A = A_episode + omega * A_step.
    Step level: within each group (same task), turns that start from the SAME environment state
    (an 'anchor state') form a step-group; their discounted returns-from-here are normalised
    against each other. Anchor groups of size 1 get A_step = 0."""
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
            std = r.std()
            a_step = (r - r.mean()) / (std + eps) if std > 0 else torch.zeros_like(r)
            for (i, j, _), a in zip(items, a_step.tolist()):
                A[i][j] += omega * a
    return A


def masked_pg_loss(logp, adv, mask):
    """logp/adv/mask: [N_turns_total] for agent tokens; tool tokens never enter (mask=0)."""
    return -(logp * adv * mask).sum() / mask.sum().clamp(min=1)


# ----------------------------------------------------------------------------------------
# (b) Training
# ----------------------------------------------------------------------------------------
def train(adv_mode, iters=150, n_tasks=16, G=8, lr=3e-3, seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    policy = nn.Sequential(nn.Linear(N_FEAT, 64), nn.Tanh(), nn.Linear(64, N_ACT))
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    log = []
    for it in range(iters):
        episodes = []
        for _ in range(n_tasks):
            task_seed = random.randrange(10 ** 9)
            for _ in range(G):                          # G rollouts of the SAME task -> a group
                episodes.append(rollout(policy, TwoHopLookupEnv(task_seed)))
        adv = (episode_group_advantages(episodes, G) if adv_mode == "grpo"
               else gigpo_advantages(episodes, G))
        feats = torch.stack([t["feat"] for ep in episodes for t in ep])
        acts = torch.tensor([t["action"] for ep in episodes for t in ep])
        A = torch.tensor([a for ep_a in adv for a in ep_a])
        logp = torch.log_softmax(policy(feats), -1).gather(1, acts[:, None]).squeeze(1)
        loss = masked_pg_loss(logp, A, torch.ones_like(A))
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
    toks, mask = build_token_sequence_and_mask([N_ACT + 2 * N_KEYS] * 0, demo)
    print("ideal trajectory  :", " ".join(VOCAB_NAMES[t] for t in toks))
    print("loss mask         :", mask, "(tool outputs are masked out)\n")
    for mode in ["grpo", "gigpo"]:
        res = [train(mode, seed=s)[1] for s in range(2)]
        for name, sl in [("first 10 its", slice(0, 10)), ("its 40-60", slice(40, 60)), ("last 10 its", slice(-10, None))]:
            succ = sum(sum(x[0] for x in r[sl]) / len(r[sl]) for r in res) / 2
            turns = sum(sum(x[1] for x in r[sl]) / len(r[sl]) for r in res) / 2
            print(f"{mode:6s} {name:13s} success {succ:.3f}  mean turns {turns:.2f}")
        print()
    policy, _ = train("gigpo", seed=0)
    env = TwoHopLookupEnv(seed=12345)
    ep = rollout(policy, env)
    toks, mask = build_token_sequence_and_mask([], ep)
    print(f"trained agent on a new task (q={env.q}, answer={env.v}):", " ".join(VOCAB_NAMES[t] for t in toks))
