"""
Multi-agent RL in miniature.

Part A (cooperative, CTDE value factorisation, PyTorch):
  The two-step cooperative game from the QMIX paper (Rashid et al., 2018).
    step 1 (state s1): agent 1's action picks the next state: a1=0 -> 2A, a1=1 -> 2B.
    state 2A: every joint action pays 7.
    state 2B: payoff [[0, 1], [1, 8]]  (only joint action (1,1) pays 8).
  Optimal team return is 8 (go to 2B and coordinate on (1,1)).
  We train, from the same uniformly-random joint-action data:
    * IQL  : independent Q-learners, each treating the other as part of the environment
    * VDN  : Q_tot = Q_1 + Q_2                       (Sunehag et al., 2017)
    * QMIX : Q_tot = f_mix(Q_1, Q_2; s), monotonic  (weights from a hypernetwork, |w| >= 0)

Part B (competitive, self-play, NumPy):
  Rock-paper-scissors. Naive "best-respond to the latest self" cycles forever, while
  fictitious play (best-respond to the AVERAGE of past selves, the idea behind leagues and
  PSRO-style population training) converges in average strategy to the Nash equilibrium.

Run:  python3 code/multi-agent-rl.py     (about 20-60 s on CPU)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)

# ============================================================================
# Part A: two-step cooperative game
# ============================================================================
S1, S2A, S2B, TERMINAL = 0, 1, 2, 3
PAYOFF_2B = np.array([[0.0, 1.0], [1.0, 8.0]])


def env_step(state, a1, a2):
    """Returns (next_state, team_reward, done)."""
    if state == S1:
        return (S2A if a1 == 0 else S2B), 0.0, False
    if state == S2A:
        return TERMINAL, 7.0, True
    return TERMINAL, float(PAYOFF_2B[a1, a2]), True


def collect_uniform(n_episodes, rng):
    data = []
    for _ in range(n_episodes):
        s, done = S1, False
        while not done:
            a1, a2 = rng.integers(2), rng.integers(2)
            s2, r, done = env_step(s, a1, a2)
            data.append((s, a1, a2, r, s2, float(done)))
            s = s2
    return [torch.tensor(np.array(x)) for x in zip(*data)]


def onehot_state(s):
    return F.one_hot(s.long(), 4).float()


class AgentQ(nn.Module):
    """Per-agent utility Q_i(o_i, .). Each agent observes the state + its own id."""

    def __init__(self, h=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4 + 2, h), nn.ReLU(), nn.Linear(h, 2))

    def forward(self, s, agent_id):
        idv = F.one_hot(torch.full_like(s, agent_id).long(), 2).float()
        return self.net(torch.cat([onehot_state(s), idv], -1))


class QMixer(nn.Module):
    """Monotonic mixing network: weights produced by hypernetworks of the state and
    passed through abs() so dQ_tot/dQ_i >= 0 (Rashid et al., 2018)."""

    def __init__(self, n_agents=2, embed=8):
        super().__init__()
        self.n, self.e = n_agents, embed
        self.hyper_w1 = nn.Linear(4, n_agents * embed)
        self.hyper_b1 = nn.Linear(4, embed)
        self.hyper_w2 = nn.Linear(4, embed)
        self.hyper_b2 = nn.Sequential(nn.Linear(4, embed), nn.ReLU(), nn.Linear(embed, 1))

    def forward(self, qs, s):              # qs: [B, n_agents]
        st = onehot_state(s)
        w1 = self.hyper_w1(st).abs().view(-1, self.n, self.e)
        h = F.elu(torch.bmm(qs.unsqueeze(1), w1).squeeze(1) + self.hyper_b1(st))
        w2 = self.hyper_w2(st).abs()
        return (h * w2).sum(-1) + self.hyper_b2(st).squeeze(-1)


def train(method, data, steps=3000, gamma=0.99, lr=5e-3, seed=0):
    torch.manual_seed(seed)
    agents = AgentQ()                          # parameter sharing (agent id in the input)
    mixer = QMixer() if method == "qmix" else None
    params = list(agents.parameters()) + (list(mixer.parameters()) if mixer else [])
    opt = torch.optim.Adam(params, lr=lr)
    s, a1, a2, r, s2, d = data
    for _ in range(steps):
        q1, q2 = agents(s, 0), agents(s, 1)
        q1a, q2a = q1.gather(1, a1[:, None]).squeeze(1), q2.gather(1, a2[:, None]).squeeze(1)
        with torch.no_grad():  # decentralised greedy maxima at s' (the IGM property)
            n1, n2 = agents(s2, 0).max(1).values, agents(s2, 1).max(1).values
        if method == "iql":    # each agent regresses its OWN Q onto the team reward
            loss = F.mse_loss(q1a, r + gamma * (1 - d) * n1) + F.mse_loss(q2a, r + gamma * (1 - d) * n2)
        else:
            if method == "vdn":
                q_tot = q1a + q2a
                with torch.no_grad():
                    nxt = n1 + n2
            else:
                q_tot = mixer(torch.stack([q1a, q2a], 1), s)
                with torch.no_grad():
                    nxt = mixer(torch.stack([n1, n2], 1), s2)
            loss = F.mse_loss(q_tot, r + gamma * (1 - d) * nxt)  # ONE team TD error (CTDE)
        opt.zero_grad(); loss.backward(); opt.step()
    return agents


@torch.no_grad()
def greedy_return(agents):
    s, done, G, path = S1, False, 0.0, []
    while not done:
        st = torch.tensor([s])
        a1, a2 = int(agents(st, 0).argmax()), int(agents(st, 1).argmax())
        path.append(("s1", "2A", "2B")[s] + f":({a1},{a2})")
        s, r, done = env_step(s, a1, a2)
        G += r
    return G, " -> ".join(path)


# ============================================================================
# Part B: self-play on rock-paper-scissors
# ============================================================================
RPS = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], float)  # row player's payoff


def exploitability(p):
    """How much a best-responder gains against mixed strategy p (0 at Nash)."""
    return float((RPS @ p).max())


def naive_self_play(iters=12):
    p = np.array([1.0, 0, 0])
    seq = []
    for _ in range(iters):
        br = np.eye(3)[(RPS @ p).argmax()]  # best response to the CURRENT self
        p = br
        seq.append("RPS"[int(br.argmax())])
    return "".join(seq), exploitability(p)


def fictitious_play(iters=3000):
    counts = np.array([1.0, 0, 0])
    for _ in range(iters):
        avg = counts / counts.sum()
        counts += np.eye(3)[(RPS @ avg).argmax()]  # best response to the AVERAGE past self
    avg = counts / counts.sum()
    return avg, exploitability(avg)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    data = collect_uniform(2000, rng)
    print("Part A - two-step cooperative game (optimal team return = 8)")
    print(f"{'method':<8}{'greedy return':>15}   greedy joint actions")
    for m in ["iql", "vdn", "qmix"]:
        G, path = greedy_return(train(m, data))
        print(f"{m.upper():<8}{G:>15.1f}   {path}")

    print("\nPart B - self-play on rock-paper-scissors")
    seq, ex = naive_self_play()
    print(f"naive self-play (latest self)  : {seq} ... cycles; exploitability {ex:.2f}")
    avg, ex = fictitious_play()
    print(f"fictitious play (average self) : avg strategy {np.round(avg, 3)}; exploitability {ex:.3f}")
