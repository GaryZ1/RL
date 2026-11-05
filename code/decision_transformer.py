"""
Decision Transformer (Chen et al., 2021) in miniature.

A small causal transformer is trained, purely by supervised learning, to predict the
dataset action given the recent context of (return-to-go, state, action) tokens.
At test time we *ask* for a return by setting the initial return-to-go, and the
model tries to behave like a trajectory that achieved it.

Environment: 6x6 grid, start (0,0), goal (5,5), reward -1 per step, max 20 steps
(so returns are in [-20, -10]; -10 is optimal). The offline dataset mixes
behavior policies of very different quality (epsilon-greedy with random epsilon).

Run:  python3 code/decision_transformer.py   (roughly 30-90 s on CPU)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
np.random.seed(0)
torch.manual_seed(0)

N, MAX_T, N_ACT = 6, 20, 4
MOVES = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # R, U, L, D
K = 8  # context length in timesteps (3K tokens)


# ----------------------------------------------------------------------------
# 1) Environment + mixed-quality offline dataset
# ----------------------------------------------------------------------------
class Grid:
    def reset(self):
        self.pos, self.t = (0, 0), 0
        return self.state()

    def state(self):
        return self.pos[0] * N + self.pos[1]

    def step(self, a):
        dx, dy = MOVES[a]
        self.pos = (min(max(self.pos[0] + dx, 0), N - 1), min(max(self.pos[1] + dy, 0), N - 1))
        self.t += 1
        done = self.pos == (N - 1, N - 1) or self.t >= MAX_T
        return self.state(), -1.0, done


def behavior(env, eps):
    if np.random.rand() < eps:
        return np.random.randint(N_ACT)
    x, y = env.pos
    return np.random.choice([a for a, ok in [(0, x < N - 1), (1, y < N - 1)] if ok] or [0])


def make_dataset(n_traj=1500):
    env, trajs = Grid(), []
    for _ in range(n_traj):
        eps = np.random.rand()  # quality varies from expert to random
        s, done, S, A, R = env.reset(), False, [], [], []
        while not done:
            a = behavior(env, eps)
            S.append(s); A.append(a)
            s, r, done = env.step(a)
            R.append(r)
        rtg = np.cumsum(R[::-1])[::-1].copy()  # return-to-go  \hat R_t = sum_{t'>=t} r_t'
        trajs.append((np.array(S), np.array(A), rtg.astype(np.float32)))
    return trajs


def sample_batch(trajs, B=64):
    """Random windows of length K, left-padded. Returns tensors [B, K]."""
    S = np.zeros((B, K), np.int64); A = np.zeros((B, K), np.int64)
    G = np.zeros((B, K), np.float32); T = np.zeros((B, K), np.int64); M = np.zeros((B, K), np.float32)
    for b in range(B):
        s, a, g = trajs[np.random.randint(len(trajs))]
        end = np.random.randint(1, len(s) + 1)
        st = max(0, end - K); L = end - st
        S[b, -L:], A[b, -L:], G[b, -L:] = s[st:end], a[st:end], g[st:end]
        T[b, -L:], M[b, -L:] = np.arange(st, end), 1.0
    return [torch.tensor(x) for x in (S, A, G, T, M)]


# ----------------------------------------------------------------------------
# 2) The Decision Transformer
# ----------------------------------------------------------------------------
class DecisionTransformer(nn.Module):
    def __init__(self, d=64, n_layer=2, n_head=4, rtg_scale=20.0):
        super().__init__()
        self.rtg_scale = rtg_scale
        self.emb_s = nn.Embedding(N * N, d)
        self.emb_a = nn.Embedding(N_ACT, d)
        self.emb_r = nn.Linear(1, d)
        self.emb_t = nn.Embedding(MAX_T + 1, d)      # timestep embedding shared by the 3 tokens
        self.emb_type = nn.Parameter(torch.zeros(3, d))
        layer = nn.TransformerEncoderLayer(d, n_head, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, n_layer, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, N_ACT)

    def forward(self, S, A, G, T, M):
        B, L = S.shape
        te = self.emb_t(T)
        r_tok = self.emb_r(G[..., None] / self.rtg_scale) + te + self.emb_type[0]
        s_tok = self.emb_s(S) + te + self.emb_type[1]
        a_tok = self.emb_a(A) + te + self.emb_type[2]
        x = torch.stack([r_tok, s_tok, a_tok], 2).reshape(B, 3 * L, -1)  # (R_1,s_1,a_1,R_2,...)
        causal = torch.triu(torch.ones(3 * L, 3 * L, dtype=torch.bool), 1)
        pad = (M == 0).repeat_interleave(3, dim=1)
        h = self.tf(x, mask=causal, src_key_padding_mask=pad)
        h = self.ln(h).reshape(B, L, 3, -1)
        return self.head(h[:, :, 1])  # predict a_t from the output at the s_t token


def dt_loss(model, batch):
    S, A, G, T, M = batch
    logits = model(S, A, G, T, M)
    ce = F.cross_entropy(logits.reshape(-1, N_ACT), A.reshape(-1), reduction="none")
    return (ce * M.reshape(-1)).sum() / M.sum()


@torch.no_grad()
def rollout(model, target_return, n=50):
    env, rets = Grid(), []
    for _ in range(n):
        s, done = env.reset(), False
        S, A, G, T = [s], [0], [target_return], [0]
        ret = 0.0
        while not done:
            ctx = [torch.tensor([x[-K:]]) for x in (S, A, G, T)]
            M = torch.ones_like(ctx[0], dtype=torch.float32)
            logits = model(ctx[0], ctx[1], ctx[2].float(), ctx[3], M)[0, -1]
            a = int(torch.distributions.Categorical(logits=logits).sample())
            A[-1] = a
            s, r, done = env.step(a)
            ret += r
            S.append(s); A.append(0); G.append(G[-1] - r); T.append(min(env.t, MAX_T))  # R_{t+1} = R_t - r_t
        rets.append(ret)
    return float(np.mean(rets)), float(np.std(rets))


if __name__ == "__main__":
    trajs = make_dataset()
    data_rets = np.array([t[2][0] for t in trajs])
    print(f"Dataset: {len(trajs)} trajectories; return mean {data_rets.mean():.1f}, "
          f"min {data_rets.min():.0f}, max {data_rets.max():.0f} (optimal = -10)")
    model = DecisionTransformer()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for step in range(1500):
        loss = dt_loss(model, sample_batch(trajs))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 500 == 0:
            print(f"step {step:4d}  action CE loss {loss.item():.3f}")
    model.eval()
    print("\nReturn-conditioned rollouts (sampled actions, 50 episodes each):")
    print(f"{'target return':>14}{'achieved mean':>15}{'std':>7}")
    for target in [-10, -13, -16, -20]:
        m, sd = rollout(model, float(target))
        print(f"{target:>14}{m:>15.2f}{sd:>7.2f}")
    print("\nA well-trained DT tracks the requested return: high targets -> near-optimal paths,\n"
          "low targets -> deliberately meandering behaviour seen in the low-quality data.")
