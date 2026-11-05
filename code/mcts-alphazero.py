"""AlphaZero-style self-play (PUCT MCTS + policy/value network) on tic-tac-toe.

Self-contained: numpy + torch. Trains in ~1 minute on CPU and then plays against
(a) a uniformly random player and (b) a perfect minimax player.
    python3 code/mcts-alphazero.py
"""
import functools
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
rng = np.random.default_rng(0)
torch.manual_seed(0)

# ----------------------------------------------------------------------------------------------
# 1. Game: tic-tac-toe in CANONICAL form -- the board is always seen from the player to move (+1)
# ----------------------------------------------------------------------------------------------
LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]


def legal_moves(b):
    return np.flatnonzero(b == 0)


def next_state(b, a):
    """Current player (+1) plays square a; flip signs so the opponent becomes the player to move."""
    b2 = b.copy()
    b2[a] = 1
    return -b2


def terminal_value(b):
    """Value for the player to move: -1 if the opponent just completed a line, 0 for a draw, None otherwise."""
    for i, j, k in LINES:
        if b[i] + b[j] + b[k] == -3:
            return -1.0
    if not (b == 0).any():
        return 0.0
    return None


# ----------------------------------------------------------------------------------------------
# 2. Policy/value network f_theta(s) -> (p, v)
# ----------------------------------------------------------------------------------------------
class PolicyValueNet(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(27, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.policy_head, self.value_head = nn.Linear(hidden, 9), nn.Linear(hidden, 1)

    @staticmethod
    def encode(boards):  # (N, 9) in {-1,0,1} -> (N, 27) one-hot planes: mine / theirs / empty
        boards = torch.as_tensor(np.asarray(boards), dtype=torch.float32)
        return torch.cat([(boards == 1).float(), (boards == -1).float(), (boards == 0).float()], dim=-1)

    def forward(self, boards):
        h = self.body(self.encode(boards))
        return self.policy_head(h), torch.tanh(self.value_head(h)).squeeze(-1)

    @torch.no_grad()
    def predict(self, b):
        logits, v = self(b[None])
        logits = logits[0].numpy()
        mask = np.full(9, -np.inf)
        mask[legal_moves(b)] = 0.0
        p = np.exp(logits + mask - np.max(logits + mask))
        return p / p.sum(), float(v[0])


# ----------------------------------------------------------------------------------------------
# 3. MCTS with the PUCT selection rule used by AlphaGo Zero / AlphaZero
# ----------------------------------------------------------------------------------------------
class Node:
    __slots__ = ("prior", "N", "W", "children")

    def __init__(self, prior):
        self.prior, self.N, self.W, self.children = prior, 0, 0.0, {}

    def Q(self):  # mean value from the perspective of the player who chose the move INTO this node
        return self.W / self.N if self.N else 0.0


class MCTS:
    def __init__(self, net, n_sims=50, c_puct=1.5, dir_alpha=0.6, dir_eps=0.25):
        self.net, self.n_sims, self.c_puct, self.dir_alpha, self.dir_eps = net, n_sims, c_puct, dir_alpha, dir_eps

    def _expand(self, node, b):
        p, v = self.net.predict(b)
        for a in legal_moves(b):
            node.children[a] = Node(p[a])
        return v

    def _select(self, node):
        sqrt_n = math.sqrt(node.N)
        best, best_score = None, -1e9
        for a, child in node.children.items():
            # PUCT: exploit Q, explore in proportion to the network prior and inversely to visit count
            u = self.c_puct * child.prior * sqrt_n / (1 + child.N)
            score = child.Q() + u
            if score > best_score:
                best, best_score = a, score
        return best, node.children[best]

    def search(self, b, add_noise=False):
        root = Node(1.0)
        self._expand(root, b)
        if add_noise:  # Dirichlet noise at the root keeps self-play exploring
            acts = list(root.children)
            noise = rng.dirichlet([self.dir_alpha] * len(acts))
            for a, n in zip(acts, noise):
                root.children[a].prior = (1 - self.dir_eps) * root.children[a].prior + self.dir_eps * n
        for _ in range(self.n_sims):
            node, s, path = root, b, [root]
            while node.children:                     # 1) SELECT down the tree
                a, node = self._select(node)
                s = next_state(s, a)
                path.append(node)
            v = terminal_value(s)                    # 2) EVALUATE leaf (exact if terminal)
            if v is None:
                v = self._expand(node, s)            # 3) EXPAND with network priors, value v
            for n in reversed(path):                 # 4) BACK UP, flipping sign every ply
                n.N += 1
                n.W += -v                            # node stores value for the player who moved into it
                v = -v
        visits = np.zeros(9)
        for a, child in root.children.items():
            visits[a] = child.N
        return visits


# ----------------------------------------------------------------------------------------------
# 4. Self-play data generation, with the 8 board symmetries as data augmentation
# ----------------------------------------------------------------------------------------------
IDX = np.arange(9).reshape(3, 3)
SYMS = [np.rot90(IDX, k).ravel() for k in range(4)] + [np.rot90(IDX.T, k).ravel() for k in range(4)]


def self_play_game(mcts, temp_moves=5):
    b, history = np.zeros(9), []
    while terminal_value(b) is None:
        visits = mcts.search(b, add_noise=True)
        pi = visits / visits.sum()                   # policy target = normalized visit counts
        history.append((b.copy(), pi))
        if len(history) <= temp_moves:               # temperature 1 for the opening, then greedy
            a = rng.choice(9, p=pi)
        else:
            a = int(np.argmax(visits))
        b = next_state(b, a)
    z = terminal_value(b)                            # outcome for the player to move at the END
    data = []
    for s, pi in reversed(history):                  # walk back, flipping perspective each ply
        z = -z
        for perm in SYMS:
            data.append((s[perm], pi[perm], z))
    return data


def train(net, opt, buffer, steps=60, batch=128):
    for _ in range(steps):
        idx = rng.integers(0, len(buffer), size=batch)
        s = np.stack([buffer[i][0] for i in idx])
        pi = torch.as_tensor(np.stack([buffer[i][1] for i in idx]), dtype=torch.float32)
        z = torch.as_tensor([buffer[i][2] for i in idx], dtype=torch.float32)
        logits, v = net(s)
        # AlphaZero loss: (z - v)^2 - pi^T log p   (+ L2 via weight_decay in the optimizer)
        loss = F.mse_loss(v, z) - (pi * F.log_softmax(logits, -1)).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item()


# ----------------------------------------------------------------------------------------------
# 5. Opponents for evaluation: random and perfect (minimax)
# ----------------------------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def minimax_value(key):
    b = np.array(key)
    tv = terminal_value(b)
    if tv is not None:
        return tv
    return max(-minimax_value(tuple(next_state(b, a))) for a in legal_moves(b))


def optimal_moves(b):
    vals = {a: -minimax_value(tuple(next_state(b, a))) for a in legal_moves(b)}
    best = max(vals.values())
    return {a for a, v in vals.items() if v == best}


def perfect_move(b):
    return rng.choice(sorted(optimal_moves(b)))   # random among game-theoretically optimal moves


def all_positions():
    """Every reachable non-terminal position (4,520 in canonical form) -- used to grade move quality."""
    seen, stack = set(), [tuple(np.zeros(9))]
    while stack:
        key = stack.pop()
        if key in seen or terminal_value(np.array(key)) is not None:
            continue
        seen.add(key)
        stack.extend(tuple(next_state(np.array(key), a)) for a in legal_moves(np.array(key)))
    return [np.array(k) for k in seen]


def play(agent_fn, opp_fn, agent_first):
    b, agent_turn = np.zeros(9), agent_first
    while terminal_value(b) is None:
        b = next_state(b, agent_fn(b) if agent_turn else opp_fn(b))
        agent_turn = not agent_turn
    # terminal_value is for the player to move; if that is the opponent, flip it
    return terminal_value(b) * (1 if agent_turn else -1)   # +1 agent win, 0 draw, -1 loss


def match(agent_fn, opp_fn, games=40):
    res = [play(agent_fn, opp_fn, agent_first=(g % 2 == 0)) for g in range(games)]
    return sum(r > 0 for r in res), sum(r == 0 for r in res), sum(r < 0 for r in res)


def main(iterations=40, games_per_iter=15):
    net = PolicyValueNet()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
    mcts, buffer, t0 = MCTS(net, n_sims=50), [], time.time()
    random_fn = lambda b: rng.choice(legal_moves(b))
    for it in range(1, iterations + 1):
        for _ in range(games_per_iter):
            buffer.extend(self_play_game(mcts))
        buffer = buffer[-20_000:]                    # sliding window replay buffer
        loss = train(net, opt, buffer)
        if it % 5 == 0:
            print(f"iter {it:3d} | buffer {len(buffer):6d} | loss {loss:.3f} | {time.time() - t0:5.1f}s")

    eval_mcts = MCTS(net, n_sims=100)
    az_fn = lambda b: int(np.argmax(eval_mcts.search(b)))
    raw_fn = lambda b: int(np.argmax(net.predict(b)[0]))      # network alone, no search
    positions = all_positions()
    sample = [positions[i] for i in rng.choice(len(positions), size=500, replace=False)]
    for name, fn in [("raw policy net ", raw_fn), ("net + MCTS(100)", az_fn)]:
        w, d, l = match(fn, random_fn, games=100)
        w2, d2, l2 = match(fn, perfect_move, games=100)
        opt_rate = np.mean([fn(b) in optimal_moves(b) for b in sample])
        print(f"{name} | optimal-move rate {opt_rate:5.1%} | vs random W/D/L {w:3d}/{d:3d}/{l:3d} "
              f"| vs perfect W/D/L {w2:3d}/{d2:3d}/{l2:3d}")
    _, v_empty = net.predict(np.zeros(9))
    print(f"value of empty board predicted {v_empty:+.2f} (true game-theoretic value: 0 = draw)")


if __name__ == "__main__":
    main()
