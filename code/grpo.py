"""GRPO (Group Relative Policy Optimization) -- loss on tensors + tiny runnable toy demo.

Part (a): `group_relative_advantages` and `grpo_loss` operate on plain tensors of per-token
log-probs, rewards and masks -- the same shapes TRL / verl / OpenRLHF use internally.
Part (b): a tiny GRU "language model" learns to emit 4 digits (0..3) that sum to a target
given in the prompt. Reward is verifiable (1 if the sum is right, else 0).

Run:  python3 code/grpo.py      (CPU, ~20-40 s)
"""
import copy

import torch
import torch.nn as nn

torch.manual_seed(0)
torch.set_num_threads(1)  # tiny models: multi-threading only adds overhead


# ----------------------------------------------------------------------------------------
# (a) Reusable pieces: advantages, KL estimator, GRPO loss
# ----------------------------------------------------------------------------------------
def group_relative_advantages(rewards: torch.Tensor, group_size: int,
                              scale_by_std: bool = True, eps: float = 1e-6) -> torch.Tensor:
    """rewards: [B] with B = n_prompts * group_size, completions of one prompt contiguous.
    Returns per-sequence advantages A_i = (r_i - mean_group) / (std_group + eps)."""
    r = rewards.view(-1, group_size)
    mean = r.mean(dim=1, keepdim=True)
    adv = r - mean
    if scale_by_std:  # original GRPO; Dr. GRPO sets this False
        adv = adv / (r.std(dim=1, keepdim=True) + eps)
    return adv.view(-1)


def k3_kl(logp: torch.Tensor, logp_ref: torch.Tensor) -> torch.Tensor:
    """Per-token k3 estimator of KL(pi_theta || pi_ref) (Schulman's 'approximating KL' note):
    k3 = exp(logp_ref - logp) - (logp_ref - logp) - 1  >= 0, unbiased when tokens ~ pi_theta."""
    d = logp_ref - logp
    return torch.exp(d) - d - 1.0


def masked_mean(x, mask, dim=None):
    return (x * mask).sum(dim) / mask.sum(dim).clamp(min=1.0)


def grpo_loss(logp: torch.Tensor, logp_old: torch.Tensor, logp_ref: torch.Tensor,
              advantages: torch.Tensor, mask: torch.Tensor, clip_eps: float = 0.2,
              beta: float = 0.04, loss_agg: str = "seq-mean-token-mean",
              kl_is_correction: bool = False):
    """GRPO surrogate (to MINIMISE).

    logp, logp_old, logp_ref: [B, T] log pi(o_t | q, o_<t) of the *sampled* tokens under the
                              current, behaviour (old) and reference policy.
    advantages: [B] sequence-level (outcome supervision) or [B, T] token-level (process sup.).
    mask: [B, T] 1 for completion tokens, 0 for padding.
    loss_agg: "seq-mean-token-mean" = original GRPO (1/G sum_i 1/|o_i| sum_t),
              "token-mean" = DAPO-style (sum over all tokens / number of tokens).
    kl_is_correction: multiply k3 by pi_theta/pi_old (DeepSeek-V3.2's 'unbiased KL estimate'
              when samples come from pi_old rather than pi_theta).
    """
    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(1).expand_as(logp)
    ratio = torch.exp(logp - logp_old)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    pg = -torch.min(surr1, surr2)                       # [B, T]
    kl = k3_kl(logp, logp_ref)
    if kl_is_correction:
        kl = ratio * kl
    per_token = pg + beta * kl                          # KL is *in the loss*, not the reward

    if loss_agg == "seq-mean-token-mean":
        loss = masked_mean(per_token, mask, dim=1).mean()
    elif loss_agg == "token-mean":
        loss = masked_mean(per_token, mask)
    else:
        raise ValueError(loss_agg)
    with torch.no_grad():
        clipped = ((ratio - 1.0).abs() > clip_eps).float()
        stats = {"clip_frac": masked_mean(clipped, mask).item(),
                 "kl": masked_mean(kl, mask).item()}
    return loss, stats


def off_policy_sequence_mask(logp: torch.Tensor, logp_old: torch.Tensor, advantages: torch.Tensor,
                             mask: torch.Tensor, delta: float = 0.5) -> torch.Tensor:
    """DeepSeek-V3.2-style 'off-policy sequence masking' (sketch): drop sequences that have a
    NEGATIVE advantage AND whose mean log(pi_old / pi_theta) exceeds delta (too off-policy).
    Returns a [B, 1] multiplier to apply to the token mask."""
    seq_kl = masked_mean(logp_old - logp, mask, dim=1)         # [B]
    drop = (advantages < 0) & (seq_kl > delta)
    return (~drop).float().unsqueeze(1)


# ----------------------------------------------------------------------------------------
# (b) Tiny toy LM + verifiable task
# ----------------------------------------------------------------------------------------
VOCAB, LENGTH, MAX_TARGET = 4, 4, 12        # digits 0..3, 4 tokens, target in 0..12


class TinyGRUPolicy(nn.Module):
    """Autoregressive policy pi(o_t | prompt, o_<t) with a GRU cell; prompt = target id."""

    def __init__(self, n_prompts, vocab, hidden=64):
        super().__init__()
        self.V = vocab
        self.prompt_emb = nn.Embedding(n_prompts, hidden)
        self.tok_emb = nn.Embedding(vocab + 1, hidden)      # index `vocab` = BOS
        self.gru = nn.GRUCell(hidden, hidden)
        self.head = nn.Linear(hidden, vocab)

    def forward(self, prompts, tokens):
        """Teacher-forced logits [B, T, V] for predicting tokens[:, t]."""
        B, T = tokens.shape
        h = torch.tanh(self.prompt_emb(prompts))
        inp = torch.full((B,), self.V, dtype=torch.long)
        out = []
        for t in range(T):
            h = self.gru(self.tok_emb(inp), h)
            out.append(self.head(h))
            inp = tokens[:, t]
        return torch.stack(out, 1)

    def token_logp(self, prompts, tokens):
        logits = self.forward(prompts, tokens)
        return torch.log_softmax(logits, -1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)

    @torch.no_grad()
    def sample(self, prompts, T, greedy=False):
        B = prompts.shape[0]
        h = torch.tanh(self.prompt_emb(prompts))
        inp = torch.full((B,), self.V, dtype=torch.long)
        toks = []
        for _ in range(T):
            h = self.gru(self.tok_emb(inp), h)
            logits = self.head(h)
            inp = logits.argmax(-1) if greedy else torch.distributions.Categorical(logits=logits).sample()
            toks.append(inp)
        return torch.stack(toks, 1)


def reward_fn(prompts, tokens):
    """Verifiable reward: 1 if digits sum to the target, else 0."""
    return (tokens.sum(1) == prompts).float()


def train(steps=250, n_prompts_per_step=32, G=8, mu=2, lr=3e-3, beta=0.04):
    policy = TinyGRUPolicy(MAX_TARGET + 1, VOCAB)
    ref = copy.deepcopy(policy).eval()                  # frozen reference = initial policy
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    eval_prompts = torch.arange(MAX_TARGET + 1)

    for step in range(steps + 1):
        # 1) sample a batch of prompts, G completions each (contiguous groups)
        prompts = torch.randint(0, MAX_TARGET + 1, (n_prompts_per_step,)).repeat_interleave(G)
        tokens = policy.sample(prompts, LENGTH)
        mask = torch.ones_like(tokens, dtype=torch.float)
        rewards = reward_fn(prompts, tokens)
        adv = group_relative_advantages(rewards, G)
        with torch.no_grad():
            logp_old = policy.token_logp(prompts, tokens)
            logp_ref = ref.token_logp(prompts, tokens)
        # 2) mu gradient steps on the same batch (old policy fixed -> ratio != 1 after 1st step)
        for _ in range(mu):
            logp = policy.token_logp(prompts, tokens)
            loss, stats = grpo_loss(logp, logp_old, logp_ref, adv, mask, beta=beta)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
        if step % 50 == 0:
            with torch.no_grad():
                greedy = policy.sample(eval_prompts, LENGTH, greedy=True)
                acc = reward_fn(eval_prompts, greedy).mean().item()
            print(f"step {step:4d} | train reward {rewards.mean():.3f} | greedy acc over all "
                  f"targets {acc:.3f} | kl {stats['kl']:.4f} | clip_frac {stats['clip_frac']:.3f}")
    return policy


if __name__ == "__main__":
    # sanity checks of the tensor functions
    r = torch.tensor([1., 0., 0., 0., 1., 1., 1., 1.])
    print("group adv (G=4):", group_relative_advantages(r, 4).round(decimals=3).tolist())
    lp = torch.log(torch.tensor([[0.5, 0.25]]))
    print("k3 KL when equal policies (should be 0):", k3_kl(lp, lp).tolist())
    policy = train()
    for tgt in [0, 3, 6, 9, 12]:
        out = policy.sample(torch.tensor([tgt]), LENGTH, greedy=True)[0].tolist()
        print(f"target {tgt:2d} -> {out} (sum {sum(out)})")
