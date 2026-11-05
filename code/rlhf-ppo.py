"""
RLHF with PPO (InstructGPT-style) -- loss functions + tiny runnable toy demo.

Pipeline:  (1) SFT on noisy demonstrations  ->  (2) reward model (here: a fixed scoring
function standing in for a trained RM; see reward-modeling.py for training one)
->  (3) PPO with a per-token KL penalty to the frozen reference (SFT) model.

Four "models" exactly as in RLHF: actor (policy), critic (value net), reward model, reference.

Toy task: the prompt is one token p in {0..V-1}; the ideal response is the "counting"
sequence p+1, p+2, ..., p+T (mod V). The reward model scores the fraction of correct tokens.

Run:  python3 code/rlhf-ppo.py      (CPU, ~20-40 s)
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_num_threads(1)  # tiny models: single thread is fastest

# ----------------------------------------------------------------------------------
# (a) Reusable RLHF-PPO pieces operating on [B, T] tensors (mirrors TRL / OpenRLHF)
# ----------------------------------------------------------------------------------

def masked_mean(x, mask, dim=None):
    if dim is None:
        return (x * mask).sum() / mask.sum().clamp(min=1)
    return (x * mask).sum(dim) / mask.sum(dim).clamp(min=1)


def masked_whiten(x, mask, shift_mean=True, eps=1e-8):
    """Normalize to zero mean / unit variance over valid (mask==1) tokens."""
    mean = masked_mean(x, mask)
    var = masked_mean((x - mean) ** 2, mask)
    out = (x - mean) * torch.rsqrt(var + eps)
    if not shift_mean:
        out = out + mean
    return out * mask


def compute_token_rewards(rm_score, logp_old, logp_ref, mask, kl_coef):
    """Token-level reward shaping used by InstructGPT / TRL PPO:
         r_t = -kl_coef * (log pi_old(y_t) - log pi_ref(y_t))      for every response token
         r_T += rm_score                                            on the LAST valid token
    rm_score: [B] sequence-level score from the reward model.
    Returns rewards [B, T] and the per-token KL estimate [B, T]."""
    kl = (logp_old - logp_ref) * mask                  # k1 estimator of KL(pi || pi_ref)
    rewards = -kl_coef * kl
    last_idx = mask.sum(dim=1).long() - 1              # index of final response token
    rewards[torch.arange(rewards.size(0)), last_idx] += rm_score
    return rewards, kl


def compute_gae(rewards, values, mask, gamma=1.0, lam=0.95):
    """Generalized Advantage Estimation over response tokens (each token = one action).
    values: [B, T] critic predictions V(s_t). The value after the last token is 0 (episode ends)."""
    B, T = rewards.shape
    adv = torch.zeros_like(rewards)
    last_gae = torch.zeros(B)
    for t in reversed(range(T)):
        next_v = values[:, t + 1] if t + 1 < T else torch.zeros(B)
        next_v = next_v * (mask[:, t + 1] if t + 1 < T else 0.0)   # 0 beyond the end
        delta = rewards[:, t] + gamma * next_v - values[:, t]
        last_gae = delta + gamma * lam * last_gae * (mask[:, t + 1] if t + 1 < T else 0.0)
        adv[:, t] = last_gae
    returns = adv + values
    return adv * mask, returns * mask


def ppo_policy_loss(logp_new, logp_old, advantages, mask, clip_eps=0.2):
    """Clipped surrogate objective, averaged over valid tokens."""
    ratio = torch.exp(logp_new - logp_old)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    loss = -masked_mean(torch.min(surr1, surr2), mask)
    clipfrac = masked_mean(((ratio - 1).abs() > clip_eps).float(), mask)
    return loss, clipfrac


def ppo_value_loss(values_new, values_old, returns, mask, clip_eps_v=0.2):
    """Clipped value loss (as in OpenAI's lm-human-preferences / TRL)."""
    v_clipped = values_old + torch.clamp(values_new - values_old, -clip_eps_v, clip_eps_v)
    l1 = (values_new - returns) ** 2
    l2 = (v_clipped - returns) ** 2
    return 0.5 * masked_mean(torch.max(l1, l2), mask)


class AdaptiveKLController:
    """Ziegler et al. (2019): adjust kl_coef so that the observed KL tracks a target."""
    def __init__(self, init_kl_coef=0.1, target=6.0, horizon=10000):
        self.value, self.target, self.horizon = init_kl_coef, target, horizon

    def update(self, current_kl, n_steps):
        err = max(min(current_kl / self.target - 1, 0.2), -0.2)
        self.value *= 1 + err * n_steps / self.horizon

# ----------------------------------------------------------------------------------
# (b) Tiny toy: GRU "language model" actor + critic, scripted reward model
# ----------------------------------------------------------------------------------
V, T, H = 10, 6, 64


class TinyLM(nn.Module):
    """Embedding -> GRU -> head. With out_dim=V it is a policy; with out_dim=1 a critic."""
    def __init__(self, out_dim=V):
        super().__init__()
        self.emb = nn.Embedding(V, H)
        self.gru = nn.GRU(H, H, batch_first=True)
        self.head = nn.Linear(H, out_dim)

    def forward(self, tokens):                # tokens [B, L] -> [B, L, out_dim]
        h, _ = self.gru(self.emb(tokens))
        return self.head(h)


def response_logps(model, prompts, responses):
    """log pi(y_t | p, y_<t) for each response token. Input = [p, y_1..y_{T-1}]."""
    inp = torch.cat([prompts[:, None], responses[:, :-1]], dim=1)
    logits = model(inp)
    return torch.log_softmax(logits, -1).gather(-1, responses[..., None]).squeeze(-1)


def critic_values(critic, prompts, responses):
    """V(s_t) where state s_t = (p, y_<t) -- the state *before* emitting token t."""
    inp = torch.cat([prompts[:, None], responses[:, :-1]], dim=1)
    return critic(inp).squeeze(-1)


@torch.no_grad()
def generate(model, prompts):
    tok, h, out = prompts[:, None], None, []
    for _ in range(T):
        x, h = model.gru(model.emb(tok), h)
        tok = torch.distributions.Categorical(logits=model.head(x[:, -1])).sample()[:, None]
        out.append(tok)
    return torch.cat(out, dim=1)


def reward_model(prompts, responses):
    """Stand-in reward model: fraction of tokens matching the counting pattern p+1, p+2, ..."""
    target = (prompts[:, None] + torch.arange(1, T + 1)[None]) % V
    return (responses == target).float().mean(dim=1)


def sft(model, steps=300, correct_prob=0.5):
    """Stage 1: supervised fine-tuning on noisy demonstrations (each token correct w.p. 0.5)."""
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(steps):
        p = torch.randint(0, V, (128,))
        target = (p[:, None] + torch.arange(1, T + 1)[None]) % V
        noise = torch.randint(0, V, target.shape)
        demo = torch.where(torch.rand(target.shape) < correct_prob, target, noise)
        loss = -response_logps(model, p, demo).mean()
        opt.zero_grad(); loss.backward(); opt.step()


def main():
    actor = TinyLM(V)
    sft(actor)
    ref = copy.deepcopy(actor).eval()                     # frozen reference = SFT model
    for prm in ref.parameters():
        prm.requires_grad_(False)
    critic = TinyLM(1)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=1e-3)
    kl_ctl = AdaptiveKLController(init_kl_coef=0.05, target=1.0, horizon=2000)
    B, ppo_epochs, n_mb = 128, 4, 4

    for it in range(1, 151):
        # ---- rollout phase (no grad) ----
        prompts = torch.randint(0, V, (B,))
        with torch.no_grad():
            responses = generate(actor, prompts)
            mask = torch.ones(B, T)
            logp_old = response_logps(actor, prompts, responses)
            logp_ref = response_logps(ref, prompts, responses)
            values_old = critic_values(critic, prompts, responses)
            score = reward_model(prompts, responses)
            # reward whitening/normalization of the RM score (common trick)
            score_n = (score - score.mean()) / (score.std() + 1e-8)
            rewards, kl = compute_token_rewards(score_n, logp_old, logp_ref, mask, kl_ctl.value)
            adv, returns = compute_gae(rewards, values_old, mask, gamma=1.0, lam=0.95)
            adv = masked_whiten(adv, mask)
        # ---- PPO update phase ----
        for _ in range(ppo_epochs):
            perm = torch.randperm(B)
            for mb in perm.chunk(n_mb):
                logp_new = response_logps(actor, prompts[mb], responses[mb])
                v_new = critic_values(critic, prompts[mb], responses[mb])
                pl, clipfrac = ppo_policy_loss(logp_new, logp_old[mb], adv[mb], mask[mb])
                vl = ppo_value_loss(v_new, values_old[mb], returns[mb], mask[mb])
                loss = pl + 0.5 * vl
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                opt.step()
        seq_kl = kl.sum(1).mean().item()
        kl_ctl.update(seq_kl, B)
        if it % 15 == 0 or it == 1:
            print(f"iter {it:3d}  RM score {score.mean():.3f}  seq-KL(pi||ref) {seq_kl:6.3f}  "
                  f"kl_coef {kl_ctl.value:.4f}  clipfrac {clipfrac:.3f}")

    with torch.no_grad():
        p = torch.randint(0, V, (1000,))
        print(f"\nfinal RM score (1000 prompts): actor {reward_model(p, generate(actor, p)).mean():.3f}"
              f"  vs  SFT/ref {reward_model(p, generate(ref, p)).mean():.3f}")
        p = torch.tensor([3])
        print("example: prompt 3 ->", generate(actor, p).tolist()[0], " (ideal [4,5,6,7,8,9])")


if __name__ == "__main__":
    main()
