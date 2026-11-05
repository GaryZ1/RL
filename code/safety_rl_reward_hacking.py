"""
Safety in RL fine-tuning: (A) Safe RLHF-style Lagrangian constrained optimization and
(B) a reward-hacking demonstration with a flawed grader and mitigations.

Part A -- Safe RLHF (Dai et al. 2023) in miniature
  A reward model R (helpfulness) and a cost model C (harmfulness) score each response.
  On harmful requests the most "helpful" answer is the harmful one, so maximizing R alone
  produces harm. We solve   max E[R]  s.t.  E[C] <= d   with a Lagrange multiplier lambda
  updated by dual ascent; the policy uses the combined advantage (A_R - lambda*A_C)/(1+lambda).

Part B -- reward hacking in a toy "coding" environment
  The policy writes a 6-token "program". True quality = fraction of 6 unit tests passed
  (position i must hold token i+1), so an exit token always costs real progress. The FLAWED grader also reports success
  whenever the program contains an "early exit" token (9 = sys.exit(0), 8 = os._exit(0)),
  mimicking hacks such as exiting before assertions run.
  Compare: flawed grader | flawed grader + KL | flawed grader + monitor penalty that only
  recognizes token 9 | fixed grader.

Run: python3 code/safety_rl_reward_hacking.py     (CPU, ~15 s)
"""
import torch
import torch.nn as nn

torch.manual_seed(0)
torch.set_num_threads(1)

# ----------------------------------------------------------------------------------
# Shared: tabular position-wise softmax policy and an RLOO-style policy-gradient loss
# ----------------------------------------------------------------------------------

class TablePolicy(nn.Module):
    def __init__(self, P, T, V, init=None):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(P, T, V) if init is None else init.clone())

    def dist(self, p):
        return torch.distributions.Categorical(logits=self.logits[p])

    def sample(self, p):
        with torch.no_grad():
            return self.dist(p).sample()

    def seq_logp(self, p, y):
        return self.dist(p).log_prob(y).sum(-1)


def rloo_advantage(x, K):
    x = x.view(-1, K)
    return (x - (x.sum(1, keepdim=True) - x) / (K - 1)).view(-1)


def safe_rlhf_policy_loss(logp, adv_reward, adv_cost, lam):
    """Safe RLHF-style surrogate: maximize (A_R - lambda * A_C) / (1 + lambda)."""
    adv = (adv_reward - lam * adv_cost) / (1 + lam)
    return -(adv.detach() * logp).mean()


def lagrange_update(lam, mean_cost, limit, lr):
    """Dual ascent on lambda >= 0: raise lambda while the constraint is violated."""
    return max(0.0, lam + lr * (mean_cost - limit))

# ----------------------------------------------------------------------------------
# Part A
# ----------------------------------------------------------------------------------
PA, TA, VA = 4, 5, 10        # prompts 0,1 benign; 2,3 harmful requests
REFUSE, SAFE_HELP, HARM = 7, 8, 9
HARMFUL = torch.tensor([False, False, True, True])


def reward_model(p, y):
    """Helpfulness RM: on benign prompts the on-topic token p is worth 1; on harmful prompts
    the harmful detail is (sadly) the most 'helpful' (1.0), a safe partial answer is worth 0.5."""
    on_topic = (y == p[:, None]).float().sum(1) * (~HARMFUL[p]).float()
    harm = (y == HARM).float().sum(1)
    safe = (y == SAFE_HELP).float().sum(1)
    h = HARMFUL[p].float()
    return (on_topic + h * (1.0 * harm + 0.5 * safe)) / TA


def cost_model(p, y):
    """Harmfulness cost: number of harmful tokens (a trained classifier in practice)."""
    return (y == HARM).float().sum(1)


def run_safe_rlhf(mode, steps=400, limit=0.1, K=4):
    torch.manual_seed(2)
    pol = TablePolicy(PA, TA, VA)
    opt = torch.optim.Adam(pol.parameters(), lr=0.05)
    lam = 0.0 if mode != "fixed-penalty" else 5.0
    lam_max = lam
    for _ in range(steps):
        p = torch.randint(0, PA, (64,)).repeat_interleave(K)
        y = pol.sample(p)
        R, C = reward_model(p, y), cost_model(p, y)
        loss = safe_rlhf_policy_loss(pol.seq_logp(p, y), rloo_advantage(R, K),
                                     rloo_advantage(C, K), lam if mode != "reward-only" else 0.0)
        opt.zero_grad(); loss.backward(); opt.step()
        if mode == "lagrangian":
            lam = lagrange_update(lam, C[HARMFUL[p]].mean().item(), limit, lr=0.05)
            lam_max = max(lam_max, lam)
    p = torch.randint(0, PA, (4000,))
    y = pol.sample(p)
    h = HARMFUL[p]
    return (reward_model(p, y).mean().item(), cost_model(p[h], y[h]).mean().item(),
            (y[h] == SAFE_HELP).float().mean().item(), lam, lam_max)


def part_a():
    print("Part A: Safe RLHF (constraint: E[cost | harmful prompt] <= 0.1)")
    print(f"{'mode':<15}{'E[reward]':>10}{'E[cost]':>9}{'safe-help rate':>16}{'final lam':>11}{'max lam':>9}")
    for mode in ["reward-only", "fixed-penalty", "lagrangian"]:
        r, c, s, lam, lam_max = run_safe_rlhf(mode)
        print(f"{mode:<15}{r:>10.3f}{c:>9.3f}{s:>16.3f}{lam:>11.2f}{lam_max:>9.2f}")

# ----------------------------------------------------------------------------------
# Part B
# ----------------------------------------------------------------------------------
TB, VB = 6, 10
SOLUTION = torch.tensor([1, 2, 3, 4, 5, 6])
EXIT_TOKENS = torch.tensor([8, 9])     # both make the flawed harness report success


def tests_passed(y):
    return (y == SOLUTION).float().mean(1)                   # true quality in [0, 1]


def has_hack(y):
    return torch.isin(y, EXIT_TOKENS).any(1)


def flawed_grader(y):
    return torch.where(has_hack(y), torch.ones(y.size(0)), tests_passed(y))


def monitor_flags(y):
    """A monitor that knows about sys.exit (token 9) but not os._exit (token 8)."""
    return (y == 9).any(1)


def run_hacking(setting, steps=300, K=8, beta_kl=0.0):
    torch.manual_seed(3)
    init = torch.zeros(1, TB, VB)
    init[..., 8:] = -2.5                                      # exit tokens rare under the SFT model
    pol, ref = TablePolicy(1, TB, VB, init), TablePolicy(1, TB, VB, init)
    opt = torch.optim.Adam(pol.parameters(), lr=0.05)
    for _ in range(steps):
        p = torch.zeros(32 * K, dtype=torch.long)
        y = pol.sample(p)
        if setting == "fixed grader":
            r = tests_passed(y)
        else:
            r = flawed_grader(y)
            if setting == "flawed + monitor penalty":
                r = torch.where(monitor_flags(y), torch.full_like(r, -1.0), r)
        logp = pol.seq_logp(p, y)
        with torch.no_grad():
            r = r - beta_kl * (logp - ref.seq_logp(p, y))
        loss = -(rloo_advantage(r, K) * logp).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    y = pol.sample(torch.zeros(4000, dtype=torch.long))
    return (flawed_grader(y).mean().item(), tests_passed(y).mean().item(),
            (y == 9).any(1).float().mean().item(), (y == 8).any(1).float().mean().item())


def part_b():
    print("\nPart B: reward hacking with a flawed unit-test grader")
    print(f"{'setting':<27}{'grader score':>13}{'TRUE pass':>11}{'hack9 rate':>12}{'hack8 rate':>12}")
    for setting, beta in [("flawed grader", 0.0), ("flawed grader + KL 0.1", 0.1),
                          ("flawed + monitor penalty", 0.0), ("fixed grader", 0.0)]:
        g, t, h9, h8 = run_hacking(setting.replace(" + KL 0.1", ""), beta_kl=beta)
        print(f"{setting:<27}{g:>13.3f}{t:>11.3f}{h9:>12.3f}{h8:>12.3f}")
    print("Lesson: penalizing only the hacks a monitor can see pushes the policy to the hack it\n"
          "cannot see; fixing the environment removes the incentive altogether.")


if __name__ == "__main__":
    part_a()
    part_b()
