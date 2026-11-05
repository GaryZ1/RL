"""Process reward models: Math-Shepherd-style MC step labels, PRM training, PRM-guided best-of-N,
and PRIME-style implicit PRM (process rewards learned from outcome labels only).

Toy "reasoning" task: a problem is (start s0, target T). A solution is K=6 steps, each step
adds -1, +1 or +2. The final answer s0 + sum(steps) is correct iff it equals T. A step is
"valid" if the target is still reachable afterwards (ground truth we use only for evaluation).

(a) Functions: mc_step_labels, train_prm, prm aggregation (min / product), PRM step advantages,
    implicit_prm_token_rewards, implicit_prm_ce_loss, prime_advantages.
(b) Demos: PRM accuracy vs ground-truth step validity, best-of-N selection, implicit-PRM
    credit assignment, and a short outcome-only vs PRIME-style RL comparison.

Run:  python3 code/process-reward-prime.py      (CPU, ~1 min)
"""
import copy

import torch
import torch.nn as nn

torch.manual_seed(0)
torch.set_num_threads(1)

STEPS = torch.tensor([-1, 1, 2])
K = 6                       # steps per solution
D_MIN, D_MAX = -K, 2 * K    # remaining distance T - s range


# ---------------------------- toy problem utilities ----------------------------------------
def reachable_table():
    """reach[m][d] = can a remaining distance d be covered with exactly m steps?"""
    reach = [{0}]
    for m in range(1, K + 1):
        reach.append({d + s for d in reach[m - 1] for s in STEPS.tolist()})
    return reach


REACH = reachable_table()


def feat(dist, step):
    """State features: one-hot remaining distance (T - s) and step index."""
    return torch.cat([nn.functional.one_hot(dist - D_MIN, D_MAX - D_MIN + 1),
                      nn.functional.one_hot(step, K + 1)], -1).float()


N_FEAT = (D_MAX - D_MIN + 1) + (K + 1)


def mlp(out):
    return nn.Sequential(nn.Linear(N_FEAT, 64), nn.Tanh(), nn.Linear(64, out))


def sample_problems(n):
    """Problems whose initial distance d0 = T - s0 is solvable in K steps."""
    solvable = [d for d in range(D_MIN, D_MAX + 1) if d in REACH[K]]
    return torch.tensor(solvable)[torch.randint(len(solvable), (n,))]


@torch.no_grad()
def complete(policy, dist, step):
    """Roll out the policy from (dist, step) to the end. Returns final distances, actions, feats."""
    dist = dist.clone()
    acts, feats = [], []
    for t in range(int(step[0]), K):
        f = feat(dist.clamp(D_MIN, D_MAX), torch.full_like(dist, t))
        a = torch.distributions.Categorical(logits=policy(f)).sample()
        acts.append(a)
        feats.append(f)
        dist = dist - STEPS[a]
    return dist, acts, feats


def solve(policy, d0):
    """Full solution. Returns actions [B,K], feats [B,K,F], per-step distances [B,K+1], correct [B]."""
    dist_final, acts, feats = complete(policy, d0, torch.zeros_like(d0))
    acts = torch.stack(acts, 1)
    dists = torch.cat([d0[:, None], d0[:, None] - STEPS[acts].cumsum(1)], 1)
    return acts, torch.stack(feats, 1), dists, (dist_final == 0).float()


def step_valid(dists):
    """Ground-truth validity of each step k=1..K: is the target still reachable afterwards?"""
    B = dists.shape[0]
    v = torch.zeros(B, K)
    for k in range(1, K + 1):
        v[:, k - 1] = torch.tensor([int(d) in REACH[K - k] for d in dists[:, k].tolist()]).float()
    return v


# ----------------------------------------------------------------------------------------
# (a1) Math-Shepherd: Monte-Carlo step labels  +  PRM training
# ----------------------------------------------------------------------------------------
def mc_step_labels(policy, dists, n_rollouts=8):
    """For every prefix (after step k), run n_rollouts completions with the policy.
    soft label = fraction of completions reaching the right answer; hard label = any correct."""
    B = dists.shape[0]
    soft = torch.zeros(B, K)
    for k in range(1, K + 1):
        d = dists[:, k].repeat_interleave(n_rollouts)
        if k == K:
            ok = (d == 0).float()
        else:
            final, _, _ = complete(policy, d, torch.full_like(d, k))
            ok = (final == 0).float()
        soft[:, k - 1] = ok.view(B, n_rollouts).mean(1)
    return soft, (soft > 0).float()


def train_prm(dists, labels, iters=400):
    """PRM: classifier on the prefix state after each step -> P(prefix leads to a correct answer)."""
    prm = mlp(1)
    opt = torch.optim.Adam(prm.parameters(), lr=3e-3)
    steps = torch.arange(1, K + 1).expand(dists.shape[0], K)
    X = feat(dists[:, 1:].clamp(D_MIN, D_MAX).reshape(-1), steps.reshape(-1))
    y = labels.reshape(-1)
    for _ in range(iters):
        loss = nn.functional.binary_cross_entropy_with_logits(prm(X).squeeze(-1), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return prm


@torch.no_grad()
def prm_step_scores(prm, dists):
    steps = torch.arange(1, K + 1).expand(dists.shape[0], K)
    X = feat(dists[:, 1:].clamp(D_MIN, D_MAX).reshape(-1), steps.reshape(-1))
    return torch.sigmoid(prm(X)).view(-1, K)


def aggregate_prm(step_scores, how="min"):
    """Solution score from step scores: min (weakest step) or product (Lightman et al. used the
    product of step correctness probabilities)."""
    return step_scores.min(1).values if how == "min" else step_scores.prod(1)


def prm_step_advantages(step_scores, prev_score):
    """A dense 'progress' signal for RL: r_k = PRM(prefix_k) - PRM(prefix_{k-1})."""
    s = torch.cat([prev_score[:, None], step_scores], 1)
    return s[:, 1:] - s[:, :-1]


# ----------------------------------------------------------------------------------------
# (a2) PRIME: implicit PRM from outcome labels
# ----------------------------------------------------------------------------------------
def implicit_prm_token_rewards(logp_phi, logp_ref, beta=0.5):
    """Implicit process reward of token/step t: r_t = beta * log pi_phi(y_t|y_<t) / pi_ref(y_t|y_<t).
    Their sum is the sequence-level implicit reward r_phi(y) = beta * log pi_phi(y)/pi_ref(y)."""
    return beta * (logp_phi - logp_ref)


def implicit_prm_ce_loss(logp_phi, logp_ref, outcome, mask, beta=0.5):
    """Train pi_phi with cross-entropy on OUTCOME labels: sigma(r_phi(y)) should predict correctness."""
    r_seq = (implicit_prm_token_rewards(logp_phi, logp_ref, beta) * mask).sum(1)
    return nn.functional.binary_cross_entropy_with_logits(r_seq, outcome)


def prime_advantages(outcome, proc_rewards, mask, G, proc_coef=1.0):
    """PRIME-style advantages (sketch): leave-one-out (RLOO) baseline for both the outcome reward
    and the implicit process rewards, process part turned into a return-to-go, then summed."""
    r = outcome.view(-1, G)
    a_out = (r - (r.sum(1, keepdim=True) - r) / (G - 1)).view(-1, 1) * mask
    seq_mean = (proc_rewards * mask).sum(1) / mask.sum(1)
    sm = seq_mean.view(-1, G)
    baseline = ((sm.sum(1, keepdim=True) - sm) / (G - 1)).view(-1, 1)
    proc = (proc_rewards - baseline) * mask
    a_proc = torch.flip(torch.cumsum(torch.flip(proc, [1]), 1), [1])     # return-to-go
    return a_out + proc_coef * a_proc


def logp_of(model, feats, acts):
    return torch.log_softmax(model(feats), -1).gather(-1, acts.unsqueeze(-1)).squeeze(-1)


# ----------------------------------------------------------------------------------------
# (b) Demos
# ----------------------------------------------------------------------------------------
def make_base_policy():
    """A weak 'base model': briefly trained to imitate random valid steps (so it is OK but flawed)."""
    pol = mlp(3)
    opt = torch.optim.Adam(pol.parameters(), lr=3e-3)
    for _ in range(60):
        d = torch.randint(D_MIN, D_MAX + 1, (512,))
        k = torch.randint(0, K, (512,))
        good = torch.stack([torch.tensor([int(dd - s) in REACH[K - kk - 1] for s in STEPS.tolist()])
                            for dd, kk in zip(d.tolist(), k.tolist())]).float()
        target = (good + 0.25) / (good + 0.25).sum(1, keepdim=True)   # soft, deliberately noisy
        loss = -(target * torch.log_softmax(pol(feat(d, k)), -1)).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return pol


def rl(policy0, mode, iters=80, n=64, G=8, lr=2e-3, seed=0, phi_init=None, proc_coef=0.2):
    torch.manual_seed(seed)
    pol = copy.deepcopy(policy0)
    ref = copy.deepcopy(policy0)
    # implicit PRM: initialised from the reference (PRIME) or from a warmed-up implicit PRM
    prm_phi = copy.deepcopy(phi_init if phi_init is not None else policy0)
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    opt_phi = torch.optim.Adam(prm_phi.parameters(), lr=1e-3)
    accs = []
    for _ in range(iters):
        d0 = sample_problems(n).repeat_interleave(G)
        acts, feats, dists, correct = solve(pol, d0)
        mask = torch.ones_like(acts, dtype=torch.float)
        if mode == "outcome":
            r = correct.view(-1, G)
            adv = (r - (r.sum(1, keepdim=True) - r) / (G - 1)).view(-1, 1) * mask   # RLOO
        else:
            # online implicit-PRM update on the fresh outcome labels, then dense rewards
            lp_phi = logp_of(prm_phi, feats, acts)
            with torch.no_grad():
                lp_ref = logp_of(ref, feats, acts)
            l_phi = implicit_prm_ce_loss(lp_phi, lp_ref, correct, mask)
            opt_phi.zero_grad()
            l_phi.backward()
            opt_phi.step()
            with torch.no_grad():
                proc = implicit_prm_token_rewards(logp_of(prm_phi, feats, acts), lp_ref)
            adv = prime_advantages(correct, proc, mask, G, proc_coef)
        loss = -(logp_of(pol, feats, acts) * adv).sum() / mask.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        accs.append(correct.mean().item())
    return accs


if __name__ == "__main__":
    base = make_base_policy()
    d0 = sample_problems(400)
    acts, feats, dists, correct = solve(base, d0)
    print(f"base policy accuracy: {correct.mean():.3f}")

    # --- Math-Shepherd MC labels + PRM
    soft, hard = mc_step_labels(base, dists, n_rollouts=8)
    valid = step_valid(dists)
    print(f"MC hard labels agree with ground-truth step validity on {(hard == valid).float().mean():.3f} "
          f"of steps (false 'bad' labels come from unlucky rollouts)")
    prm = train_prm(dists, hard)          # Math-Shepherd "hard estimation" labels
    test_d0 = sample_problems(300)
    _, _, tdists, tcorrect = solve(base, test_d0)
    tscores = prm_step_scores(prm, tdists)
    tvalid = step_valid(tdists)
    print(f"PRM step classification accuracy (score>0.5 vs validity) on fresh solutions: "
          f"{((tscores > 0.5).float() == tvalid).float().mean():.3f}")
    first_bad = (tvalid == 0).float().argmax(1)
    has_bad = (tvalid == 0).any(1)
    pred_bad = (tscores < 0.5).float().argmax(1)
    print(f"PRM localises the FIRST invalid step exactly in "
          f"{(pred_bad[has_bad] == first_bad[has_bad]).float().mean():.3f} of flawed solutions")

    # --- best-of-N with different selectors
    N = 8
    bo_d0 = sample_problems(300).repeat_interleave(N)
    _, _, bdists, bcorrect = solve(base, bo_d0)
    sc = prm_step_scores(prm, bdists)
    finals = (bo_d0 - bdists[:, -1]).view(-1, N)      # "answer" = distance actually covered
    bc = bcorrect.view(-1, N)
    sel = {"random pick": torch.zeros(len(bc), dtype=torch.long),
           "majority vote on answer": torch.stack([(row == row.mode().values).float().argmax()
                                                  for row in finals]),
           "PRM (min over steps)": aggregate_prm(sc, "min").view(-1, N).argmax(1),
           "PRM (product)": aggregate_prm(sc, "prod").view(-1, N).argmax(1)}
    print(f"\nbest-of-{N} accuracy (pass@{N} upper bound {bc.max(1).values.mean():.3f}):")
    for name, idx in sel.items():
        print(f"  {name:24s} {bc.gather(1, idx[:, None]).mean():.3f}")
    adv = prm_step_advantages(sc[:2], torch.full((2,), 0.5))
    print("PRM progress rewards for two solutions:", [[round(x, 2) for x in r] for r in adv.tolist()])

    # --- PRIME: implicit PRM trained only on outcome labels gives step-level credit
    ref = copy.deepcopy(base)
    phi = copy.deepcopy(base)
    opt = torch.optim.Adam(phi.parameters(), lr=1e-3)
    for _ in range(300):
        dd = sample_problems(256)
        a, f, _, c = solve(base, dd)
        with torch.no_grad():
            lr_ = logp_of(ref, f, a)
        loss = implicit_prm_ce_loss(logp_of(phi, f, a), lr_, c, torch.ones_like(lr_))
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        tok_r = implicit_prm_token_rewards(logp_of(phi, feats, acts), logp_of(ref, feats, acts))
    # step that breaks validity = first invalid step
    v = valid
    breaking = (v == 0) & torch.cat([torch.ones(v.shape[0], 1), v[:, :-1]], 1).bool()
    keeping = (v == 1)
    print(f"\nimplicit PRM (trained on outcome labels only): mean step reward on steps that keep the "
          f"target reachable {tok_r[keeping].mean():+.4f} vs. steps that break it {tok_r[breaking].mean():+.4f}")

    # --- short RL comparison
    print("\nRL from the base policy, mean of 2 seeds (train accuracy):")
    # Honest finding on this toy: the outcome reward (K=6 steps, G=8) is already informative, and the
    # online implicit PRM is noisy, so adding its dense rewards does not help and a large weight hurts.
    runs = {"outcome-only RLOO": dict(mode="outcome"),
            "PRIME-style, proc_coef 0.05": dict(mode="prime", proc_coef=0.05),
            "PRIME-style, proc_coef 0.2": dict(mode="prime", proc_coef=0.2),
            "PRIME-style, warmed-up phi, proc_coef 0.2": dict(mode="prime", phi_init=phi)}
    for name, kw in runs.items():
        curves = [rl(base, seed=s, **kw) for s in range(2)]
        print(f"  {name:44s} first10 {sum(sum(c[:10]) / 10 for c in curves) / 2:.3f} | "
              f"last10 {sum(sum(c[-10:]) / 10 for c in curves) / 2:.3f}")
