"""RLVR (RL with Verifiable Rewards) -- a small verifier library + a toy RLVR loop.

(a) Verifiers: math answer extraction/normalisation/equivalence, an R1-style format reward,
    and a sandboxed-ish unit-test runner for code (subprocess + timeout).
(b) Toy RLVR loop: a tiny GRU policy writes text like "<answer>7</answer>" for prompts "a+b".
    The *text* is scored by the verifier (accuracy + format), exactly like real RLVR, and the
    policy is trained with a group-mean baseline (GRPO-lite). We also report pass@k.

Run:  python3 code/rlvr.py      (CPU, ~20-40 s)
"""
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from fractions import Fraction

import torch
import torch.nn as nn

torch.manual_seed(0)
torch.set_num_threads(1)


# ----------------------------------------------------------------------------------------
# (a) Verifier library
# ----------------------------------------------------------------------------------------
def extract_boxed(text: str):
    """Return the content of the LAST \\boxed{...} (handles nested braces), or None."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    i = text.find("{", idx)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
    return None  # unbalanced braces -> treat as no answer


def extract_answer(text: str):
    """Priority: <answer>...</answer>  >  \\boxed{}  >  'the answer is X'  >  last number."""
    m = re.findall(r"<answer>(.*?)</answer>", text, flags=re.S)
    if m:
        inner = m[-1]
        return extract_boxed(inner) or inner.strip()
    b = extract_boxed(text)
    if b is not None:
        return b
    m = re.findall(r"answer is[:\s]*\$?([^\s$]+)", text, flags=re.I)
    if m:
        return m[-1].rstrip(".")
    nums = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", text)
    return nums[-1] if nums else None


def normalize_answer(s: str) -> str:
    """Light LaTeX/string normalisation (real systems, e.g. math-verify, do much more)."""
    s = s.strip().strip("$").strip()
    s = re.sub(r"\\text\{(.*?)\}", r"\1", s)
    s = s.replace("\\left", "").replace("\\right", "").replace("\\!", "").replace("\\,", "")
    s = s.replace("dfrac", "frac").replace("tfrac", "frac")
    s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)   # \frac{a}{b} -> (a)/(b)
    s = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", s)
    s = s.replace("^\\circ", "").replace("\\%", "").replace("%", "")
    s = re.sub(r"(\d),(\d{3})", r"\1\2", s)                          # 1,000 -> 1000
    s = s.replace(" ", "").rstrip(".")
    if re.fullmatch(r"\(?-?\d+\)?/\(?\d+\)?", s):                    # (3)/(4) -> 3/4
        s = s.replace("(", "").replace(")", "")
    return s


def _to_number(s: str):
    try:
        return Fraction(s)          # handles "3/4", "-2", "0.5"
    except (ValueError, ZeroDivisionError):
        return None


def math_equal(pred, gt, rel_tol=1e-6) -> bool:
    """Is the predicted answer equivalent to the ground truth?"""
    if pred is None:
        return False
    p, g = normalize_answer(str(pred)), normalize_answer(str(gt))
    if p == g:
        return True
    pn, gn = _to_number(p), _to_number(g)
    if pn is not None and gn is not None:
        return math.isclose(float(pn), float(gn), rel_tol=rel_tol, abs_tol=1e-9)
    return False


FORMAT_RE = re.compile(r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$", re.S)


def format_reward(text: str) -> float:
    """DeepSeek-R1-Zero style: reasoning in <think>..</think>, final answer in <answer>..</answer>."""
    return 1.0 if FORMAT_RE.match(text) else 0.0


def math_reward(completion: str, gt: str, format_weight: float = 0.1) -> float:
    acc = 1.0 if math_equal(extract_answer(completion), gt) else 0.0
    return acc + format_weight * format_reward(completion)


_TEST_HARNESS = r'''
import json, sys
passed = 0
tests = json.loads(sys.argv[1])
ns = {}
try:
    exec(open(sys.argv[2]).read(), ns)
except Exception:
    print(json.dumps({"passed": 0, "total": len(tests)})); sys.exit(0)
for t in tests:
    try:
        exec(t, ns); passed += 1
    except Exception:
        pass
print(json.dumps({"passed": passed, "total": len(tests)}))
'''


def run_unit_tests(code: str, tests: list, timeout: float = 5.0) -> float:
    """Execute candidate code + assert-style tests in a separate Python process with a timeout.
    Returns the fraction of tests passed. NOTE: a subprocess is NOT a security sandbox -- real
    RLVR systems run untrusted code in containers / gVisor / firejail with no network."""
    with tempfile.TemporaryDirectory() as d:
        src, harness = os.path.join(d, "cand.py"), os.path.join(d, "harness.py")
        open(src, "w").write(code)
        open(harness, "w").write(_TEST_HARNESS)
        try:
            out = subprocess.run([sys.executable, "-I", harness, json.dumps(tests), src],
                                 capture_output=True, text=True, timeout=timeout, cwd=d)
            res = json.loads(out.stdout.strip().splitlines()[-1])
            return res["passed"] / max(res["total"], 1)
        except (subprocess.TimeoutExpired, json.JSONDecodeError, IndexError):
            return 0.0


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al., 2021): 1 - C(n-c, k) / C(n, k)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


# ----------------------------------------------------------------------------------------
# (b) Toy RLVR loop: the policy writes TEXT that the verifier scores
# ----------------------------------------------------------------------------------------
TOKENS = [str(d) for d in range(10)] + ["<answer>", "</answer>", "<eos>", "hmm"]
EOS = TOKENS.index("<eos>")
MAX_LEN, N_A = 5, 5                      # prompts "a+b" with a, b in 0..4
PROMPTS = [(a, b) for a in range(N_A) for b in range(N_A)]


class TinyGRUPolicy(nn.Module):
    def __init__(self, n_prompts, vocab, hidden=64):
        super().__init__()
        self.V = vocab
        self.prompt_emb = nn.Embedding(n_prompts, hidden)
        self.tok_emb = nn.Embedding(vocab + 1, hidden)
        self.gru = nn.GRUCell(hidden, hidden)
        self.head = nn.Linear(hidden, vocab)

    def forward(self, prompts, tokens):
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
        return torch.log_softmax(self(prompts, tokens), -1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)

    @torch.no_grad()
    def sample(self, prompts, T):
        """Sample until <eos> (or T tokens). Returns tokens [B,T] and mask [B,T]."""
        B = prompts.shape[0]
        h = torch.tanh(self.prompt_emb(prompts))
        inp = torch.full((B,), self.V, dtype=torch.long)
        toks, masks = [], []
        alive = torch.ones(B, dtype=torch.bool)
        for _ in range(T):
            h = self.gru(self.tok_emb(inp), h)
            inp = torch.distributions.Categorical(logits=self.head(h)).sample()
            inp = torch.where(alive, inp, torch.full_like(inp, EOS))   # pad with <eos>
            toks.append(inp)
            masks.append(alive.float())
            alive = alive & (inp != EOS)
        return torch.stack(toks, 1), torch.stack(masks, 1)


def detokenize(tok_row, mask_row):
    out = []
    for t, m in zip(tok_row.tolist(), mask_row.tolist()):
        if m == 0 or t == EOS:
            break
        out.append(TOKENS[t])
    return "".join(out)


def toy_reward(text: str, gt: int) -> float:
    """Accuracy via the math verifier + a small format bonus for '<answer>X</answer>'."""
    fmt = 1.0 if re.fullmatch(r"<answer>[^<>]*</answer>", text) else 0.0
    m = re.findall(r"<answer>(.*?)</answer>", text)
    acc = 1.0 if (m and math_equal(m[-1], str(gt))) else 0.0
    return acc + 0.2 * fmt


def evaluate(policy, n=16, k=8):
    """pass@1 and pass@k (unbiased) over all prompts with n samples each."""
    p1, pk = [], []
    for pid, (a, b) in enumerate(PROMPTS):
        prompts = torch.full((n,), pid)
        toks, mask = policy.sample(prompts, MAX_LEN)
        c = sum(toy_reward(detokenize(toks[i], mask[i]), a + b) >= 1.0 for i in range(n))
        p1.append(pass_at_k(n, c, 1))
        pk.append(pass_at_k(n, c, k))
    return sum(p1) / len(p1), sum(pk) / len(pk)


def train_rlvr(steps=600, n_prompts=16, G=8, lr=3e-3):
    policy = TinyGRUPolicy(len(PROMPTS), len(TOKENS))
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    print("before RL: pass@1 = %.3f, pass@8 = %.3f" % evaluate(policy))
    for step in range(steps + 1):
        pids = torch.randint(0, len(PROMPTS), (n_prompts,)).repeat_interleave(G)
        toks, mask = policy.sample(pids, MAX_LEN)
        texts = [detokenize(toks[i], mask[i]) for i in range(len(pids))]
        rewards = torch.tensor([toy_reward(t, sum(PROMPTS[p])) for t, p in zip(texts, pids.tolist())])
        r = rewards.view(-1, G)
        adv = ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-6)).view(-1, 1)
        logp = policy.token_logp(pids, toks)
        loss = -((logp * adv * mask).sum(1) / mask.sum(1)).mean()   # on-policy, 1 step/batch
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 150 == 0:
            a, b = PROMPTS[pids[0].item()]
            print(f"step {step:3d} | mean reward {rewards.mean():.3f} | sample for '{a}+{b}': {texts[0]!r}")
    print("after  RL: pass@1 = %.3f, pass@8 = %.3f" % evaluate(policy))
    return policy


if __name__ == "__main__":
    # --- verifier unit checks ---
    assert extract_boxed(r"so \boxed{\frac{3}{4}} done") == r"\frac{3}{4}"
    assert math_equal(r"\frac{3}{4}", "0.75") and math_equal(r"\dfrac{3}{4}", "3/4")
    assert math_equal("1,000", "1000") and math_equal(r"\text{5}", "5") and not math_equal("5", "6")
    assert extract_answer("The answer is 42.") == "42"
    good = "<think>2+2 is 4</think> <answer>\\boxed{4}</answer>"
    assert math_reward(good, "4") == 1.1 and math_reward("<answer>5</answer>", "4") == 0.0
    code_ok = "def add(a, b):\n    return a + b\n"
    code_bad = "def add(a, b):\n    return a - b\n"
    code_hang = "def add(a, b):\n    while True: pass\n"
    tests = ["assert add(1, 2) == 3", "assert add(-1, 1) == 0", "assert add(0, 0) == 0"]
    print("unit tests: correct=%.2f wrong=%.2f infinite-loop=%.2f" % (
        run_unit_tests(code_ok, tests), run_unit_tests(code_bad, tests),
        run_unit_tests(code_hang, ["assert add(1,2)==3"], timeout=2.0)))
    print("pass@k(n=10, c=2, k=5) = %.3f" % pass_at_k(10, 2, 5))
    print("verifier checks passed\n")
    train_rlvr()
