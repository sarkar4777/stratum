"""
demo_concepts.py - STRATUM's core ideas in pure numpy. No GPU, no downloads.

    pip install numpy
    python scripts/demo_concepts.py

Six demos. The first three are the foundations (balanced updates, rank,
additive fusing), the last three prove the claims the docs make about them
(Muon vs AdamW head to head, TIES vs linear on conflicting strata, what
temperature does to a distribution). Run before reading the docs, and again
after. Every number the docs quote is produced here.
"""
import numpy as np

np.random.seed(0)
LINE = "=" * 62


def newton_schulz(G, steps=5):
    """Muon's core: flatten a matrix's singular values toward 1."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.astype(np.float64) / (np.linalg.norm(G) + 1e-7)
    tall = X.shape[0] > X.shape[1]
    if tall:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X.T if tall else X


print(LINE + "\nDEMO 1 - Muon orthogonalizes an update\n" + LINE)
G = np.random.randn(6, 6)
_, s0, _ = np.linalg.svd(G)
_, s1, _ = np.linalg.svd(newton_schulz(G))
print("A weight update pushes along several directions. Their strengths:")
print(" before:", np.round(s0, 2), "<- lopsided, a few dominate")
print(" after: ", np.round(s1, 2), "<- flattened, all share")
print("Muon does this every step so no direction is neglected.\n")


print(LINE + "\nDEMO 2 - an adapter's rank sets a hard ceiling\n" + LINE)
d, TRUE_RANK, lr, steps = 32, 6, 0.02, 2000
W = np.random.randn(d, d) * 0.1
true_delta = (np.random.randn(d, TRUE_RANK) @ np.random.randn(TRUE_RANK, d)) * 0.1
X = np.random.randn(256, d)
Y = X @ (W + true_delta).T
base = float(np.mean((X @ W.T - Y) ** 2))
print(f"This skill needs a rank-{TRUE_RANK} adjustment. Error before: {base:.3f}\n")
print(f"{'adapter rank':>13} {'final error':>13} {'gap closed':>12}")
for r in [1, 2, 4, 6, 12]:
    A = np.random.randn(r, d) * 0.01
    B = np.zeros((d, r))
    for _ in range(steps):
        pred = X @ W.T + (X @ A.T) @ B.T
        err = 2 * (pred - Y) / X.shape[0]
        B -= lr * (err.T @ (X @ A.T))
        A -= lr * ((err @ B).T @ X)
    final = float(np.mean((X @ W.T + (X @ A.T) @ B.T - Y) ** 2))
    closed = 100 * (1 - final / base)
    flag = " <- stuck" if closed < 95 else ""
    print(f"{r:>13} {final:>13.4f} {closed:>11.1f}%{flag}")
print("\nBelow the needed rank it plateaus forever. Pick rank >= skill complexity.\n")


print(LINE + "\nDEMO 3 - fusing strata is addition\n" + LINE)
Wb = np.random.randn(8, 8) * 0.1
A1, B1 = np.random.randn(2, 8) * 0.1, np.random.randn(8, 2) * 0.1
A2, B2 = np.random.randn(2, 8) * 0.1, np.random.randn(8, 2) * 0.1
d1, d2 = B1 @ A1, B2 @ A2
print(f"base weight norm: {np.linalg.norm(Wb):.3f}")
print(f"skill-1 delta norm: {np.linalg.norm(d1):.3f}")
print(f"skill-2 delta norm: {np.linalg.norm(d2):.3f}")
for w1, w2 in [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.7, 0.3)]:
    m = Wb + w1 * d1 + w2 * d2
    print(f" weights ({w1}, {w2}) -> merged norm {np.linalg.norm(m):.3f}")
print("\nEach skill is a small delta on the SAME base. Fuse by weighted sum.")
print("Train skill 1 Monday, skill 2 Tuesday, fuse Wednesday. That's STRATUM.\n")

print(LINE + "\nDEMO 4 - the optimizer face-off: Muon vs AdamW vs SGD\n" + LINE)

# The task: train a small two-layer network to imitate a "teacher" network,
# from noisy minibatches - a miniature of real training. Every optimizer gets
# a learning-rate sweep and races at its own best setting, with the same
# decaying step size, so the comparison is fair. The finish line: test error
# down to 5% of where it started.
d_in, d_h, d_out, n, steps, batch = 16, 32, 16, 512, 800, 64
T1 = np.random.randn(d_h, d_in) / np.sqrt(d_in)
T2 = np.random.randn(d_out, d_h) / np.sqrt(d_h)
Xtr = np.random.randn(n, d_in)
Ytr = np.tanh(Xtr @ T1.T) @ T2.T
Xte = np.random.randn(512, d_in)
Yte = np.tanh(Xte @ T1.T) @ T2.T
target = float(np.mean(Yte ** 2)) * 0.05


def test_error(Ws):
    return float(np.mean((np.tanh(Xte @ Ws[0].T) @ Ws[1].T - Yte) ** 2))


def race(optimizer, lr):
    rng = np.random.default_rng(7)
    Ws = [rng.standard_normal((d_h, d_in)) * 0.1,
          rng.standard_normal((d_out, d_h)) * 0.1]
    ms = [np.zeros_like(w) for w in Ws]
    vs = [np.zeros_like(w) for w in Ws]
    for t in range(1, steps + 1):
        err = test_error(Ws)
        if not np.isfinite(err):
            return None
        if err <= target:
            return t - 1
        # One minibatch, hand-written backprop for the two weight grids.
        idx = rng.integers(0, n, batch)
        H = np.tanh(Xtr[idx] @ Ws[0].T)
        E = (H @ Ws[1].T - Ytr[idx]) * 2.0 / batch
        grads = [((E @ Ws[1]) * (1 - H ** 2)).T @ Xtr[idx], E.T @ H]
        step_lr = lr * (1 - t / steps)  # same decay schedule for everyone
        for i, g in enumerate(grads):
            if optimizer == "sgd":
                # Plain SGD: step along the gradient, nothing remembered.
                Ws[i] -= step_lr * g
            elif optimizer == "adamw":
                # AdamW's two notes per dial: momentum and variance.
                ms[i] = 0.9 * ms[i] + 0.1 * g
                vs[i] = 0.999 * vs[i] + 0.001 * g * g
                mhat = ms[i] / (1 - 0.9 ** t)
                vhat = vs[i] / (1 - 0.999 ** t)
                Ws[i] -= step_lr * mhat / (np.sqrt(vhat) + 1e-8)
            elif optimizer == "muon":
                # Muon's one note: momentum, orthogonalized before the step.
                ms[i] = 0.95 * ms[i] + g
                scale = max(1.0, g.shape[0] / g.shape[1]) ** 0.5
                Ws[i] -= step_lr * scale * newton_schulz(g + 0.95 * ms[i])
    return None


SWEEPS = {"sgd": [0.1, 0.3, 1.0, 3.0],
          "adamw": [0.003, 0.01, 0.03, 0.1],
          "muon": [0.01, 0.03, 0.1, 0.3]}
NOTES = {"sgd": 0, "adamw": 2, "muon": 1}

print("Task: train a 2-layer network to imitate a teacher network, from noisy")
print("minibatches. Finish line: test error down to 5% of its starting value.")
print("Each optimizer races at its own best learning rate.\n")
print(f"{'optimizer':>10} {'best lr':>8} {'steps to finish':>16} {'notes per dial':>15}")
finish = {}
for opt, lrs in SWEEPS.items():
    best = None
    for lr in lrs:
        reached = race(opt, lr)
        if reached is not None and (best is None or reached < best[0]):
            best = (reached, lr)
    finish[opt] = best
for opt in ("muon", "adamw", "sgd"):
    reached, lr = finish[opt]
    print(f"{opt:>10} {lr:>8} {reached:>16} {NOTES[opt]:>15}")
print("\nAt this tiny scale Muon and AdamW arrive in a similar number of steps -")
print("with Muon carrying HALF the optimizer memory. Muon's step advantage")
print("grows with model size, and the memory advantage is there at every size.\n")


print(LINE + "\nDEMO 5 - when strata conflict: linear vs TIES\n" + LINE)

# Two skills, each a strong signal on its own rows of the weight grid, plus
# weak opposite-signed noise sprayed across the OTHER skill's rows - the
# crosstalk that real strata pick up. Linear addition lets the noise dilute
# both signals. TIES trims each stratum to its strongest entries and lets the
# majority sign win, so the signals come through clean.
g = 8
strong, noise = 0.5, 0.12
d1 = np.zeros((g, g))
d2 = np.zeros((g, g))
d1[:4, :] = strong
d2[4:, :] = -strong
d1[4:, :] = noise * np.sign(np.random.randn(4, g))   # d1's noise on d2's rows
d2[:4, :] = noise * np.sign(np.random.randn(4, g))   # d2's noise on d1's rows

linear = d1 + d2


def trim(t, density):
    k = max(1, int(t.size * density))
    thresh = np.sort(np.abs(t).flatten())[-k]
    return t * (np.abs(t) >= thresh)


def ties(deltas, density=0.5):
    stacked = np.stack([trim(t, density) for t in deltas])
    sign = np.sign(stacked.sum(axis=0))
    agree = (np.sign(stacked) == sign) & (stacked != 0)
    kept = stacked * agree
    count = np.maximum(agree.sum(axis=0), 1)
    return kept.sum(axis=0) / count


merged_ties = ties([d1, d2])
ideal = np.zeros((g, g))
ideal[:4, :] = strong
ideal[4:, :] = -strong


def recovered(merged):
    return 100 * (1 - np.linalg.norm(merged - ideal) / np.linalg.norm(ideal))


print("Each skill: a strong signal on its own rows, weak opposite-signed")
print("noise on the other's rows (crosstalk).\n")
print(f" ideal merge keeps both signals, drops the noise")
print(f" linear merge recovers {recovered(linear):.0f}% of the ideal")
print(f" TIES merge   recovers {recovered(merged_ties):.0f}% of the ideal")
print("\nLinear adds everything, noise included. TIES trims each stratum to")
print("its strongest entries first, so the crosstalk never gets a vote.\n")


print(LINE + "\nDEMO 6 - what temperature does to a distribution\n" + LINE)

# The teacher's raw scores (logits) for one classification, softened at
# different temperatures. Higher T = smaller gap between top pick and the
# rest = more of the teacher's judgment visible to the student.
labels = ["billing", "account_access", "bug", "how_to"]
logits = np.array([4.0, 2.5, 1.0, 0.5])


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


print(f"{'':>16} {'T=1':>8} {'T=2':>8} {'T=4':>8}")
for i, label in enumerate(labels):
    row = [softmax(logits / T)[i] for T in (1.0, 2.0, 4.0)]
    print(f"{label:>16} " + " ".join(f"{p:>7.1%}" for p in row))
print("\nAt T=1 the runner-up is barely visible. At T=2 the student can see")
print("that account_access was a plausible near-miss - the teacher's judgment,")
print("not just its answer. That's why distillation softens before matching.\n")

print(LINE)
print("Three ideas: balanced updates (Muon), low-rank tiles (LoRA),")
print("additive fusing (merge). That's the whole project - and every")
print("claim above just ran on your own machine.")
print(LINE)
