"""
demo_concepts.py - STRATUM's three core ideas in pure numpy. No GPU, no downloads.

    pip install numpy
    python scripts/demo_concepts.py

Run before reading the docs, and again after. Every number the docs quote is
produced here.
"""
import numpy as np

np.random.seed(0)
LINE = "=" * 62


def newton_schulz(G, steps=5):
    """Muon's core: flatten a matrix's singular values toward 1."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.astype(np.float64) / (np.linalg.norm(G) + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X


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

print(LINE)
print("Three ideas: balanced updates (Muon), low-rank tiles (LoRA),")
print("additive fusing (merge). That's the whole project.")
print(LINE)
