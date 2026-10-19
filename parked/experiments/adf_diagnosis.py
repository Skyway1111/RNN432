"""Why does ADF fail: wrong key masses, or wrong conditional state distributions?

Finding: research/angles/adf.md
"""
import numpy as np

from az.estimators import adf
from az.model import load

np.set_printoptions(precision=2, suppress=True, linewidth=200)
w = load("example")
Wi, Wh, _ = w
m0, mu, S, p, q = adf.propagate(w, [0.0])
print(f"keys {len(m0)}, mass range {m0.min():.4f}..{m0.max():.4f} (exact 1/90 = {1 / 90:.4f})")
x = np.random.default_rng(0).standard_normal((2**21, 10))
h = np.zeros((len(x), 16))
for t in range(10):
    h = np.maximum(h @ Wh.T + x[:, t : t + 1] * Wi, 0)
srt = np.argsort(-x, 1)
for k in np.argsort(-m0)[:2]:
    sel = (srt[:, 0] == p[k]) & (srt[:, 1] == q[k])
    print(f"key ({p[k]},{q[k]}): ADF mass {m0[k]:.4f}, Monte Carlo {sel.mean():.4f}")
    print("  mean h  ADF", mu[k, :16])
    print("          MC ", h[sel].mean(0))
    print("  sd h    ADF", np.sqrt(np.diag(S[k, :16, :16])))
    print("          MC ", h[sel].std(0))
