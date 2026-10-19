"""Is the fragile direction of each W_hh row the state's dominant principal direction, so noise damage is bias sensitivity times state variance?

For each model and row n, noise of the same energy as a full-row draw (16 sigma^2) is put entirely along one direction
of the row: the top two principal directions of the state covariance sum_t E[h_t h_t^T], or W_hi (input leakage).
Compared with the full-row drop from noise_fragility_map. Also the share of the state's variance on its top directions.

Finding: research/findings/noise.md
"""
import numpy as np
import torch as th

from az.model import NAMES, W, accuracy, load, trace
from experiments.common import inputs

SIGMA, DRAWS, N = 0.002, 16, 2**20
AMP = SIGMA * 4  # one direction carrying a full row's noise energy


def drop(base, acc, n, u, seed):
    rng = np.random.default_rng(seed)
    ws = []
    for _ in range(DRAWS):
        Wh = base.Wh.copy()
        Wh[n] += AMP * rng.standard_normal() * u
        ws.append(W(base.Wi, Wh, base.Wo))
    return acc - accuracy(ws, N, seed=1).mean()


x = inputs(2**18, seed=5)
for i, name in enumerate(NAMES):
    base = load(name)
    acc = accuracy([base], N, seed=1)[0]
    h = th.relu(trace(base, x)[1]).reshape(-1, 16)
    C = (h.T @ h / len(h)).cpu().numpy()
    lam, V = np.linalg.eigh(C)
    v1, v2 = V[:, -1], V[:, -2]
    wi = base.Wi / np.linalg.norm(base.Wi)
    print(f"{name}: accuracy {acc:.4f}; state variance share on PC1 {lam[-1] / lam.sum():.3f}, PC1+2 {(lam[-1] + lam[-2]) / lam.sum():.3f}; "
          f"|PC1 . Wi| {abs(v1 @ wi):.3f}; sqrt(tr C) {lam.sum() ** 0.5:.1f}; sigma * sqrt(tr C) {SIGMA * lam.sum() ** 0.5:.3f}")
    rows = []
    for n in range(16):
        m = np.zeros((16, 16)); m[n] = 1
        rng = np.random.default_rng(100 * i + n)
        full = acc - accuracy([W(base.Wi, base.Wh + SIGMA * m * rng.standard_normal((16, 16)), base.Wo) for _ in range(DRAWS)], N, seed=1).mean()
        d = [drop(base, acc, n, u, 10_000 * i + 100 * n + k) for k, u in enumerate([v1, v2, wi])]
        rows.append((full, *d))
        print(f"  row {n:2d}: full {full:+.4f}  along PC1 {d[0]:+.4f}  PC2 {d[1]:+.4f}  Wi {d[2]:+.4f}")
    r = np.array(rows)
    print(f"  sums: full {r[:, 0].sum():.4f}  PC1 {r[:, 1].sum():.4f}  PC2 {r[:, 2].sum():.4f}  Wi {r[:, 3].sum():.4f}")
