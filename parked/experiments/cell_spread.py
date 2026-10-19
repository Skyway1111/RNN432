"""Is gauss_margin's grid error the spread of the margin's mean within a cell, and does adding it to the variance remove it?

Phi(-mu / sd) at a cell's mass-weighted node ignores how mu varies across the cell. If mu is locally linear with gradient a
over (T, g1, g2) and the cell's covariance of (T, g1, g2) is C, the cell's average P(wrong) is about Phi(-mu / sqrt(sd^2 +
a' C a)). a comes from finite differences of mu between neighbouring nodes and C from the exact density, so the correction
costs no forward passes. Reference: the same design estimator on a fine grid (30 T bins x 25 x 25 gap bins), so the design's
own error cancels; a half-resolution grid (15 x 13 x 13) checks the reference. Twelve suite models, copies 0 and 1 of each
base; the gauss_margin levels with and without the correction; the error is estimate - reference accuracy.

Finding: research/angles/margin-gaussian.md
"""
import numpy as np
import torch as th

from az.bench import suite
from az.estimators.gauss_margin import GAP_EDGES, LADDER, cell_spread, cells, moments, t_edges
from experiments.common import DEV

th.set_grad_enabled(False)
labels, ws = suite(0.002, 32)
pick = [i * 32 + j for i in range(6) for j in range(2)]
FINE_G = np.array([0, 0.0035, 0.007, 0.011, 0.015, 0.0225, 0.03, 0.04, 0.05, 0.065, 0.08, 0.1, 0.12, 0.15, 0.18, 0.225,
                   0.27, 0.33, 0.4, 0.5, 0.6, 0.8, 1.0, 1.4, 2.0, np.inf])
HALF_G = FINE_G[[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 25]]
GRIDS = {"fine": (t_edges(30), FINE_G), "half": (t_edges(15), HALF_G)}
GRIDS.update({f"level {lv}": (t_edges(lv[0]), np.array(GAP_EDGES[lv[1]])) for lv in LADDER})


def accuracy(w, grid):
    """Accuracy on a grid, plain and with the within-cell spread of mu added to the variance."""
    te, ge = grid
    mass, nodes, cov = cells(te, ge)
    mu, sd = moments(w, nodes)  # (cells, families)
    spread = cell_spread(mu, nodes, cov, (len(te) - 1, len(ge) - 1, len(ge) - 1))
    mass = th.tensor(mass, device=DEV)[:, None]
    return [1 - float((mass * th.special.ndtr(-mu / v.sqrt())).sum() / mu.shape[1]) for v in (sd ** 2, sd ** 2 + spread)]


rows = []
names = ["half"] + [f"{g} {'+ spread' if s else 'plain'}" for g in GRIDS if g.startswith("level") for s in (False, True)]
for i in pick:
    ref, ref_s = accuracy(ws[i], GRIDS["fine"])
    r = [accuracy(ws[i], GRIDS["half"])[0] - ref]
    r += [a - ref for g in GRIDS if g.startswith("level") for a in accuracy(ws[i], GRIDS[g])]
    rows.append(r + [ref_s - ref])
    print(f"{labels[i]:10s} fine reference {ref:.4f}; " + " ".join(f"{n} {e:+.4f}" for n, e in zip(names + ["fine + spread"], rows[-1])), flush=True)
rows = np.array(rows)
print("\nestimate - fine-grid reference accuracy over 12 models: mean, rms")
for n, col in zip(names + ["fine + spread"], rows.T):
    print(f"  {n:24s} {col.mean():+.5f} {np.sqrt((col ** 2).mean()):.5f}")
