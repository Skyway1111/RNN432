"""The two untried levers on the family-stratified lattice: an exact sweep over the last input, and prefix-sharing trees.

Both exploit the recurrence's sequential cost, not the weights. On the held-out suite (noise seeds 5000 + i, truth cached
by experiments/lattice_validation.py), at equal ops against az.estimators.lattice:
  sweep      each lattice node fixes positions 0-8; the value at position 9, whatever its rank, is integrated exactly over
             its conditional interval under phi (az/estimators/sweep.py, shipped as the baseline after this experiment).
  tree(9, b) each lattice node is a root fixing positions 0-8; position 9 takes b values stratified over its conditional
             interval (offset from the root's Kronecker coordinate), each one recurrent step and a readout.
  tree(8, 9, b)  positions 0-7 fixed; position 8 takes b stratified values, position 9 b values given each; when the two
             are order-adjacent the leaves carry the exact weight of position 9's conditional mass, normalized in closed
             form.
  tree + sweep   position 8 branched b ways, position 9 swept.
Part 0 checks the sweep against 4,000 equal-mass nodes per line and the trees' unbiasedness at a large budget.
Variance components of the indicator along the sequence come from research/angles/value-sketch.md.

Finding: research/angles/family-lattice.md
"""
import numpy as np
import torch as th

from az.estimators import lattice as L
from az.estimators.sweep import PER_ROOT, RANK, READ, STEP, SWEEP_FIXED, SWEEP_KINK, SWEEP_SEG, Phi, interval, prefix, sweep
from az.estimators.sweep import roots as sweep_roots
from az.model import DATA, NAMES, SAMPLE_OPS, logits, load, noisy
from az.ops import FN
from experiments.common import DEV

VAL = [w for i, name in enumerate(NAMES) for w in noisy(load(name), 0.002, 32, seed=5000 + i)]
PV = np.load(DATA / "cache" / "val_suite_truth.npy")


def ppf(u):
    return th.special.ndtri(u.clamp(1e-300, 1 - 1e-16))


def roots(kt, km):
    return sweep_roots(kt, km, 12)  # two Kronecker coordinates beyond the lattice's ten, for the leaves' offsets


def leaves(lo, hi, b, u):
    """b values stratified over (lo, hi) under phi, offset by u in [0, 1): (N, b)."""
    j = (th.arange(b, device=DEV, dtype=th.float64) + u[:, None]) / b
    return ppf(Phi(lo)[:, None] + (Phi(hi) - Phi(lo))[:, None] * j)


def est_lattice(w, X, q):
    hits = (logits([w], X[None])[0].argmax(1) == q).double()
    return hits.mean().item(), len(X) * (SAMPLE_OPS + L.NODE_FN * FN)


def est_sweep(w, W, X, fam, q, u):
    lo, hi = interval(X, fam, 9, list(range(9)))
    p, kinks = sweep(W, prefix(W, X, 9), q, lo, hi)
    ops = len(X) * (9 * STEP + (L.NODE_FN + 2) * FN + SWEEP_FIXED) + (kinks * (SWEEP_KINK + SWEEP_SEG) + SWEEP_SEG).sum().item()
    return p.mean().item(), ops


def est_tree9(w, W, X, fam, q, u, b):
    Wi, Wh, Wo = W
    lo, hi = interval(X, fam, 9, list(range(9)))
    h = prefix(W, X, 9)
    v = leaves(lo, hi, b, u[:, 0])
    h10 = th.relu(h[:, None] @ Wh.T + v[..., None] * Wi)
    hits = ((h10 @ Wo.T).argmax(2) == q[:, None]).double()
    return hits.mean().item(), len(X) * (9 * STEP + b * (STEP + READ) + (L.NODE_FN + 2 + b) * FN)


def mass_norm(lo8, hi8, lo9, hi9, kind):
    """Z / Z8: the mean over v8 ~ phi on (lo8, hi8) of position 9's conditional mass, in closed form. kind 0: the
    interval of 9 does not move with v8; +1: its top is min(hi9, v8); -1: its bottom is max(lo9, v8)."""
    F = Phi
    Z8 = F(hi8) - F(lo8)
    flat = (F(hi9) - F(lo9)).clamp(min=0)
    cut_hi = th.minimum(hi8, hi9)  # kind +1: below hi9 the top is v8
    part_hi = (F(cut_hi) ** 2 / 2 - F(lo9) * F(cut_hi)) - (F(lo8) ** 2 / 2 - F(lo9) * F(lo8))
    z_hi = th.where(cut_hi > lo8, part_hi, 0.0) + flat * (F(hi8) - F(th.maximum(lo8, hi9))).clamp(min=0)
    cut_lo = th.maximum(lo8, lo9)  # kind -1: above lo9 the bottom is v8
    part_lo = (F(hi9) * F(hi8) - F(hi8) ** 2 / 2) - (F(hi9) * F(cut_lo) - F(cut_lo) ** 2 / 2)
    z_lo = th.where(hi8 > cut_lo, part_lo, 0.0) + flat * (F(th.minimum(hi8, lo9)) - F(lo8)).clamp(min=0)
    Z = th.where(kind == 0, Z8 * flat, th.where(kind > 0, z_hi, z_lo))
    return Z / Z8


def est_tree89(w, W, X, fam, q, u, b, swept=False):
    Wi, Wh, Wo = W
    N = len(X)
    rk = RANK[fam]
    lo8, hi8 = interval(X, fam, 8, list(range(8)))
    lo9, hi9 = interval(X, fam, 9, list(range(8)))
    r8, r9 = rk[:, 8], rk[:, 9]
    between = ((rk[:, :8] > th.minimum(r8, r9)[:, None]) & (rk[:, :8] < th.maximum(r8, r9)[:, None])).any(1)
    kind = th.where((r8 == r9) | between, 0, th.where(r8 < r9, 1, -1))
    v8 = leaves(lo8, hi8, b, u[:, 0])  # (N, b)
    lo9v = th.where(kind[:, None] < 0, th.maximum(lo9[:, None], v8), lo9[:, None])
    hi9v = th.where(kind[:, None] > 0, th.minimum(hi9[:, None], v8), hi9[:, None])
    wgt = (Phi(hi9v) - Phi(lo9v)).clamp(min=0) / mass_norm(lo8, hi8, lo9, hi9, kind)[:, None]  # mean 1 over v8
    h8 = prefix(W, X, 8)
    h9 = th.relu(h8[:, None] @ Wh.T + v8[..., None] * Wi).reshape(N * b, 16)
    ops = N * (8 * STEP + b * STEP + (L.NODE_FN + 4 + b + 2 * b + 8) * FN)
    if swept:
        p, kinks = sweep(W, h9, q.repeat_interleave(b), lo9v.reshape(-1), hi9v.reshape(-1))
        p = p.reshape(N, b)
        ops += N * b * (2 * FN + SWEEP_FIXED) + (kinks * (SWEEP_KINK + SWEEP_SEG) + SWEEP_SEG).sum().item()
    else:
        v9 = leaves(lo9v.reshape(-1), hi9v.reshape(-1), b, u[:, 1].repeat_interleave(b))  # (N b, b)
        h10 = th.relu(h9[:, None] @ Wh.T + v9[..., None] * Wi)
        p = ((h10 @ Wo.T).argmax(2) == q.repeat_interleave(b)[:, None]).double().mean(1).reshape(N, b)
        ops += N * b * (b * (STEP + READ) + b * FN)
    return (wgt * p).mean().item(), ops


def run(label, fn, per_root, budget, b=None):
    X, fam, q, u = roots(*L.strata(budget, per_root))
    est = np.array([fn(w, [th.tensor(a, device=DEV, dtype=th.float64) for a in w], X, fam, q, u) if b is None else
                    fn(w, [th.tensor(a, device=DEV, dtype=th.float64) for a in w], X, fam, q, u, b) for w in VAL])
    err, ops = est[:, 0] - PV, est[:, 1].mean()
    ratio = np.mean(err ** 2) / (np.mean(PV * (1 - PV)) * SAMPLE_OPS / ops)
    print(f"  {label:18s} roots {len(X):7d}, ops {ops:.2e} ({ops / len(X) / SAMPLE_OPS:.2f} passes per root), rmse {np.sqrt(np.mean(err ** 2)):.5f}, "
          f"bias {err.mean():+.5f}, mse/samp {ratio:.3f}", flush=True)
    return ratio


print("0. checks")
X, fam, q, u = roots(2, 2)
W = [th.tensor(a, device=DEV, dtype=th.float64) for a in VAL[0]]
lo, hi = interval(X, fam, 9, list(range(9)))
p, kinks = sweep(W, prefix(W, X, 9), q, lo, hi)
us = (th.arange(4000, device=DEV, dtype=th.float64) + 0.5) / 4000
ref = []
for c in range(0, len(X), 256):
    v = ppf(Phi(lo[c : c + 256])[:, None] + (Phi(hi[c : c + 256]) - Phi(lo[c : c + 256]))[:, None] * us)
    xx = X[c : c + 256, None].expand(-1, 4000, -1).clone()
    xx[:, :, 9] = v
    ref.append((logits([VAL[0]], xx.reshape(1, -1, 10))[0].argmax(1).reshape(-1, 4000) == q[c : c + 256, None]).double().mean(1))
d = p - th.cat(ref)
print(f"  sweep against 4,000 nodes per line: max |diff| {d.abs().max():.4f}, mean {d.mean():+.5f}; kinks inside the interval, mean {kinks.double().mean():.1f}")
print("  unbiasedness at 3e9 ops on example/0 (truth {:.5f}):".format(PV[0]))
for label, fn, per, b in (("lattice", lambda w, W, X, fam, q, u: est_lattice(w, X, q), SAMPLE_OPS + 120, None),
                          ("sweep", est_sweep, PER_ROOT, None), ("tree(9, 4)", est_tree9, 4300, 4), ("tree(8, 9, 2)", est_tree89, 4800, 2)):
    X, fam, q, u = roots(*L.strata(3e9, per))
    e, ops = fn(VAL[0], W, X, fam, q, u) if b is None else fn(VAL[0], W, X, fam, q, u, b)
    print(f"    {label:14s} {e:.5f} ({e - PV[0]:+.5f}) at {ops:.2e} ops")

for budget in (1e7, 1e8, 1e9):
    print(f"\nbudget {budget:.0e}")
    run("lattice", lambda w, W, X, fam, q, u: est_lattice(w, X, q), SAMPLE_OPS + L.NODE_FN * FN, budget)
    run("sweep", est_sweep, PER_ROOT, budget)
    run("tree(9, 2)", est_tree9, 3500, budget, 2)
    run("tree(9, 4)", est_tree9, 4300, budget, 4)
    run("tree(8, 9, 2)", est_tree89, 4800, budget, 2)
    run("tree(8, 2) + sweep", lambda w, W, X, fam, q, u, b: est_tree89(w, W, X, fam, q, u, b, swept=True), 5600, budget, 2)
