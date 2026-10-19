"""Family-stratified lattice with the last input integrated exactly: the cheapest exact deduction, the last step's kinks.

Each root fixes positions 0-8 from the family-stratified lattice (az/estimators/lattice.py); the value at position 9,
whatever rank the family gives it, has the conditional law N(0, 1) restricted to the interval its rank allows (between
the largest fixed value of lower rank and the smallest of higher rank). Given h_9 the last pre-activations are affine in
that value, so the logits are affine between at most 16 kinks (one per neuron), and the correct set is an interval per
segment: P(correct | root) is exact. It removes about 45% of the indicator's variance for 0.27 of a pass, where the
exact S-line (sline.py) removes 50-75% for four to seven passes; prefix-sharing trees remove nothing. The gain is the
size of a lattice shift: 1.3x at 1e9 on the held-out suite, a tie on the bench suite (research/angles/family-lattice.md).
Not a mechanistic estimate: its error is equidistribution over the roots, and it reads the weights only by running
them; with the lattice it is the baseline a mechanistic estimate must beat.
Ops: nine recurrent steps, the inverse CDFs, then A = Wh h_9, the kinks and their sort, the logits' affine form at the
low end, one neuron's toggle per kink, and nine roots, the interval and two Phi per segment.
"""
import numpy as np
import torch as th
from scipy.stats import norm

from az.estimators.gauss_margin import FAMS, Q
from az.estimators.lattice import NODE_FN, family_nodes, place, strata
from az.ops import FN

DEV = "cuda" if th.cuda.is_available() else "cpu"
STEP = 16 * 16 + 16  # one recurrent step
READ = 160  # logits from h_10
SWEEP_FIXED = 256 + 16 + 64 + 2 * READ  # A = Wh h_9, the kinks, their sort, the logits' affine form at the low end
SWEEP_KINK, SWEEP_SEG = 20, 50  # one neuron's toggle in the affine logits; nine roots, the interval, two Phi
PER_ROOT = 9 * STEP + (NODE_FN + 2) * FN + SWEEP_FIXED + 6 * (SWEEP_KINK + SWEEP_SEG)  # about 5.6 kinks inside on average
RANK = th.full((720, 10), 3, dtype=th.long, device=DEV)  # M 0, S 1, T 2, lower 3
for k, (p, q, r) in enumerate(FAMS):
    RANK[k, p], RANK[k, q], RANK[k, r] = 0, 1, 2
Phi = th.special.ndtr


def roots(kt: int, km: int, d: int = 10):
    """Lattice roots: inputs (N, 10) from the family lattice with kt x km strata, family index, label, and the
    Kronecker coordinates beyond the lattice's ten (N, d - 10) for experiments that branch on the last positions."""
    u, T, M = family_nodes(kt, km, d)
    S = norm.ppf(norm.cdf(T) + (norm.cdf(M) - norm.cdf(T)) * u[..., 9])
    z = norm.ppf(norm.cdf(T)[..., None] * u[..., 2:9])
    n = kt * km
    X = th.tensor(place(T, M, S, z).reshape(-1, 10), device=DEV)
    return X, th.arange(720, device=DEV).repeat_interleave(n), th.tensor(np.repeat(Q, n), device=DEV), th.tensor(u[..., 10:].reshape(len(X), -1), device=DEV)


def interval(X, fam, pos: int, fixed: list):
    """Conditional interval of the value at `pos` given the values at the `fixed` positions and the family's ranks."""
    rk = RANK[fam]
    mine = rk[:, pos, None]
    below = th.where(rk[:, fixed] > mine, X[:, fixed], -th.inf).max(1).values
    above = th.where(rk[:, fixed] < mine, X[:, fixed], th.inf).min(1).values
    return below, above


def prefix(W, X, steps: int):
    Wi, Wh, _ = W
    h = th.zeros(len(X), 16, dtype=X.dtype, device=X.device)
    for t in range(steps):
        h = th.relu(h @ Wh.T + X[:, t, None] * Wi)
    return h


def sweep(W, h, q, lo, hi):
    """Exact P(correct) with the last input over (lo, hi) under phi, given h_9; and the kinks inside the interval."""
    Wi, Wh, Wo = W
    A, B = h @ Wh.T, Wi
    r = th.where(B != 0, -A / B, th.inf)
    ok = (r > lo[:, None]) & (r < hi[:, None])
    bounds = th.cat([lo[:, None], th.where(ok, r, hi[:, None]).sort(1).values, hi[:, None]], 1)
    plo, phi = bounds[:, :-1], bounds[:, 1:]
    on = ((A[:, None] + B * ((plo + phi) / 2)[..., None]) > 0).to(h.dtype)  # (N, 17, 16)
    Lc, Ld = (on * A[:, None]) @ Wo.T, (on * B) @ Wo.T
    qq = q[:, None, None].expand(-1, 17, 1)
    a, b = Lc.gather(2, qq) - Lc, Ld.gather(2, qq) - Ld
    other = th.ones_like(a, dtype=th.bool).scatter_(2, qq, False)
    root = -a / b
    lo_c = th.maximum(plo, th.where(other & (b > 0), root, -th.inf).max(2).values)
    hi_c = th.minimum(phi, th.where(other & (b < 0), root, th.inf).min(2).values)
    good = ~(other & (b == 0) & (a <= 0)).any(2) & (phi > plo)
    mass = ((Phi(hi_c) - Phi(lo_c)).clamp(min=0) * good).sum(1)
    return mass / (Phi(hi) - Phi(lo)), ok.sum(1)


def estimate(w, budget):
    X, fam, q, _ = roots(*strata(budget, PER_ROOT))
    W = [th.tensor(a, device=DEV, dtype=th.float64) for a in w]
    lo, hi = interval(X, fam, 9, list(range(9)))
    with th.no_grad():
        p, kinks = zip(*(sweep(W, prefix(W, X[c : c + 2**18], 9), q[c : c + 2**18], lo[c : c + 2**18], hi[c : c + 2**18]) for c in range(0, len(X), 2**18)))
    kinks = th.cat(kinks)
    ops = len(X) * (9 * STEP + (NODE_FN + 2) * FN + SWEEP_FIXED) + (kinks * (SWEEP_KINK + SWEEP_SEG) + SWEEP_SEG).sum().item()
    return th.cat(p).mean().item(), ops
