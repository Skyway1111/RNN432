"""Exact S-lines over a family-stratified lattice: preintegrate the 2nd max's value, cover the rest by a Kronecker sequence.

Along the line where only the 2nd largest value S moves between T and M, the model is piecewise linear in S and its error
set is the two ends, the two near-tie wedges (parked/research/angles/family-lattice.md). So P(correct | everything but S) is
computed exactly: the state before position q is fixed; from q on, each piece's state is affine in S, so each step's
breakpoints are roots of 16 affine functions; at the end each piece's correct set is an interval, weighed by S's law
N(0, 1) restricted to (T, M). Families are enumerated. (T, M | T) are stratified by equal conditional mass, K_T x K_M per
family; within each stratum the position, and the seven lower inputs (quantiles of N(0, 1) truncated above at T), come
from the family-stratified lattice (az/estimators/lattice.py, which this variant extends with the exact S-integral; at
equal ops the two tie). Ops: every multiply-add of the line tracking, plus the inverse CDFs per node.
"""
from functools import cache

import numpy as np
import torch as th
from scipy.stats import norm

from az.estimators.gauss_margin import FAMS, Q
from az.estimators.lattice import family_nodes, place
from az.model import SAMPLE_OPS
from az.ops import FN

STEP = 16 * 16 + 16  # one recurrent step on a point state
PIECE_STEP = 2 * 256 + 3 * 16  # W c, W d, input add, root and mask on one piece
PIECE_READ = 2 * 160 + 9 * 4 + 2 * FN  # Wo c, Wo d, nine half-line bounds, two Phi
SCREEN = 2 * 160 + 10 * 4  # level_mass: Wo c, Wo d, each logit's best end
LEVEL_READ = 10 * (20 + FN) + 30 + 18 * FN  # level_mass, per piece and level past the screen: roots, sort, Phi at the ends
NODE = 10 * FN  # inverse CDFs for T, M and the seven lower inputs, and Phi(T), Phi(M)
PASSES_PER_LINE = 6.0  # measured mean, used only to pick the strata from the budget
DEV = "cuda" if th.cuda.is_available() else "cpu"


@cache
def nodes(kt: int, km: int, shift: tuple = (0.5,) * 9):
    """Inputs (lines, 10) with x_q a placeholder (S is integrated), label slots q, and the S range (T, M) per line, from
    the family-stratified lattice (az/estimators/lattice.py) in 9 dimensions. shift is the sequence's offset (Roberts'
    standard 1/2; experiments vary it to measure how much a particular lattice is luck)."""
    u, T, M = family_nodes(kt, km, 9, shift)
    z = norm.ppf(norm.cdf(T)[..., None] * u[..., 2:])
    X = place(T, M, np.zeros_like(T), z).reshape(-1, 10)
    return tuple(th.tensor(a, device=DEV) for a in (X, np.repeat(Q, kt * km), T.reshape(-1), M.reshape(-1)))


def pieces(Wi, Wh, Wo, X, q, T, M):
    """Track every line where x_q = S runs over (T, M): on each piece the state, and so the logits, are affine in S.
    Returns per piece its interval (lo, hi), its line's index into X, and logits Lc + Ld S (pieces, 10); ops per line."""
    ops = th.zeros(len(X), dtype=X.dtype, device=X.device)
    out = []
    for qq in range(10):
        sel = (q == qq).nonzero().squeeze(1)
        if not len(sel):
            continue
        x = X[sel]
        h = th.zeros(len(sel), 16, dtype=X.dtype, device=X.device)
        for t in range(qq):
            h = th.relu(h @ Wh.T + x[:, t, None] * Wi)
        ops[sel] += qq * STEP
        lo, hi, line = T[sel].clone(), M[sel].clone(), th.arange(len(sel), device=X.device)
        c, d = h, th.zeros_like(h)
        for t in range(qq, 10):  # on each piece the state is c + d S
            A = c @ Wh.T + (0 if t == qq else x[line, t, None] * Wi)
            B = d @ Wh.T + (Wi if t == qq else 0)
            ops.index_add_(0, sel[line], th.full_like(lo, PIECE_STEP))
            r = -A / B
            ok = (B != 0) & (r > lo[:, None]) & (r < hi[:, None])
            bounds = th.cat([lo[:, None], th.where(ok, r, hi[:, None]).sort(1).values, hi[:, None]], 1)
            keep = (th.arange(17, device=X.device) <= ok.sum(1, keepdim=True)) & (bounds[:, 1:] > bounds[:, :-1])
            par = th.arange(len(lo), device=X.device)[:, None].expand(-1, 17)[keep]
            lo, hi = bounds[:, :-1][keep], bounds[:, 1:][keep]
            on = ((A[par] + B[par] * ((lo + hi) / 2)[:, None]) > 0).to(X.dtype)
            c, d, line = on * A[par], on * B[par], line[par]
        out.append((lo, hi, sel[line], c @ Wo.T, d @ Wo.T))
    lo, hi, line, Lc, Ld = (th.cat(z) for z in zip(*out))
    return lo, hi, line, Lc, Ld, ops


def gaps(Lc, Ld, q):
    """Each piece's l_k - l_q = a + b S, with the label's own entry at a = -inf."""
    a, b = Lc - Lc.gather(1, q[:, None]), Ld - Ld.gather(1, q[:, None])
    return a.scatter(1, q[:, None], -th.inf), b.scatter(1, q[:, None], 0.0)


def sline(Wi, Wh, Wo, X, q, T, M):
    """Exact P(correct) on each line where x_q = S runs over (T, M) with law N(0,1) restricted there; ops per line."""
    lo, hi, line, Lc, Ld, ops = pieces(Wi, Wh, Wo, X, q, T, M)
    ops += NODE
    ops.index_add_(0, line, th.full_like(lo, PIECE_READ))
    a, b = gaps(Lc, Ld, q[line])  # correct where a + b S < 0 for every other logit
    root = -a / b
    lo = th.maximum(lo, th.where(b < 0, root, -th.inf).max(1).values)
    hi = th.minimum(hi, th.where(b > 0, root, th.inf).min(1).values)
    good = ~((b == 0) & (a >= 0)).any(1)
    mass = (th.special.ndtr(hi) - th.special.ndtr(lo)).clamp(min=0) * good
    out = th.zeros(len(X), dtype=X.dtype, device=X.device).index_add_(0, line, mass)
    return out / (th.special.ndtr(M) - th.special.ndtr(T)), ops


def ndtr_diff(a, b):
    """Phi(b) - Phi(a) for a <= b, accurate in both tails."""
    return th.where(a > 0, th.special.ndtr(-a) - th.special.ndtr(-b), th.special.ndtr(b) - th.special.ndtr(a))


def level_mass(Wi, Wh, Wo, X, q, T, M, levels):
    """int_T^M phi(s) 1{max_k l_k - l_q >= c |x|} ds on each line, per level c (lines, L), and ops per line; x_q's own
    entry in X is ignored. On a piece l_k - l_q = a + b s, and a + b s - c sqrt(s^2 + rho^2) is concave in s, so each
    logit's event set is one interval: the roots of (b^2 - c^2) s^2 + 2ab s + a^2 - c^2 rho^2, or a half-line where
    |b| > c. The event is their union, measured by sorting the intervals."""
    lo, hi, line, Lc, Ld, ops = pieces(Wi, Wh, Wo, X, q, T, M)
    a, b = (v[:, None] for v in gaps(Lc, Ld, q[line]))  # (P, 1, 10)
    rho2 = ((X**2).sum(1) - X.gather(1, q[:, None])[:, 0] ** 2)[line, None, None]
    c = levels.view(1, -1, 1)
    top = th.maximum(a + b * lo[:, None, None], a + b * hi[:, None, None]).amax((1, 2))
    live = top[:, None] >= levels * rho2[:, 0, 0].sqrt()[:, None]  # (P, L): |x| >= rho, so below this no event
    ops.index_add_(0, line, SCREEN + live.sum(1).to(X.dtype) * LEVEL_READ)
    D = a**2 + rho2 * (b**2 - c**2)
    qd = -(a * b + th.where(a * b >= 0, 1.0, -1.0) * c * D.clamp(min=0).sqrt())  # stable roots
    ra, rb = qd / (b**2 - c**2), (a**2 - c**2 * rho2) / qd
    r1, r2 = th.minimum(ra, rb), th.maximum(ra, rb)
    inside, lin = b.abs() < c, c == 0
    st = th.where(inside, r1, th.where(b > 0, r2, -th.inf))
    en = th.where(inside, r2, th.where(b > 0, th.inf, r1))
    empty = inside & ((a < 0) | (D < 0))
    z0 = -a / b  # c = 0: the half-line a + b s >= 0
    st = th.where(lin, th.where(b > 0, z0, th.where((b < 0) | (a >= 0), -th.inf, th.inf)), st)
    en = th.where(lin, th.where(b < 0, z0, th.where((b > 0) | (a >= 0), th.inf, -th.inf)), en)
    empty = (empty & ~lin) | th.isinf(a) | st.isnan() | en.isnan()
    st, en = th.maximum(st, lo[:, None, None]), th.minimum(en, hi[:, None, None])
    st = th.where(empty | (en <= st), th.inf, st)
    st, o = st.sort(-1)
    en = th.where(st == th.inf, th.inf, en.gather(-1, o))
    prev = th.cat([th.full_like(en[..., :1], -th.inf), en[..., :-1].cummax(-1).values], -1)
    s0 = th.maximum(st, prev)
    ok = (en > s0) & en.isfinite()
    mass = th.where(ok, ndtr_diff(th.where(ok, s0, 0.0), th.where(ok, en, 0.0)), 0.0).sum(-1) * live
    return th.zeros(len(X), len(levels), dtype=X.dtype, device=X.device).index_add_(0, line, mass), ops


def strata(budget):
    """The largest (k, k) or (k, k + 1) strata per family the budget affords, at least (1, 1)."""
    n = budget / (len(FAMS) * PASSES_PER_LINE * SAMPLE_OPS)
    k = max(1, int(np.sqrt(n)))
    return (k, k + 1) if k * (k + 1) <= n else (k, k)


def estimate(w, budget):
    X, q, T, M = nodes(*strata(budget))
    W = [th.tensor(a, device=DEV, dtype=th.float64) for a in w]
    with th.no_grad():
        p, ops = zip(*(sline(*W, X[c : c + 2**17], q[c : c + 2**17], T[c : c + 2**17], M[c : c + 2**17]) for c in range(0, len(X), 2**17)))
    return th.cat(p).mean().item(), th.cat(ops).sum().item()
