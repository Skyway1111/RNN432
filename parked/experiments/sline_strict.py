"""Can the S-line estimator beat sampling without its quasi-random sequence, by strict quadrature rules and designs?

Exact S-lines (az.estimators.sline.sline) at the nodes of tensor rules: T by Gauss-Legendre on its quantile scale (or panels
graded toward the low tail); M through s = (Phi(M) - Phi(T)) / (1 - Phi(T)), density 2 s, by Gauss-Legendre (or panels
graded toward s = 0, small M - T). The lower inputs from gauss_margin's 16-run design, either all 16 runs at every node
for every family (identical), or one run per node with the run index rotating with family and node (a classical design's
rows spread across families, the closest strict analogue of the quasi-random sequence); and the same construction with
64 quantiles of the largest lower input, rotated. Scored on the whole suite.

Finding: research/angles/family-lattice.md
"""
import numpy as np
import torch as th
from scipy.special import roots_legendre
from scipy.stats import norm

from az.bench import suite, truth
from az.estimators.gauss_margin import FAMS, FREE, H8, Q, truncated
from az.estimators.sline import sline
from az.model import SAMPLE_OPS
from experiments.common import DEV

labels, ws = suite(0.002, 32)
P = truth(0.002, 32)
_t = np.linspace(-6, 6, 400001)
_cdf = np.cumsum(360 * norm.pdf(_t) * norm.cdf(_t) ** 7 * (1 - norm.cdf(_t)) ** 2)
_cdf /= _cdf[-1]


def rule(edges, k):
    """Gauss-Legendre with k nodes on each panel of edges."""
    x, w = roots_legendre(k)
    a, b = np.array(edges[:-1])[:, None], np.array(edges[1:])[:, None]
    return ((a + b) / 2 + (b - a) / 2 * x).ravel(), ((b - a) / 2 * w).ravel()


def os_rows(T, R):
    """(nT, R, 7): gauss_margin's order-statistic design with R quantiles of the largest lower input (R = 16 is
    gauss_margin.lower_inputs): run j puts it at quantile ((5 j mod R) + 1/2) / R of its law in slot j mod 7, and the
    other six at mean +- sd of N(0, 1) truncated at it, signs from H8 and its mirror, cycled."""
    v = norm.ppf(norm.cdf(T)[:, None] * (((5 * np.arange(R)) % R + 0.5) / R) ** (1 / 7))
    mu, sd = (a.cpu().numpy() for a in truncated(th.tensor(v, device=DEV)))
    signs = np.vstack([H8, -H8])[:, :6]
    Z = np.zeros((len(T), R, 7))
    for j in range(R):
        rest = [i for i in range(7) if i != j % 7]
        Z[:, j, rest] = mu[:, j, None] + sd[:, j, None] * signs[j % 16]
        Z[:, j, j % 7] = v[:, j]
    return Z


def lines(t_rule, s_rule, rotate, R=16):
    (u, wt), (s, ws_) = t_rule, s_rule
    T, wt = np.interp(u, _cdf, _t), wt / wt.sum()
    ws_ = ws_ * 2 * s / (ws_ * 2 * s).sum()
    Z = os_rows(T, R)  # (nT, R, 7)
    TT = np.repeat(T[:, None], len(s), 1)
    MM = norm.ppf(norm.cdf(TT) + (1 - norm.cdf(TT)) * s)
    W = wt[:, None] * ws_[None]
    F, nT, nS = len(FAMS), len(T), len(s)
    if rotate:
        run = (np.arange(F)[:, None, None] + 5 * np.arange(nT)[:, None] + 3 * np.arange(nS)) % R
        z = np.take_along_axis(Z[None, :, None].repeat(nS, 2).repeat(F, 0), run[..., None, None].repeat(7, 4), 3)[..., 0, :]
        R = 1
    else:  # every run at every node
        z = np.broadcast_to(Z[None, :, None], (F, nT, nS, R, 7))
    z = z.reshape(F, nT, nS, R, 7)
    X = np.zeros((F, nT, nS, R, 10))
    for k, (p, _, r) in enumerate(FAMS):
        X[k][..., FREE[k]] = z[k]
        X[k, ..., r], X[k, ..., p] = TT[..., None], MM[..., None]
    shape = (F, nT, nS, R)
    return [th.tensor(a, device=DEV) for a in (X.reshape(-1, 10), np.repeat(Q, nT * nS * R), np.broadcast_to(TT[None, ..., None], shape).reshape(-1),
            np.broadcast_to(MM[None, ..., None], shape).reshape(-1), np.broadcast_to(W[None, ..., None] / (F * R), shape).reshape(-1))]


def score(name, args):
    X, q, T, M, wl = args
    e = []
    for w in ws:
        W = [th.tensor(a, device=DEV, dtype=th.float64) for a in w]
        acc = ops = 0.0
        for c in range(0, len(X), 400_000):
            p, o = sline(*W, X[c : c + 400_000], q[c : c + 400_000], T[c : c + 400_000], M[c : c + 400_000])
            acc, ops = acc + (p * wl[c : c + 400_000]).sum().item(), ops + o.sum().item()
        e.append((acc, ops))
    e = np.array(e)
    err, ops = e[:, 0] - P, e[:, 1].mean()
    ms = np.mean(P * (1 - P)) / (ops / SAMPLE_OPS)
    print(f"{name:52s} ops {ops:.2e} rmse {np.sqrt(np.mean(err ** 2)):.4f} bias {err.mean():+.4f} mse/samp {np.mean(err ** 2) / ms:.1f}", flush=True)


score("identical 16 runs, T 4 x s 6 Gauss", lines(rule([0, 1], 4), rule([0, 1], 6), False))
score("identical 16 runs, T 6 x s 8 Gauss", lines(rule([0, 1], 6), rule([0, 1], 8), False))
score("rotated 1 run, T 6 x s 8 Gauss", lines(rule([0, 1], 6), rule([0, 1], 8), True))
score("rotated 1 run, T 3-panel x s 4-panel (graded), 3 nodes each", lines(rule([0, 0.05, 0.3, 1], 3), rule([0, 0.04, 0.15, 0.45, 1], 3), True))
score("rotated 1 run, T 5-panel x s 6-panel (graded), 3 nodes each", lines(rule([0, 0.01, 0.05, 0.2, 0.5, 1], 3), rule([0, 0.02, 0.06, 0.15, 0.35, 0.65, 1], 3), True))
score("rotated 1 run of 64, T 12 x s 8 Gauss", lines(rule([0, 1], 12), rule([0, 1], 8), True, R=64))
score("rotated 1 run of 64, T 24 x s 8 Gauss", lines(rule([0, 1], 24), rule([0, 1], 8), True, R=64))
