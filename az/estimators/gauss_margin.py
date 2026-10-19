"""Gaussian-margin estimator: exact quadrature over (T, M - S, S - T), uniform families, a Gaussian margin over the lower inputs.

Input space factors as: the family (p, q, r), uniform over 720; the third-largest value T and the two top gaps, with the exact
order-statistic density 720 phi(T + g1 + g2) phi(T + g2) phi(T) Phi(T)^7; and the seven lower inputs, i.i.d. N(0, 1) truncated
above at T, one per free position. The first two are integrated by quadrature on a (T, g1, g2) grid with exact bin masses.
Over the lower inputs the margin l_q - max other is taken as Gaussian (parked/research/angles/margin-gaussian.md), with mean and sd
from a 16-run design that places the largest lower input at quantiles of its law, since the margin turns sharply when it nears
T, and the other six about their mean below it (lower_inputs). Within a cell mu varies about linearly over (T, g1, g2), which
adds a' C a to the margin's variance (a: mu's gradient between neighbouring nodes, C: the cell's covariance).
Accuracy = 1 - sum of mass x mean over families of Phi(-mu / sqrt(sd^2 + a' C a)).
Budget picks the grid from LADDER; ops count the forward passes, the spread, a sqrt and a Phi per cell and family, and
the design's four transcendentals per run and cell.
"""
from functools import cache

import numpy as np
import torch as th
from scipy.linalg import hadamard
from scipy.stats import norm

from az.model import N_IN, SAMPLE_OPS, logits
from az.ops import FN

FAMS = [(p, q, r) for p in range(10) for q in range(10) for r in range(10) if len({p, q, r}) == 3]
FREE = np.array([[t for t in range(10) if t not in f] for f in FAMS])
Q = np.array([q for _, q, _ in FAMS])
H8 = hadamard(8)[:, 1:].astype(float)  # 8 runs x 7 factors, +-1
OS_Q = ((5 * np.arange(16)) % 16 + 0.5) / 16  # run j: quantile of the largest lower input
OS_TOP = np.arange(16) % 7  # run j: the free slot that holds it
OS_REST = np.array([[i for i in range(7) if i != s] for s in OS_TOP])  # the other six slots
OS_SIGNS = np.vstack([H8, -H8])[:, :6]  # their signs about the mean, H8 and its mirror
GAP_EDGES = {
    6: [0, 0.02, 0.05, 0.1, 0.2, 0.45, np.inf],
    8: [0, 0.015, 0.03, 0.06, 0.11, 0.2, 0.4, 0.8, np.inf],
    10: [0, 0.01, 0.02, 0.04, 0.07, 0.12, 0.2, 0.35, 0.6, 1.0, np.inf],
    12: [0, 0.007, 0.015, 0.03, 0.05, 0.08, 0.12, 0.18, 0.27, 0.4, 0.6, 1.0, np.inf],
}
TAIL = [0.0, 0.005, 0.015, 0.03, 0.06]  # quantiles of T bounding the low-tail bins, where accuracy falls steeply
LADDER = [(5, 6), (7, 8), (9, 10), (12, 12)]  # (T bins, gap bins per axis), by rising cost
SPREAD_OPS = 30  # per cell and family: three differences and quotients, the quadratic form a' C a, the sum
DEV = "cuda" if th.cuda.is_available() else "cpu"


def t_edges(n_t: int) -> np.ndarray:
    """Edges of the T bins: the low-tail quantiles in TAIL, then equal mass."""
    t = np.arange(-3.0, 3.5, 0.01)
    f_t = 360 * norm.pdf(t) * norm.cdf(t) ** 7 * (1 - norm.cdf(t)) ** 2  # density of the 3rd largest of 10
    cdf_t = np.cumsum(f_t) * 0.01
    cdf_t /= cdf_t[-1]
    qs = np.concatenate([TAIL, TAIL[-1] + (1 - TAIL[-1]) * np.arange(1, n_t - len(TAIL) + 2) / (n_t - len(TAIL) + 1)])
    e = np.interp(qs, cdf_t, t)
    e[0], e[-1] = -np.inf, np.inf
    return e


def cells(te: np.ndarray, ge: np.ndarray):
    """Cells of the tensor grid with T edges te and gap edges ge: exact masses (up to the lattice), mass-weighted nodes
    (T, g1, g2), and the covariance of (T, g1, g2) within each cell. Cell index = (T bin, g1 bin, g2 bin) in C order."""
    t = th.arange(-3.0, 3.5, 0.01, dtype=th.float64, device=DEV)
    ge_ = np.concatenate([np.arange(0, 0.05, 0.0005), np.arange(0.05, 0.5, 0.002), np.arange(0.5, 5.0001, 0.01)])
    g, dg = (th.tensor(a, device=DEV) for a in ((ge_[1:] + ge_[:-1]) / 2, np.diff(ge_)))  # midpoints, fine near 0
    n_t, n_g = len(te) - 1, len(ge) - 1
    bt = th.bucketize(t, th.tensor(te[1:-1], device=DEV), right=True)
    bg = th.bucketize(g, th.tensor(ge[1:-1], device=DEV), right=True)
    G1, G2 = (a.reshape(-1) for a in th.meshgrid(g, g, indexing="ij"))
    W = (dg[:, None] * dg[None, :]).reshape(-1)
    B = (bg[:, None] * n_g + bg[None, :]).reshape(-1)
    acc = th.zeros(n_t * n_g * n_g, 10, dtype=th.float64, device=DEV)  # sums of 1, x, and x x^T entries
    for i, ti in enumerate(t):
        x = th.stack([th.full_like(G1, ti), G1, G2], 1)
        dens = (W * th.special.ndtr(ti) ** 7 * th.exp(-(ti ** 2 + (ti + G2) ** 2 + (ti + G1 + G2) ** 2) / 2))[:, None]
        f = th.cat([dens, dens * x, dens * x[:, [0, 1, 2, 0, 0, 1]] * x[:, [0, 1, 2, 1, 2, 2]]], 1)
        acc.index_add_(0, B + int(bt[i]) * n_g * n_g, f)
    mass = acc[:, 0]
    mean = acc[:, 1:4] / mass[:, None]
    sec = acc[:, 4:] / mass[:, None]
    cov = th.zeros(len(mass), 3, 3, dtype=th.float64, device=DEV)
    for k, (a, b) in enumerate([(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]):
        cov[:, a, b] = cov[:, b, a] = sec[:, k] - mean[:, a] * mean[:, b]
    return (mass / mass.sum()).cpu().numpy(), mean.cpu().numpy(), cov.cpu().numpy()


@cache
def grid(level: tuple[int, int]):
    """Masses, nodes and covariances of a LADDER level's cells."""
    return cells(t_edges(level[0]), np.array(GAP_EDGES[level[1]]))


def quadrature(n_t: int, n_g: int):
    return grid((n_t, n_g))[:2]


def cell_spread(mu: th.Tensor, nodes: np.ndarray, cov: np.ndarray, shape: tuple) -> th.Tensor:
    """(cells, families) variance of mu across each cell, a' C a: a is mu's gradient over (T, g1, g2) by finite
    differences between neighbouring nodes of the tensor grid of the given shape, C the cell's covariance."""
    m = mu.reshape(*shape, -1)
    X = th.tensor(nodes, device=mu.device).reshape(*shape, 3)
    a = []
    for k in range(3):
        i = th.arange(shape[k], device=mu.device)
        lo, hi = (i - 1).clamp(min=0), (i + 1).clamp(max=shape[k] - 1)
        dx = X[..., k].index_select(k, hi) - X[..., k].index_select(k, lo)
        a.append((m.index_select(k, hi) - m.index_select(k, lo)) / dx[..., None])
    a = th.stack(a, -1).reshape(len(nodes), -1, 3)
    return th.einsum("cfi,cij,cfj->cf", a, th.tensor(cov, device=mu.device), a)


def truncated(t: th.Tensor):
    """Mean and sd of N(0, 1) truncated above at t."""
    a = th.exp(-t ** 2 / 2) / np.sqrt(2 * np.pi) / th.special.ndtr(t)
    return -a, (1 - t * a - a * a).clamp(min=1e-12).sqrt()


def lower_inputs(t: th.Tensor) -> th.Tensor:
    """(cells, 16, 7) values of the seven lower inputs, one per free slot in order, given T = t (cells,). Run j puts their
    maximum, whose law is (Phi(v) / Phi(T))^7, at quantile OS_Q[j] in slot OS_TOP[j], and the other six at the mean +- sd
    of N(0, 1) truncated above at that maximum."""
    v = th.special.ndtri(th.special.ndtr(t)[:, None] * th.tensor(OS_Q, device=t.device) ** (1 / 7))
    mu, sd = truncated(v)
    low = th.zeros(len(t), len(OS_Q), 7, dtype=t.dtype, device=t.device)
    low.scatter_(2, th.tensor(OS_REST, device=t.device).expand(len(t), -1, -1), mu[..., None] + sd[..., None] * th.tensor(OS_SIGNS, device=t.device))
    return low.scatter_(2, th.tensor(OS_TOP, device=t.device)[None, :, None].expand(len(t), -1, 1), v[..., None])


def margins(w, nodes: np.ndarray, low: th.Tensor | None = None) -> th.Tensor:
    """(cells, families, runs) margins l_q - max other at each node (T, g1, g2). The lower inputs, one per free slot in
    order, are low (cells, runs, 7) shared by all families or (cells, families, runs, 7), by default lower_inputs(T)."""
    t = th.tensor(nodes, device=DEV)
    low = lower_inputs(t[:, 0]) if low is None else low
    top = th.stack([t[:, 0] + t[:, 2] + t[:, 1], t[:, 0] + t[:, 2], t[:, 0]], 1)  # M, S, T
    pos, free, q = (th.tensor(a, device=DEV) for a in (FAMS, FREE, Q))
    C, F, R = len(nodes), len(FAMS), low.shape[-2]
    low = low[:, None].expand(-1, F, -1, -1) if low.dim() == 3 else low
    out = []
    for c in range(0, C, max(1, 2**20 // (F * R))):
        sl = slice(c, min(C, c + max(1, 2**20 // (F * R))))
        n = sl.stop - sl.start
        x = th.zeros(n, F, R, N_IN, dtype=th.float64, device=DEV)
        x.scatter_(3, free[None, :, None].expand(n, -1, R, -1), low[sl].contiguous())
        x.scatter_(3, pos[None, :, None].expand(n, -1, R, -1), top[sl, None, None].expand(-1, F, R, -1).contiguous())
        L = logits([w], x.reshape(1, -1, N_IN))[0].reshape(n, F, R, N_IN)
        lq = L.gather(3, q[None, :, None, None].expand(n, -1, R, 1))[..., 0]
        out.append(lq - L.scatter(3, q[None, :, None, None].expand(n, -1, R, 1), -1e9).max(3).values)
    return th.cat(out)


def moments(w, nodes: np.ndarray):
    """Design mean and sd of the margin, (cells, families)."""
    m = margins(w, nodes)
    return m.mean(2), m.std(2, correction=0).clamp(min=1e-9)


def estimate(w, budget):
    level = LADDER[0]
    for n_t, n_g in LADDER:
        if n_t * n_g * n_g * len(FAMS) * len(OS_Q) * SAMPLE_OPS <= budget:
            level = (n_t, n_g)
    mass, nodes, cov = grid(level)
    with th.no_grad():
        mu, sd = moments(w, nodes)
        var = sd ** 2 + cell_spread(mu, nodes, cov, (level[0], level[1], level[1]))
        p_wrong = th.special.ndtr(-mu / var.sqrt())  # (cells, families)
    acc = 1 - float((th.tensor(mass, device=DEV)[:, None] * p_wrong).sum() / len(FAMS))
    return acc, len(mass) * (len(FAMS) * (len(OS_Q) * SAMPLE_OPS + SPREAD_OPS + 2 * FN) + len(OS_Q) * 4 * FN)
