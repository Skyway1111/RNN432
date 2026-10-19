"""Structure-blind control for modes: the same pipeline with full-covariance Gaussian components in place of gap proposals.

The cross-entropy-optimal proposal within a family is its maximum-likelihood fit to points of the event, so per base and
level the library is a Gaussian mixture of K[level] components fitted by EM to the same events modes fits
(modes.events(): fresh noisy copies of each base, radii drawn chi_10). Nothing else changes: BETA of the nominal law,
level shares, the balance heuristic, fixed counts on a shifted Kronecker lattice seeded by the copy's weights, and a first
stage of ADAPT whose soft events re-estimate each level's component weights, the generic analog of modes' order law.
A Gaussian's angular density is closed form too: along the ray the exponent is -Q r^2 / 2 + B r + c0 with Q = u'Pu and
B = u'Pm, so the radial integral is the same K9 as for gap proposals (az/osp.py).
Ops per point: the forward pass and |x| (TAIL_OPS), 10 inverse normal CDFs and the triangular map (DRAW_OPS), the final
log and exp, and per component the symmetric quadratic form, a dot product, a log, a root and the K9 lookup.
"""
import math
import zlib
from functools import cache

import numpy as np
import torch as th

from az import osp
from az.estimators.lattice import roberts
from az.estimators.modes import base_of, counts, events, margin
from az.ops import FN
from az.rare import LEVELS, TAIL_OPS

K = {2.0: 16, 4.0: 8, 6.0: 8, 8.0: 8}  # components per level, chosen on the held-out suite (experiments/ce_score.py)
SHARE = {2.0: 0.6, 4.0: 0.2, 6.0: 0.1, 8.0: 0.1}
BETA, SALT = 0.2, 0
ADAPT, SOFT, PRIOR = 0.3, 1.0, 1.0
RIDGE, ITERS = 1e-3, 100  # covariance floor and EM iterations
DRAW_OPS, FINAL_OPS, COMP_OPS = 85 + 11 * FN, 2 * FN, 75 + 3 * FN
LEARN_OPS = 3 + FN
DEV = osp.DEV
CHUNK = 2**20


def outer(x: th.Tensor) -> th.Tensor:
    return (x[:, :, None] * x[:, None]).reshape(len(x), 100)


class Gauss:
    """K Gaussian components in R^10, stacked so that every density is two GEMMs over x and x (x) x, whatever K is."""

    def __init__(self, m: th.Tensor, cov: th.Tensor):
        self.m, self.C = m, th.linalg.cholesky_ex(cov).L  # no error check, so no sync per EM step; the ridge keeps cov PD
        P = th.cholesky_inverse(self.C)
        self.P = P.reshape(len(m), 100).T.contiguous()  # (100, K)
        self.Pm = (P @ m[:, :, None])[..., 0].T.contiguous()  # (10, K)
        self.c0 = -5 * math.log(2 * math.pi) - self.C.diagonal(dim1=1, dim2=2).log().sum(1) - 0.5 * (m * self.Pm.T).sum(1)

    def __len__(self):
        return len(self.m)

    def log_x(self, x: th.Tensor, xx: th.Tensor) -> th.Tensor:
        """(N, K) log density at points x, given xx = outer(x)."""
        return self.c0 + x @ self.Pm - 0.5 * (xx @ self.P)

    def log_ang(self, u: th.Tensor) -> th.Tensor:
        """(N, K) log angular density w.r.t. surface measure at unit directions u."""
        Q = outer(u) @ self.P
        return self.c0 - 5 * th.log(Q) + osp.logk9((u @ self.Pm) / th.sqrt(Q))

    def place(self, part: th.Tensor, z: th.Tensor) -> th.Tensor:
        """Points m + C z of parts part (sorted; 0 is the nominal law, part k component k - 1), one GEMM per part present."""
        x = z.clone()
        edges = th.searchsorted(part, th.arange(len(self) + 2, device=DEV)).tolist()
        for k in range(1, len(self) + 1):
            a, b = edges[k], edges[k + 1]
            if b > a:
                x[a:b] = self.m[k - 1] + z[a:b] @ self.C[k - 1].T
        return x


def em(x: th.Tensor, k: int, seed: int) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
    """Means, covariances and weights of a k-component mixture fitted to x by EM, means started at k distinct points of x.
    The fit runs in float32, 30x faster on consumer GPUs; the fitted law is then exact in float64, so this costs no bias."""
    gen = th.Generator(device=DEV).manual_seed(seed)
    x = x.float()
    m = x[th.randperm(len(x), generator=gen, device=DEV)[:k]]
    cov = th.cov(x.T).expand(k, 10, 10).clone()
    al = th.full((k,), 1 / k, device=DEV)
    xx = outer(x)
    ridge = RIDGE * th.eye(10, device=DEV)
    for _ in range(ITERS):
        r = th.softmax(th.log(al) + Gauss(m, cov).log_x(x, xx), 1)
        n = r.sum(0)
        al, m = n / len(x), (r.T @ x) / n[:, None]
        cov = (r.T @ xx).view(k, 10, 10) / n[:, None, None] - m[:, :, None] * m[:, None] + ridge
    return m.double(), cov.double(), al.double()


@cache
def mixture(b: int):
    """The base's components across levels, their weights in the mixture (summing to 1 - BETA), each one's index into
    LEVELS, and its library weight within its level."""
    gen = th.Generator(device=DEV).manual_seed(b)
    ms, covs, alphas, lvl = [], [], [], []
    for c, u in events()[b].items():
        u = u.to(DEV)
        m, cov, al = em(u * th.randn(len(u), 10, generator=gen, device=DEV, dtype=th.float64).norm(dim=1)[:, None], K[c], seed=b)
        ms.append(m), covs.append(cov), alphas.append(al)
        lvl.append(th.full((K[c],), LEVELS.index(c), device=DEV))
    lvl, lib = th.cat(lvl), th.cat(alphas)
    share = th.tensor([SHARE.get(c, 0.0) for c in LEVELS], dtype=th.float64, device=DEV) / sum(SHARE.values())
    return Gauss(th.cat(ms), th.cat(covs)), (1 - BETA) * share[lvl] * lib, lvl, lib


def tail(w, budget, learned=False):
    """Estimates per level and ops; with learned, also the second stage's mixture (components, weights)."""
    comps, alphas, lvl, lib = mixture(base_of(w))
    W = [th.tensor(a, dtype=th.float32, device=DEV) for a in w]
    per = TAIL_OPS + DRAW_OPS + FINAL_OPS + COMP_OPS * len(comps) + LEARN_OPS * len(comps) * ADAPT
    n = max(int(budget // per), 2 * (len(comps) + 1))
    gen = th.Generator(device=DEV).manual_seed(zlib.crc32(np.ascontiguousarray(w.Wh).tobytes()) + SALT)
    shifts = th.rand(len(comps) + 1, 10, generator=gen, device=DEV, dtype=th.float64)
    step = th.tensor(roberts(10), dtype=th.float64, device=DEV)
    lv = th.tensor(LEVELS, dtype=th.float64, device=DEV)
    beta = th.tensor([BETA], dtype=th.float64, device=DEV)

    def stage(lo, m, learn):
        """m_j points of every part j (the nominal law first) from lattice index lo_j on, weighted by the balance
        heuristic over the parts' actual shares; returns the event sums and, with learn, each component's soft mass."""
        N = int(m.sum())
        logp = th.log(m.double() / N)
        part = th.repeat_interleave(th.arange(len(m), device=DEV), m)
        idx = th.arange(N, device=DEV) - th.repeat_interleave(m.cumsum(0) - m - lo, m)
        out = th.zeros(len(LEVELS), dtype=th.float64, device=DEV)
        mass = th.zeros(len(comps), dtype=th.float64, device=DEV)
        for a in range(0, N, CHUNK):
            j = part[a : a + CHUNK]
            z = th.special.ndtri(((shifts[j] + idx[a : a + CHUNK, None].double() * step) % 1.0).clamp(1e-15, 1 - 1e-15))
            x = comps.place(j, z)
            u = x / x.norm(dim=1, keepdim=True)
            lq = th.cat([(logp[:1] - math.log(osp.AREA)).expand(len(u), 1), logp[1:] + comps.log_ang(u)], 1)
            lse = th.logsumexp(lq, 1)
            lr = th.exp(-math.log(osp.AREA) - lse)
            f = margin(W, u)
            if learn:
                mass += (lr[:, None] * th.exp(lq[:, 1:] - lse[:, None]) * th.special.ndtr((f[:, None] - lv[lvl]) / SOFT)).sum(0)
            out += ((f[:, None] >= lv) * lr[:, None]).sum(0)
        return out, mass

    cnt = counts(th.cat([beta, alphas]), n)
    if not ADAPT:
        out = (stage(th.zeros_like(cnt), cnt, False)[0] / n).cpu().numpy(), n * per + 6 * 432
        return (*out, (comps, alphas)) if learned else out
    first = (cnt * ADAPT).round().long()
    s, mass = stage(th.zeros_like(cnt), first, True)
    # each level's weights from its soft mass plus PRIOR times that mass of the library's; levels keep their share
    tot = th.zeros(len(LEVELS), dtype=th.float64, device=DEV).index_add_(0, lvl, mass)
    new = mass + PRIOR * tot[lvl] * lib
    norm = th.zeros(len(LEVELS), dtype=th.float64, device=DEV).index_add_(0, lvl, new)
    al2 = th.where(tot[lvl] > 0, new / norm[lvl] * alphas / lib, alphas)
    s2, _ = stage(first, counts(th.cat([beta, al2]), n - int(first.sum())), False)
    out = ((s + s2) / n).cpu().numpy(), n * per + 6 * 432
    return (*out, (comps, al2)) if learned else out
