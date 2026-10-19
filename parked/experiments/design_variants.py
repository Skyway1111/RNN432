"""Which deterministic design over the seven lower inputs gives gauss_margin's moments, and at what number of runs?

At every node of gauss_margin's (7, 8) grid and every family, the margin's mean and sd over the lower inputs (i.i.d. N(0,1)
truncated above at T) from a design, then Phi(-mu / sd), mass-weighted. Compared with the empirical P(wrong) at the same nodes
from 1024 truncated-normal draws, so the grid is common and the error is the presumption plus the design. Designs (runs):
  H8 (8):            the estimator's Hadamard at mean +- sd, sd with the 1/(n-1) divisor; every other design uses 1/n, which
                     is exact for a linear margin;
  H8 pop (8), fold16 (16), full 2^7 (128):  two-level designs at mean +- sd;
  H8 jitter (8):     H8 with column offsets -0.3 ... +0.3 sd, so no two lower inputs tie;
  OA27 tertile (27): orthogonal array OA(27, 7, 3, 2) from GF(3), levels at the tertile means scaled to the variance;
  Gauss 3^7 (2187):  the tensor 3-point Gauss rule of the truncated normal, product weights (exact to degree 5 per input);
  Korobov n (n):     rank-1 lattices in quantile space, u = ((k g_i) mod n + 1/2) / n, g = (1, a, a^2, ...): each input
                     takes n distinct values;
  oracle (1024):     Phi(-mu / sd) with the draws' own moments, the presumption alone.
Also the median ratio of design sd to empirical sd over (node, family) pairs with 0.02 < P(wrong) < 0.98.
Six suite models, copy 0 of each base. Reported: estimate - empirical accuracy per model, then mean and rms.

Finding: research/angles/margin-gaussian.md
"""
import itertools

import numpy as np
import torch as th
from scipy.linalg import hadamard
from scipy.stats import norm

from az.bench import suite
from az.estimators.gauss_margin import FAMS, FREE, Q, quadrature
from az.model import logits
from experiments.common import DEV

th.set_grad_enabled(False)
DRAWS = 1024
labels, ws = suite(0.002, 32)
pick = [i * 32 for i in range(6)]
pos, freet, qt = (th.tensor(a, device=DEV) for a in (FAMS, FREE, Q))
H8 = hadamard(8)[:, 1:].astype(float)
COLS = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 2, 0), (1, 0, 1), (1, 1, 1)]  # 7 projective points of PG(2, 3)
OA27 = np.array([[np.dot(r, c) % 3 for c in COLS] for r in itertools.product(range(3), repeat=3)])  # (27, 7), levels 0..2
TENSOR3 = np.array(list(itertools.product(range(3), repeat=7)))


def korobov(n, a):
    return ((np.arange(n)[:, None] * np.array([pow(a, i, n) for i in range(7)])) % n + 0.5) / n


def trunc(t):
    """Mean, sd, standardized tertile means scaled to unit variance, and the 3-point Gauss rule of N(0,1) truncated above at t."""
    a = norm.pdf(t) / norm.cdf(t)
    mu, sd = -a, np.sqrt(1 - t * a - a * a)
    e = norm.ppf(np.array([0, 1 / 3, 2 / 3, 1]) * norm.cdf(t))
    y = ((norm.pdf(e[:-1]) - norm.pdf(e[1:])) / (norm.cdf(e[1:]) - norm.cdf(e[:-1])) - mu) / sd
    z = np.linspace(-9, t, 40001)
    z = (z[1:] + z[:-1]) / 2
    wz = norm.pdf(z) / norm.pdf(z).sum()
    alpha, beta, p_prev, p, n_prev = [], [], 0 * z, 1 + 0 * z, 1.0  # Stieltjes recurrence, then Golub-Welsch
    for k in range(3):
        nk = (wz * p * p).sum()
        alpha.append((wz * z * p * p).sum() / nk)
        if k:
            beta.append(nk / n_prev)
        p_prev, p, n_prev = p, (z - alpha[k]) * p - (beta[-1] * p_prev if k else 0), nk
    nodes, vec = np.linalg.eigh(np.diag(alpha) + np.diag(np.sqrt(beta), 1) + np.diag(np.sqrt(beta), -1))
    return mu, sd, y / np.sqrt(np.mean(y * y)), nodes, vec[0] ** 2


def margins(w, t, g1, g2, low):
    """(families, runs) margins with the lower inputs low (runs, 7) or (families, runs, 7)."""
    R = low.shape[-2]
    x = th.zeros(len(FAMS), R, 10, dtype=th.float64, device=DEV)
    x.scatter_(2, freet[:, None].expand(-1, R, -1), low.expand(len(FAMS), R, 7).contiguous())
    top = th.tensor([t + g2 + g1, t + g2, t], dtype=th.float64, device=DEV)
    x.scatter_(2, pos[:, None].expand(-1, R, -1), top.expand(len(FAMS), R, 3).contiguous())
    L = logits([w], x.reshape(1, -1, 10))[0]
    q = qt.repeat_interleave(R)
    other = L.clone()
    other[th.arange(len(L), device=DEV), q] = -1e9
    return (L[th.arange(len(L), device=DEV), q] - other.max(1).values).reshape(len(FAMS), R)


def moments(m, wt=None, correction=0):
    if wt is None:
        return m.mean(1), m.std(1, correction=correction).clamp(min=1e-9)
    mu = (m * wt).sum(1)
    return mu, (((m - mu[:, None]) ** 2) * wt).sum(1).sqrt().clamp(min=1e-9)


def designs(t):
    """name -> (lower inputs (runs, 7), weights or None, sd divisor correction)."""
    mu, sd, ter, gn, gw = trunc(t)
    f = lambda a: th.tensor(a, dtype=th.float64, device=DEV)  # noqa: E731
    out = {"H8": (f(mu + sd * H8), None, 1), "H8 pop": (f(mu + sd * H8), None, 0),
           "fold16": (f(mu + sd * np.vstack([H8, -H8])), None, 0),
           "full 2^7": (f(mu + sd * np.array(list(itertools.product([-1.0, 1.0], repeat=7)))), None, 0),
           "H8 jitter": (f(mu + sd * (H8 + np.linspace(-0.3, 0.3, 7))), None, 0),
           "OA27 tertile": (f(mu + sd * ter[OA27]), None, 0),
           "Gauss 3^7": (f(gn[TENSOR3]), f(np.prod(gw[TENSOR3], 1)), 0)}
    for n, a in [(13, 6), (31, 12), (61, 17)]:
        out[f"Korobov {n}"] = (f(norm.ppf(korobov(n, a) * norm.cdf(t))), None, 0)
    return out


mass, nodes = quadrature(7, 8)
names = list(designs(0.5)) + ["oracle"]
gen = th.Generator(device=DEV).manual_seed(9)
res, ratio = [], []
for i in pick:
    w = ws[i]
    acc = np.zeros(len(names) + 1)
    rat = {k: [] for k in names}
    for c, (t, g1, g2) in enumerate(nodes):
        u = th.rand(len(FAMS), DRAWS, 7, generator=gen, device=DEV, dtype=th.float64)
        me = margins(w, t, g1, g2, th.special.ndtri(u * norm.cdf(t)))
        pe = (me < 0).double().mean(1)
        sel = (pe > 0.02) & (pe < 0.98)
        mo = {k: moments(margins(w, t, g1, g2, low), wt, cor) for k, (low, wt, cor) in designs(t).items()}
        mo["oracle"] = moments(me, correction=1)
        for k, (mu, sd) in mo.items():
            rat[k].append((sd / mo["oracle"][1])[sel])
        acc += mass[c] * np.array([th.special.ndtr(-mu / sd).mean().item() for mu, sd in mo.values()] + [pe.mean().item()])
    err = acc[-1] - acc[:-1]
    res.append(err)
    ratio.append([th.cat(rat[k]).median().item() for k in names])
    print(f"{labels[i]:10s} empirical acc {1 - acc[-1]:.4f}  " + " ".join(f"{n} {e:+.4f}" for n, e in zip(names, err)), flush=True)
res, ratio = np.array(res), np.array(ratio)
print("\nestimate - empirical accuracy at the (7, 8) nodes over 6 models: mean, rms; median sd ratio design/empirical")
for k, n in enumerate(names):
    print(f"  {n:14s} {res[:, k].mean():+.5f} {np.sqrt((res[:, k] ** 2).mean()):.5f}   {ratio[:, k].mean():.3f}")
