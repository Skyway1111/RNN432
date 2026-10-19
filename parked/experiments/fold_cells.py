"""Can coarse cells work if each cell's own spread in (T, g1, g2) is folded into the design, and how close does that come?

The fold: per cell, a 12-run Plackett-Burman design over ten factors, (T, g1, g2) at the cell's exact mean +- the columns of
its covariance's Cholesky factor (gaps clipped at 0), the seven lower inputs at mean +- sd of N(0,1) truncated above at that
run's T; P(wrong) = Phi(-mu / sd) with the 1/n sd, mass-weighted over cells. One T cell; gap cells 3 x 3 with edges
(0, 0.05, 0.3, inf), and 5 x 5 with (0, 0.02, 0.06, 0.15, 0.4, inf). Families: all 720, or the 90 of an idempotent Latin
square (p, q, L[p][q]), which balance every pair of positions. Scored as az.bench does, on the whole suite.
sharing_limits measures the presumption alone over the same 3 x 3 and 5 x 5 cells.

Finding: research/angles/margin-gaussian.md
"""
import numpy as np
import torch as th

from az.bench import suite, truth
from az.estimators.gauss_margin import FAMS, FREE, Q, cells, truncated
from az.model import SAMPLE_OPS, logits
from experiments.common import DEV

th.set_grad_enabled(False)
labels, ws = suite(0.002, 32)
p_true = truth(0.002, 32)
gen = [1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1]
PB12 = np.array([np.roll(gen, k) for k in range(11)] + [[-1] * 11], float)[:, :10]
inf = np.inf


def latin():
    L = [[a if a == b else None for b in range(10)] for a in range(10)]
    todo = [(a, b) for a in range(10) for b in range(10) if a != b]

    def fill(k):
        if k == len(todo):
            return True
        a, b = todo[k]
        for v in range(10):
            if all(L[a][j] != v for j in range(10)) and all(L[j][b] != v for j in range(10)):
                L[a][b] = v
                if fill(k + 1):
                    return True
                L[a][b] = None
        return False

    assert fill(0)
    return [FAMS.index((a, b, L[a][b])) for a in range(10) for b in range(10) if a != b]


def fold_inputs(ge, fams):
    """Mass per cell and inputs (cells, families, 12, 10) of the fold design."""
    mass, mean, cov = cells(np.array([-inf, inf]), np.array(ge))
    chol = th.linalg.cholesky(th.tensor(cov, device=DEV) + 1e-12 * th.eye(3, device=DEV, dtype=th.float64))
    tg = th.tensor(mean, device=DEV)[:, None] + th.einsum("rk,cjk->crj", th.tensor(PB12[:, :3], device=DEV), chol)
    t, g1, g2 = tg[..., 0], tg[..., 1].clamp(min=0), tg[..., 2].clamp(min=0)
    mu, sd = truncated(t)
    roles = th.cat([th.stack([t + g2 + g1, t + g2, t], -1), mu[..., None] + sd[..., None] * th.tensor(PB12[:, 3:], device=DEV)], -1)
    order = th.cat([th.tensor(FAMS, device=DEV), th.tensor(FREE, device=DEV)], 1)[fams]  # position of each role
    X = th.zeros(len(mass), len(fams), 12, 10, dtype=th.float64, device=DEV)
    X.scatter_(3, order[None, :, None].expand(len(mass), -1, 12, -1), roles[:, None].expand(-1, len(fams), -1, -1).contiguous())
    return mass, X


def estimate(w, mass, X, fams):
    L = logits([w], X.reshape(1, -1, 10))[0].reshape(X.shape)
    q = th.tensor(Q[fams], device=DEV)[None, :, None, None].expand(*X.shape[:3], 1)
    m = L.gather(3, q)[..., 0] - L.scatter(3, q, -1e9).max(3).values
    p = th.special.ndtr(-m.mean(2) / m.std(2, correction=0).clamp(min=1e-9))
    return 1 - float((th.tensor(mass, device=DEV)[:, None] * p).sum() / len(fams))


print(f"{'variant':34s} {'ops':>9s} {'rmse':>8s} {'bias':>8s} {'rmse_samp':>9s} {'mse/samp':>8s}")
for name, ge in [("3 x 3", [0, 0.05, 0.3, inf]), ("5 x 5", [0, 0.02, 0.06, 0.15, 0.4, inf])]:
    for fname, fams in [("all 720 families", list(range(720))), ("Latin 90 families", latin())]:
        mass, X = fold_inputs(ge, fams)
        e = np.array([estimate(w, mass, X, fams) for w in ws])
        ops = X.shape[0] * X.shape[1] * X.shape[2] * SAMPLE_OPS
        mse, mse_s = np.mean((e - p_true) ** 2), np.mean(p_true * (1 - p_true)) / (ops // SAMPLE_OPS)
        print(f"{'gaps ' + name + ', ' + fname:34s} {ops:9.2e} {mse ** 0.5:8.5f} {np.mean(e - p_true):+8.5f} {mse_s ** 0.5:9.5f} {mse / mse_s:8.2f}", flush=True)
