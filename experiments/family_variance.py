"""How much does accuracy vary across the 720 families (positions of the top three), and what explains it?

Twelve suite models (copies 0 and 1 of each base); per family, the error rate from 2^26 Gaussian inputs binned by the
family. Reported: the sd of the per-family error rate, the share of the indicator's variance that lies between families,
and how much of the across-family variance main effects of p, q, r explain, then the (p, q) pair, then all pairs.
This is the variance that enumerating families removes and that balanced family subsets must capture.

Finding: research/findings/errors.md
"""
import numpy as np
import torch as th

from az.bench import suite
from az.estimators.gauss_margin import FAMS
from az.model import logits
from experiments.common import DEV

labels, ws = suite(0.002, 32)
pick = [b * 32 + j for b in range(6) for j in range(2)]
F = np.array(FAMS)
fidx = th.full((1000,), -1, dtype=th.long, device=DEV)
for k, (p, q, r) in enumerate(FAMS):
    fidx[p * 100 + q * 10 + r] = k

g = th.Generator(device=DEV).manual_seed(123)
err = th.zeros(len(pick), 720, dtype=th.float64, device=DEV)
cnt = th.zeros(720, dtype=th.float64, device=DEV)
models = [ws[i] for i in pick]
for _ in range(2**26 // 2**21):
    x = th.randn(2**21, 10, generator=g, device=DEV, dtype=th.float64)
    top = x.topk(3, 1).indices
    f = fidx[top[:, 0] * 100 + top[:, 1] * 10 + top[:, 2]]
    cnt.index_add_(0, f, th.ones(2**21, dtype=th.float64, device=DEV))
    err.index_add_(1, f, (logits(models, x[None].expand(len(models), -1, -1)).argmax(-1) != top[:, 1]).double())
rate = (err / cnt).cpu().numpy()


def design(cols):
    X = [np.ones((720, 1))]
    for c in cols:
        key = F[:, c[0]] * 10 + F[:, c[1]] if isinstance(c, tuple) else F[:, c]
        X.append((key[:, None] == np.unique(key)[None, 1:]).astype(float))
    return np.concatenate(X, 1)


print(f"{'model':10s} {'err':>6s} {'sd over families':>17s} {'between-family share':>21s}   R2: p+q+r, (p,q), all pairs")
for k, i in enumerate(pick):
    y = rate[k]
    pbar = y.mean()
    r2 = [1 - np.var(y - X @ np.linalg.lstsq(X, y, rcond=None)[0]) / np.var(y)
          for X in (design([0, 1, 2]), design([(0, 1)]), design([(0, 1), (0, 2), (1, 2)]))]
    print(f"{labels[i]:10s} {pbar:6.3f} {y.std():17.3f} {y.var() / (pbar * (1 - pbar)):21.2f}   " + ", ".join(f"{v:.2f}" for v in r2))
