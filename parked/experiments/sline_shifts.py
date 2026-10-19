"""How much of the S-line lattice's score is the particular lattice, and what should one expect from an untuned one?

The shipped sequence (Roberts' R9 with offset 1/2) was adopted after an earlier lattice plateaued on this same suite, so
its scores are selected. Here the offset is replaced by random Cranley-Patterson shifts (seeded; this is an experiment,
not the estimator) and the whole suite is scored at four levels. Reported: the shipped score, then the mean and range of
mse/samp over the shifts and how many lose to sampling. The truth's own noise (2^24 inputs per model) inflates the higher
levels' MSE.

Finding: research/angles/family-lattice.md
"""
import numpy as np
import torch as th

from az.bench import suite, truth
from az.estimators import sline as S
from az.model import SAMPLE_OPS
from experiments.common import DEV

labels, ws = suite(0.002, 32)
P = truth(0.002, 32)


def score(kt, shift):
    X, q, T, M = S.nodes(kt, kt, shift)
    est, ops = [], []
    for w in ws:
        W = [th.tensor(a, device=DEV, dtype=th.float64) for a in w]
        p, o = zip(*(S.sline(*W, X[c : c + 2**17], q[c : c + 2**17], T[c : c + 2**17], M[c : c + 2**17]) for c in range(0, len(X), 2**17)))
        est.append(th.cat(p).mean().item())
        ops.append(th.cat(o).sum().item())
    S.nodes.cache_clear()
    return np.mean((np.array(est) - P) ** 2) / (np.mean(P * (1 - P)) * SAMPLE_OPS / np.mean(ops)), np.mean(ops)


for kt, n_shifts in ((2, 8), (3, 8), (6, 6), (12, 4)):
    shipped, ops = score(kt, (0.5,) * 9)
    r = np.array([score(kt, tuple(np.random.default_rng(s).random(9)))[0] for s in range(n_shifts)])
    print(f"{kt} x {kt}: ops {ops:.2e}, shipped mse/samp {shipped:.2f}; {n_shifts} random shifts: mean {r.mean():.2f}, "
          f"range {r.min():.2f}-{r.max():.2f}, above 1: {(r > 1).sum()}", flush=True)
