"""How much of the family-stratified lattice's score is its particular lattice, on the bench suite and the held-out suite?

az.estimators.lattice with its offset (Roberts' 1/2) replaced by 16 random Cranley-Patterson shifts (seeded; this is an
experiment, not the estimator), at budgets from 3e5 to 1e10, on the bench suite and on the held-out suite of
experiments/lattice_validation.py (noise seeds 5000 + i). Reported: the shipped score, then the mean, median and range
of mse/samp over the shifts and how many shifts lose to sampling.

Finding: research/angles/family-lattice.md
"""
import numpy as np
import torch as th

from az.bench import suite, truth
from az.estimators import lattice as L
from az.model import DATA, NAMES, SAMPLE_OPS, load, logits, noisy
from az.ops import FN

_, BENCH = suite(0.002, 32)
HELD = [w for i, name in enumerate(NAMES) for w in noisy(load(name), 0.002, 32, seed=5000 + i)]
SUITES = {"bench": (BENCH, truth(0.002, 32)), "held-out": (HELD, np.load(DATA / "cache" / "val_suite_truth.npy"))}
PER_NODE = SAMPLE_OPS + L.NODE_FN * FN


def ratio(ws, P, budget, shift):
    nodes = budget / PER_NODE
    X, q = L.inputs(*L.strata(budget, PER_NODE), shift) if nodes >= 720 else L.latin_inputs(max(1, int(nodes // 90)), shift)
    est = [sum((logits([w], X[c : c + 2**20][None])[0].argmax(1) == q[c : c + 2**20]).sum().item() for c in range(0, len(X), 2**20)) / len(X) for w in ws]
    L.inputs.cache_clear()
    L.latin_inputs.cache_clear()
    return np.mean((np.array(est) - P) ** 2) / (np.mean(P * (1 - P)) * SAMPLE_OPS / (len(X) * PER_NODE))


th.set_grad_enabled(False)
for name, (ws, P) in SUITES.items():
    print(f"{name} suite")
    for b in (3e5, 1e6, 3e6, 1e7, 3e7, 1e8, 1e9, 1e10):
        shipped = ratio(ws, P, b, (0.5,) * 10)
        r = np.array([ratio(ws, P, b, tuple(np.random.default_rng(s).random(10))) for s in range(16)])
        print(f"  budget {b:.0e}: shipped {shipped:.2f}; 16 shifts mean {r.mean():.2f}, median {np.median(r):.2f}, "
              f"range {r.min():.2f}-{r.max():.2f}, above 1: {(r > 1).sum()}", flush=True)
