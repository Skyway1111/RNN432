"""Baseline: Monte Carlo with budget // SAMPLE_OPS Gaussian inputs. Checks the analytic sampling curve of both
yardsticks: accuracy (az/bench.py) and the rare-event levels (az/rare.py, seeded by the weights)."""
import zlib

import numpy as np
import torch as th

from az.model import N_IN, SAMPLE_OPS, correct
from az.rare import TAIL_OPS, exceed

_rng = np.random.default_rng(0)


def estimate(w, budget):
    k = max(int(budget // SAMPLE_OPS), 1)
    x = th.as_tensor(_rng.standard_normal((1, k, N_IN)))
    return correct([w], x).double().mean().item(), k * SAMPLE_OPS


def tail(w, budget):
    k = max(int(budget // TAIL_OPS), 1)
    return exceed([w], k, seed=zlib.crc32(np.ascontiguousarray(w.Wh).tobytes()))[0] / k, k * TAIL_OPS
