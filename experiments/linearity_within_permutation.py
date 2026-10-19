"""Within a fixed permutation, is one linear map enough, and how wrong is its accuracy?

Finding: research/findings/linearity.md
"""
import numpy as np
import torch as th

from az.model import load, trace
from experiments.common import DEV

w = load("example")
g = th.Generator(device=DEV).manual_seed(6)
K = 20000
r2s, accs, accl = [], [], []
for trial in range(30):
    pi = th.randperm(10, generator=th.Generator().manual_seed(100 + trial)).to(DEV)
    y = th.randn(K, 10, generator=g, device=DEV, dtype=th.float64).sort(-1, descending=True).values
    x = th.empty_like(y)
    x[:, pi] = y
    L = trace(w, x)[0]
    Lc = L - L.mean(1, keepdim=True)
    pr = y @ th.linalg.lstsq(y, Lc).solution
    r2s.append(1 - (((Lc - pr) ** 2).sum() / ((Lc - Lc.mean(0)) ** 2).sum()).item())
    accs.append((L.argmax(-1) == pi[1]).double().mean().item())
    accl.append((pr.argmax(-1) == pi[1]).double().mean().item())
print(f"R2 of one linear map per permutation: mean {np.mean(r2s):.4f} min {np.min(r2s):.4f}")
print(f"accuracy given permutation, linear minus true: mean {np.mean(np.array(accl) - accs):+.4f}, mean abs {np.mean(np.abs(np.array(accl) - accs)):.4f}")
