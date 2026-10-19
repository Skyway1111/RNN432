"""Which input ranks carry the decision margins l_q - l_r and l_q - l_p? (Exact: margin = sum of coefficient x value, by homogeneity.)

Finding: research/findings/linearity.md
"""
import numpy as np
import torch as th

from az.model import load, trace
from experiments.common import inputs, ranks

np.set_printoptions(precision=3, suppress=True, linewidth=180)
N = 2**20
x = inputs(N, seed=4)
L, _, JL = trace(load("example"), x, jacobian=True)
srt, y = ranks(x)
p, q, r = srt[:, 0], srt[:, 1], srt[:, 2]
ar = th.arange(N, device=x.device)
for name, (a, b) in {"l_q - l_r": (q, r), "l_q - l_p": (q, p)}.items():
    c = (JL[ar, a] - JL[ar, b]).gather(1, srt)
    print(name, "coefficient mean by rank", c.mean(0).cpu().numpy())
    print(name, "coefficient sd by rank  ", c.std(0).cpu().numpy())
    print(name, "contribution sd by rank ", (c * y).std(0).cpu().numpy())
c = (JL[ar, q] - JL[ar, r]).gather(1, srt)
print("Euler check, max |margin - sum|:", ((L[ar, q] - L[ar, r]) - (c * y).sum(1)).abs().max().item())
