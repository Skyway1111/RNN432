"""Do the positions of the top ranks fix the margin's coefficients?

Finding: research/findings/linearity.md
"""
import torch as th

from az.model import load, trace
from experiments.common import group_mean, inputs, ranks

N = 2**22
x = inputs(N, seed=11)
_, _, JL = trace(load("example"), x, jacobian=True)
srt, y = ranks(x)
p, q, r, s4 = srt[:, 0], srt[:, 1], srt[:, 2], srt[:, 3]
ar = th.arange(N, device=x.device)
c = (JL[ar, q] - JL[ar, r]).gather(1, srt)  # margin l_q - l_r, coefficients by rank
del JL
keys = {"(p,q)": (p * 10 + q, 100), "(p,q,r)": ((p * 10 + q) * 10 + r, 1000), "(p,q,r,4th)": (((p * 10 + q) * 10 + r) * 10 + s4, 10000)}
for name, (key, n) in keys.items():
    within = ((c - group_mean(c, key, n)) ** 2).mean(0)
    print(name, "share of coefficient variance explained, ranks 1-6:", (1 - within / c.var(0))[:6].cpu().numpy().round(3))
key, n = keys["(p,q,r,4th)"]
m, mhat = (c * y).sum(1), (group_mean(c, key, n) * y).sum(1)
print("margin R2 with group-mean coefficients:", 1 - ((m - mhat) ** 2).mean().item() / m.var().item(),
      " sign agreement:", ((m > 0) == (mhat > 0)).double().mean().item())
