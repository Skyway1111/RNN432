"""Given (p, q), are the logits linear in the raw positional inputs, as in ARC's handcrafted model?

Finding: research/findings/linearity.md
"""
import torch as th

from az.model import load, trace
from experiments.common import inputs, ranks

N = 2**22
x = inputs(N, seed=12)
L = trace(load("example"), x)[0]
srt, _ = ranks(x)
p, q = srt[:, 0], srt[:, 1]
Lc = L - L.mean(1, keepdim=True)
res = tot = hit = 0
for c in range(100):
    m = p * 10 + q == c
    if not m.any():
        continue
    pr = x[m] @ th.linalg.lstsq(x[m], Lc[m]).solution
    res += ((Lc[m] - pr) ** 2).sum().item()
    tot += ((Lc[m] - Lc[m].mean(0)) ** 2).sum().item()
    hit += (pr.argmax(-1) == q[m]).sum().item()
print(f"logits ~ linear in x per (p,q): R2 {1 - res / tot:.4f}  surrogate accuracy {hit / N:.4f}  true {(L.argmax(-1) == q).double().mean().item():.4f}")
