"""With the full permutation of the inputs fixed, how many activation patterns remain?

Finding: research/findings/linearity.md
"""
import torch as th

from az.model import load, trace
from experiments.common import DEV

w = load("example")
g = th.Generator(device=DEV).manual_seed(5)
K = 10000
for trial in range(20):
    pi = th.randperm(10, generator=th.Generator().manual_seed(trial)).to(DEV)  # pi[rank] = position
    y = th.randn(K, 10, generator=g, device=DEV, dtype=th.float64).sort(-1, descending=True).values
    x = th.empty_like(y)
    x[:, pi] = y
    L, pre, _ = trace(w, x)
    P = (pre > 0).reshape(K, -1)
    npos = (y > 0).sum(1)
    det = 0.0
    for k in range(11):
        m = npos == k
        if m.any():
            f = P[m].double().mean(0)
            det += th.maximum(f, 1 - f).mean().item() * m.sum().item()
    print(f"pi={pi.tolist()} distinct patterns {th.unique(P, dim=0).shape[0]}  bit determinism given pi and sign count {det / K:.4f}"
          f"  accuracy {(L.argmax(-1) == pi[1]).double().mean().item():.3f}")
