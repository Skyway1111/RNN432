"""Given the positions of the top ranks, are the logits linear in the order statistics?

Finding: research/findings/linearity.md
"""
import torch as th

from az.model import load, trace
from experiments.common import inputs, ranks

N = 2**23
x = inputs(N, seed=2)
L, _, _ = trace(load("example"), x)
srt, y = ranks(x)
p, q, r = srt[:, 0], srt[:, 1], srt[:, 2]
Lc = L - L.mean(-1, keepdim=True)


def fit(F, key, n, name):
    res = tot = hit = 0
    for c in range(n):
        m = key == c
        if m.sum() < 5 * F.shape[1]:
            continue
        sol = th.linalg.lstsq(F[m], Lc[m]).solution
        pr = F[m] @ sol
        res += ((Lc[m] - pr) ** 2).sum().item()
        tot += ((Lc[m] - Lc[m].mean(0)) ** 2).sum().item()
        hit += (pr.argmax(-1) == q[m]).sum().item()
    print(f"{name}: R2 {1 - res / tot:.4f}  surrogate accuracy {hit / N:.4f}")


relu = th.relu
pq, pqr = p * 10 + q, (p * 10 + q) * 10 + r
fit(y[:, :2], pq, 100, "(p,q) | M,S")
fit(y[:, :3], pq, 100, "(p,q) | M,S,T")
fit(th.cat([y[:, :3], relu(y[:, :3])], 1), pq, 100, "(p,q) | M,S,T and ReLUs")
fit(th.cat([y, relu(y)], 1), pq, 100, "(p,q) | all sorted values and ReLUs")
fit(th.cat([y[:, :3], relu(y[:, :3])], 1), pqr, 1000, "(p,q,r) | M,S,T and ReLUs")
fit(th.cat([y, relu(y)], 1), pqr, 1000, "(p,q,r) | all sorted values and ReLUs")
fit(th.cat([y[:, :3], relu(y[:, :3]), x[:, 9:], relu(x[:, 9:])], 1), pqr, 1000, "(p,q,r) | M,S,T, last input, ReLUs")
print("true accuracy", (L.argmax(-1) == q).double().mean().item())
