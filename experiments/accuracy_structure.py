"""Where does the example model err: by position, by sign count, by tie gaps? How diverse are its activation patterns?

Finding: research/findings/errors.md, research/findings/linearity.md
"""
import collections

import numpy as np

from az.model import load, trace
from experiments.common import inputs, ranks

np.set_printoptions(precision=3, linewidth=200, suppress=True)
x = inputs(2**21, seed=1)
L, pre, _ = trace(load("example"), x)
srt, y = ranks(x)
p, q = srt[:, 0], srt[:, 1]
pred = L.argmax(-1)
ok = pred == q
print("accuracy", ok.double().mean().item())
acc = np.zeros((10, 10))
for i in range(10):
    for j in range(10):
        m = (p == i) & (q == j)
        acc[i, j] = ok[m].double().mean().item() if m.any() else 0
print("accuracy by (max position row, 2nd position col)\n", acc)
wrong = ~ok
print("errors predicting the max:", (pred[wrong] == p[wrong]).double().mean().item(),
      " the 3rd:", (pred[wrong] == srt[wrong, 2]).double().mean().item())
npos = (x > 0).sum(-1)
for k in range(11):
    m = npos == k
    print(f"positives {k}: mass {m.double().mean().item():.4f} accuracy {ok[m].double().mean().item():.4f}")
for name, gap in [("M-S", y[:, 0] - y[:, 1]), ("S-T", y[:, 1] - y[:, 2])]:
    for lo, hi in [(0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.5), (0.5, 9)]:
        m = (gap >= lo) & (gap < hi)
        print(f"{name} in [{lo},{hi}): mass {m.double().mean().item():.3f} accuracy {ok[m].double().mean().item():.4f}")
bits = np.packbits((pre[: 2**20] > 0).reshape(2**20, -1).cpu().numpy(), axis=1)
cnt = collections.Counter(r.tobytes() for r in bits)
v = np.sort(np.array(list(cnt.values())))[::-1]
cs = np.cumsum(v) / v.sum()
print("distinct 160-bit patterns in 2^20 inputs:", len(cnt))
for k in [1, 10, 100, 1000, 10000, 100000]:
    print(f"  top {k} patterns cover {cs[k - 1]:.4f}")
print("firing rate (rows neuron, cols step)\n", (pre > 0).double().mean(0).T.cpu().numpy())
