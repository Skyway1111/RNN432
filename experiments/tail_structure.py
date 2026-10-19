"""Where does the confident-error tail live, the rare event the estimator yardstick scores?

Level c: max_{j != q} l_j - l_q >= c |x| (az/rare.py). On 2^28 Gaussian inputs, one copy of each base model: the
probability at each level, and for the events at c >= 4 their share by the number of positive inputs (at most one
positive holds 1.1% of the mass), by the rank of the predicted position (1 = the max), and by the max's position.

Finding: research/findings/errors.md
"""
import numpy as np
import torch as th

from az.bench import suite
from az.rare import LEVELS

labels, ws = suite(0.002, 32)
g = th.Generator(device="cuda").manual_seed(11)
for i in (0, 32, 64, 96, 128, 160):
    Wi, Wh, Wo = (th.tensor(a, dtype=th.float32, device="cuda") for a in ws[i])
    hits, n = np.zeros(len(LEVELS)), 0
    pos, rank, pmax = np.zeros(11), np.zeros(10), np.zeros(10)
    for _ in range(64):
        x = th.randn(2**22, 10, generator=g, device="cuda")
        h = th.zeros(len(x), 16, device="cuda")
        for t in range(10):
            h = th.relu(h @ Wh.T + x[:, t, None] * Wi)
        L = h @ Wo.T
        top = x.topk(2, dim=1).indices
        q = top[:, 1:]
        best, arg = L.scatter(1, q, -th.inf).max(1)
        gap = (best - L.gather(1, q)[:, 0]) / x.norm(dim=1)
        hits += np.array([(gap >= c).sum().item() for c in LEVELS])
        n += len(x)
        sel = gap >= 4
        xs = x[sel]
        pos += th.bincount((xs > 0).sum(1), minlength=11).cpu().numpy()
        rank += th.bincount((xs > xs.gather(1, arg[sel, None])).sum(1), minlength=10).cpu().numpy()
        pmax += th.bincount(top[sel, 0], minlength=10).cpu().numpy()
    print(f"{labels[i]}: P(level c) for c = {', '.join(f'{c:g}' for c in LEVELS)}: " + " ".join(f"{v / n:.1e}" for v in hits))
    t = pos.sum()
    print(f"   events at c >= 4 ({int(t)}): at most one positive {pos[:2].sum() / t:.2f}; predicted rank 1, 3, 4 or lower "
          f"{rank[0] / t:.2f}, {rank[2] / t:.2f}, {rank[3:].sum() / t:.2f}; max at position 7-9 {pmax[7:].sum() / t:.2f}", flush=True)
