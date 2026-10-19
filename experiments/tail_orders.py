"""Under weight noise, do the tail's modes change shape or move between orders (p, q, r)?

The modes library (az/estimators/modes.py) fits each base's modes on fresh noisy copies. On held-out copies, the per-sample
gain over sampling at levels 2 and 4 of a mixture of 0.1 nominal plus that level's library components, against the same
shapes with each component's order law pi re-estimated on the copy (one EM step on orders only, from 20,000 copy events,
with 5% of the library's pi as prior), and against a full EM refit (K = 8) on those events. Then, one copy per base: the
gain at levels 2, 4 and 8 when pi is re-estimated from only n = 100, 300, 1000 events, smoothed half toward the library's
pi, as a first stage of the estimator could. Gain: sampling's relative variance per sample over the mixture's, 2^20 points.

Finding: research/findings/tail.md
"""
import torch as th

from az import osp
from az.estimators.modes import base_of, library, margin
from az.rare import LEVELS, models, truth

th.set_grad_enabled(False)
labels, ws = models("heldout")
P = truth("heldout")
DEV = osp.DEV
lib = library()


def gain(W, comps, al, p, c, n=2**20):
    gen = th.Generator(device=DEV).manual_seed(1)
    probs = th.cat([th.tensor([0.1], dtype=th.float64, device=DEV), 0.9 * al / al.sum()])
    k = th.multinomial(probs, n, replacement=True, generator=gen)
    x = th.randn(n, 10, generator=gen, device=DEV, dtype=th.float64)
    for j in range(len(comps)):
        sel = k == j + 1
        if sel.any():
            x[sel] = comps.sample(int(sel.sum()), gen, j)
    u = x / x.norm(dim=1, keepdim=True)
    h = (margin(W, u) >= c) * osp.mixture_weight(comps, probs[1:], 0.1, u)
    return ((1 - p) / p) / (((h - p) ** 2).mean() / p**2).item()


def reorder(comps, al, u, prior):
    """Each component's pi re-estimated from events u by its responsibilities, plus `prior` times its event weight of its own pi."""
    g = osp.Gaps(u)
    r = th.softmax(th.log(al) + comps.log_ang(g), 1)
    mass = th.stack([th.bincount(g.key, weights=r[:, k], minlength=osp.KEYS) for k in range(len(comps))])
    return comps.reordered(mass, prior), r.mean(0) * 0.5 + 0.5 * al / al.sum()


def copy_events(W, c, want, seed):
    g = th.Generator(device=DEV).manual_seed(seed)
    ev, n = [], 0
    while n < want and len(ev) < 128:
        z = th.randn(2**22, 10, generator=g, device=DEV, dtype=th.float64)
        u = z / z.norm(dim=1, keepdim=True)
        ev.append(u[margin(W, u) >= c])
        n += len(ev[-1])
    return th.cat(ev)[:want]


print("Part 1: library shapes and orders / library shapes, copy orders / copy refit")
for i in (0, 1, 32, 33, 64, 65, 96, 97, 128, 160, 161):
    W = [th.tensor(a, dtype=th.float32, device=DEV) for a in ws[i]]
    row = []
    for c in (2.0, 4.0):
        comps, al = lib[base_of(ws[i])][c]
        u = copy_events(W, c, 20000, i)
        nc, na = reorder(comps, al, u, prior=0.05)
        r = th.randn(len(u), 10, generator=th.Generator(device=DEV).manual_seed(i), device=DEV, dtype=th.float64).norm(dim=1)
        fc, fa = osp.em(u * r[:, None], 8)
        p = P[i][LEVELS.index(c)]
        row.append(f"c={c:g}: {gain(W, comps, al, p, c):5.2f} / {gain(W, nc, na, p, c):5.2f} / {gain(W, fc, fa, p, c):5.2f}")
    print(f"  {labels[i]:10s} " + "   ".join(row), flush=True)

print("Part 2: orders re-estimated from n events; library / n = 100, 300, 1000")
for i in range(0, 192, 32):
    W = [th.tensor(a, dtype=th.float32, device=DEV) for a in ws[i]]
    row = []
    for c in (2.0, 4.0, 8.0):
        comps, al = lib[base_of(ws[i])][c]
        u = copy_events(W, c, 1000, 100 + i)
        p = P[i][LEVELS.index(c)]
        g = [gain(W, comps, al, p, c)] + [gain(W, *reorder(comps, al, u[:n], prior=1.0), p, c) for n in (100, 300, 1000) if n <= len(u)]
        row.append(f"c={c:g}: " + " ".join(f"{v:.1f}" for v in g))
    print(f"  {labels[i]:10s} " + "   ".join(row), flush=True)
