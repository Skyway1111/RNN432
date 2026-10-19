"""Can a copy's weights predict where its tail moved, the order law of each mode, without sampling its events?

The modes library fits each base's modes on fresh noisy copies; on a held-out copy the shapes hold and the orders (p, q, r)
move (research/findings/tail.md). Per held-out copy and level (c = 2, 4), the per-sample gain over sampling of 0.1
nominal plus that level's library components, with each component's order law pi set by: the library; the oracle,
re-estimated from 20,000 of the copy's events; the library times exp(0.5 (f_copy - f_base)), f the margin at the
component's mean point placed at each of the 720 orders (720 passes per component), or at its tie point (gaps to M and T
at a tenth of their mean); and probes, 4,000 failure points of the fresh copies run on the copy, the orders re-estimated
from those still failing. Every re-estimate is smoothed half toward the library's pi. Gain: sampling's relative
variance per sample over the mixture's, on 2^20 points.

Finding: research/findings/tail.md
"""
import math

import torch as th

from az import osp
from az.estimators.modes import _bases, base_of, events, library, margin
from az.rare import LEVELS, models, truth

th.set_grad_enabled(False)
labels, ws = models("heldout")
P = truth("heldout")
DEV = osp.DEV
lib, ev = library(), events()
KEYS = th.arange(osp.KEYS, device=DEV)[osp.valid_keys()]
ORD = th.stack([KEYS // 100, KEYS // 10 % 10, KEYS % 10], 1)
M = math.sqrt(2 / math.pi)


def at_orders(comps, k, mt):
    """Component k's point with the gaps to M and T at mt times their mean and the lower gaps at theirs, at every order."""
    _, mu, _, tm, tt, tl = comps.params(k)
    T = mu - tt * mt
    x = (T - th.as_tensor(tl, device=DEV) * M).expand(720, 10).clone()
    i = th.arange(720, device=DEV)
    x[i, ORD[:, 0]], x[i, ORD[:, 1]], x[i, ORD[:, 2]] = mu + tm * mt, mu, T
    return x / x.norm(dim=1, keepdim=True)


def with_orders(comps, weights):
    """Components with pi times weight over the 720 orders, smoothed half toward their own pi."""
    new = th.zeros_like(comps.pi)
    new[:, KEYS] = comps.pi[:, KEYS] * th.stack(weights)
    return osp.OSP(new / new.sum(1, keepdim=True) + comps.pi, comps.mu, comps.sig, comps.tm, comps.tt, comps.tl)


def from_points(comps, al, u, keep):
    """Order weights from points u with keep-weights: each component's responsibility mass per order over its library pi."""
    g = osp.Gaps(u)
    r = th.softmax(th.log(al) + comps.log_ang(g), 1)
    out = []
    for k in range(len(comps)):
        m = th.bincount(g.key, weights=r[:, k] * keep, minlength=osp.KEYS)[KEYS]
        out.append(m / m.sum().clamp(min=1e-300) / comps.pi[k, KEYS].clamp(min=1e-300))
    return out


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


print("per-sample gain: library / oracle / margin at the mean point / at the tie point / probes")
for c in (2.0, 4.0):
    for i in (0, 32, 33, 64, 96, 128, 160, 161):
        W = [th.tensor(a, dtype=th.float32, device=DEV) for a in ws[i]]
        b = base_of(ws[i])
        W0 = [th.tensor(a, dtype=th.float32, device=DEV) for a in _bases()[b]]
        comps, al = lib[b][c]
        p = P[i][LEVELS.index(c)]
        g = th.Generator(device=DEV).manual_seed(i)
        found = []
        while sum(map(len, found)) < 20000:
            z = th.randn(2**22, 10, generator=g, device=DEV, dtype=th.float64)
            u = z / z.norm(dim=1, keepdim=True)
            found.append(u[margin(W, u) >= c])
        u = th.cat(found)[:20000]
        row = [gain(W, comps, al, p, c), gain(W, with_orders(comps, from_points(comps, al, u, th.ones(len(u), dtype=th.float64, device=DEV))), al, p, c)]
        for mt in (M, 0.1):
            d = [0.5 * (margin(W, at_orders(comps, k, mt)) - margin(W0, at_orders(comps, k, mt))) for k in range(len(comps))]
            row.append(gain(W, with_orders(comps, [th.exp(x - x.max()) for x in d]), al, p, c))
        pool = ev[b][c].to(DEV)
        probe = pool[th.randperm(len(pool), generator=th.Generator().manual_seed(0))[:4000].to(DEV)]
        row.append(gain(W, with_orders(comps, from_points(comps, al, probe, (margin(W, probe) >= c).double())), al, p, c))
        print(f"  c={c:g} {labels[i]:10s} " + " / ".join(f"{v:6.2f}" for v in row), flush=True)
