"""Importance sampling over the failure modes: a mixture of order-statistic gap proposals (az/osp.py) per level.

The tail sits at ties, in modes that differ by base (research/findings/tail.md): near-tie wedges at the T end and the M end,
inputs with at most one positive value whose last input is the 2nd largest, one dominant input, the top two tied. Each is a
region of the top three's positions, the 2nd largest's value and the gaps around it, which a gap proposal describes with
an exact angular density. Per base and level the library holds K[level] components fitted by EM to the events of fresh
noisy copies of that base (noise seeds 9000 + base index, outside both suites): structure found on the base, never on a
suite copy. A copy is matched to its base by its weights. The estimate is importance sampling from BETA of the nominal law
plus the base's components, weighted by the balance heuristic, so each point serves every level. Each part of the mixture
gets a fixed count of points from a randomly shifted Kronecker lattice, seeded by the copy's weights.
Weight noise moves the modes between orders (p, q, r) and keeps their shapes, so the first ADAPT of every part's points
re-estimates each component's order law from its soft events, Phi((margin - level) / SOFT) times the LR and the
component's responsibility, plus PRIOR times that mass of the library's law; the rest are drawn with the new orders.
Each stage is unbiased given the ones before it, so the estimate is too.
Events: data/cache/modes_events.pt, built by events() when missing (about a minute on an RTX 5080); library() fits them.
Ops per point: the forward pass and |x| (TAIL_OPS), 13 inverse normal CDFs, the shared gap statistics, and per component
a quadratic form over the gaps, two logs or roots, the K9 lookup and an exp (research/angles/mode-mixture.md).
"""
import zlib
from functools import cache

import numpy as np
import torch as th

from az import osp
from az.estimators.lattice import roberts
from az.model import DATA, NAMES, load, noisy
from az.ops import FN
from az.rare import LEVELS, TAIL_OPS

EVENTS = DATA / "cache" / "modes_events.pt"
K = {2.0: 8, 4.0: 4, 6.0: 4, 8.0: 4}  # components per level
FRESH, CAP, DRAWS = 16, 3000, 2**26  # fresh copies per base, events kept per copy and level, inputs per copy
SHARE = {2.0: 0.6, 4.0: 0.2, 6.0: 0.1, 8.0: 0.1}  # of the non-nominal samples, per level's components
BETA = 0.2
SALT = 0  # added to the weight-derived seed, to draw other realizations
ADAPT, SOFT, PRIOR = 0.3, 1.0, 1.0  # first-stage share whose events re-estimate the orders; soft level width; library prior
DRAW_OPS, GAP_OPS, COMP_OPS = 20 + 13 * FN, 60 + 2 * FN, 25 + 3 * FN  # see the docstring; GAP_OPS holds the final log and exp
LEARN_OPS = 3 + FN  # per first-stage point and component: responsibility, soft indicator, accumulation
DEV = osp.DEV
CHUNK = 2**20


def _forward(W, u):
    Wi, Wh, Wo = W
    h = th.zeros(len(u), 16, device=DEV)
    for t in range(10):
        h = th.addmm(u[:, t, None] * Wi, h, Wh.T).relu_()
    return h @ Wo.T


def margin(W, u: th.Tensor) -> th.Tensor:
    """max_{j != q} l_j - l_q at unit directions u (float32, as the truth counts it)."""
    L = _forward(W, u.float())
    q = u.topk(2, dim=1).indices[:, 1:]
    return (L.scatter(1, q, -th.inf).max(1).values - L.gather(1, q)[:, 0]).double()


def events() -> dict:
    """{base: {level: event directions}} pooled over FRESH noisy copies of each base, CAP per copy and level; cached."""
    if EVENTS.exists():
        return th.load(EVENTS)
    out = {}
    for b, name in enumerate(NAMES):
        ev = {c: [] for c in K}
        gen = th.Generator(device=DEV).manual_seed(9000 + b)
        for w in noisy(load(name), 0.002, FRESH, seed=9000 + b):
            W = [th.tensor(a, dtype=th.float32, device=DEV) for a in w]
            kept = {c: 0 for c in K}
            for _ in range(DRAWS // 2**22):
                z = th.randn(2**22, 10, generator=gen, device=DEV)
                u = (z / z.norm(dim=1, keepdim=True)).double()
                f = margin(W, u)
                for c in K:
                    e = u[f >= c][: CAP - kept[c]]
                    ev[c].append(e.cpu())
                    kept[c] += len(e)
        out[b] = {c: th.cat(e) for c, e in ev.items()}
    th.save(out, EVENTS)
    return out


@cache
def library() -> dict:
    """{base: {level: (components, weights)}}: K[level] components fitted by EM to the pooled events, radii drawn chi_10."""
    lib = {}
    for b, lv in events().items():
        gen = th.Generator(device=DEV).manual_seed(b)
        lib[b] = {}
        for c, u in lv.items():
            u = u.to(DEV)
            r = th.randn(len(u), 10, generator=gen, device=DEV, dtype=th.float64).norm(dim=1)
            lib[b][c] = osp.em(u * r[:, None], K[c])
    return lib


@cache
def _bases():
    return [load(n) for n in NAMES]


def base_of(w) -> int:
    return int(np.argmin([sum(np.sum((a - b) ** 2) for a, b in zip(w, base)) for base in _bases()]))


def mixture(b: int):
    """The base's components stacked, their weights in the mixture (summing to 1 - BETA), and each one's level."""
    lv = library()[b]
    comps = osp.OSP.cat([cs for cs, _ in lv.values()])
    alphas = th.cat([(1 - BETA) * SHARE[c] / sum(SHARE.values()) * al / al.sum() for c, (_, al) in lv.items()])
    levels = th.cat([th.full((len(cs),), c, dtype=th.float64, device=DEV) for c, (cs, _) in lv.items()])
    return comps, alphas, levels


def counts(probs: th.Tensor, n: int) -> th.Tensor:
    """Deterministic allocation of n points to the mixture's parts, largest remainders first."""
    raw = probs * n
    c = raw.floor().long()
    c[(raw - c).argsort(descending=True)[: n - int(c.sum())]] += 1
    return c


def tail(w, budget, learned=False):
    """Estimates per level and ops; with learned, also the second stage's mixture (components, weights)."""
    comps, alphas, levels = mixture(base_of(w))
    W = [th.tensor(a, dtype=th.float32, device=DEV) for a in w]
    per = TAIL_OPS + DRAW_OPS + GAP_OPS + COMP_OPS * len(comps) + LEARN_OPS * len(comps) * ADAPT
    n = max(int(budget // per), 2 * (len(comps) + 1))
    gen = th.Generator(device=DEV).manual_seed(zlib.crc32(np.ascontiguousarray(w.Wh).tobytes()) + SALT)
    cnt = counts(th.cat([th.tensor([BETA], dtype=th.float64, device=DEV), alphas]), n)
    first = (cnt * ADAPT).round().long() if ADAPT else cnt
    shifts = th.stack([th.rand(14, generator=gen, device=DEV, dtype=th.float64) for _ in range(len(cnt))])
    step = th.tensor(roberts(14), dtype=th.float64, device=DEV)
    lv = th.tensor(LEVELS, dtype=th.float64, device=DEV)
    mass = th.zeros(len(comps), osp.KEYS, dtype=th.float64, device=DEV)

    def stage(parts, lo, hi, learn):
        """Points lo_j..hi_j of every part j (the nominal law first), drawn from the mixture they make up, weighted by the
        balance heuristic; with learn, each point adds its soft event weight to its components' order mass."""
        m = hi - lo
        N = int(m.sum())
        beta, al = m[0].item() / N, m[1:].double() / N
        part = th.repeat_interleave(th.arange(len(m), device=DEV), m)
        idx = th.arange(N, device=DEV) - th.repeat_interleave(m.cumsum(0) - m - lo, m)  # each point's lattice index in its part
        out = th.zeros(len(LEVELS), dtype=th.float64, device=DEV)
        for a in range(0, N, CHUNK):
            j = part[a : a + CHUNK]
            U = ((shifts[j] + idx[a : a + CHUNK, None].double() * step) % 1.0).clamp(1e-15, 1 - 1e-15)
            z = th.special.ndtri(U[:, 1:])
            x = parts.place((j - 1).clamp(min=0), U[:, 0], z)
            x[j == 0] = z[j == 0, :10]
            u = x / x.norm(dim=1, keepdim=True)
            g = osp.Gaps(u)
            lq = osp.mixture_logs(parts, al, beta, g)
            lse = th.logsumexp(lq, 1)
            lr = th.exp(-np.log(osp.AREA) - lse)
            f = margin(W, u)
            if learn:
                r = th.exp(lq[:, 1:] - lse[:, None])  # responsibilities
                mass.index_add_(1, g.key, (lr[:, None] * r * th.special.ndtr((f[:, None] - levels) / SOFT)).T)
            out += ((f[:, None] >= lv) * lr[:, None]).sum(0)
        return out

    s = stage(comps, th.zeros_like(cnt), first, learn=bool(ADAPT))
    if ADAPT:
        comps = comps.reordered(mass, PRIOR)
        s += stage(comps, first, cnt, learn=False)
    out = (s / n).cpu().numpy(), n * per + 6 * 432
    return (*out, (comps, alphas)) if learned else out
