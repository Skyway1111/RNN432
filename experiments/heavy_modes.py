"""What carries modes' expected error at c = 6 and 8 on the copies that set it, and which structural levers cut it?

Six held-out copies: the four whose realized errors are largest at c = 6, 8 (s2/15, s2/28, s1/12, s1/16), s4/3, and
example/0. Each copy's events among 2^31 nominal inputs give the exact mismatch m = E_{p|A}[p / q] / P(A) of any proposal
q (relative variance per point m - 1; experiments/expected_score.py checks it against realized errors), with no
realization's luck. Part 1: the library mixture's m and the events that carry it. Part 2: m when the proposal knows the
copy's orders, or is refitted on the copy's own events (half fitted, half scored). Part 3: m with defensive changes and
with a tie atlas, each tie type's order law set by the copy's margins at its 720 tie points. Part 4: the exact S-line,
S integrated over (T, M) at each draw, its m and passes per point, on every point or only where the line is short.
Part 5: the route gradient across one order's events. Events cached in data/cache/heavy_modes/ (about 40 s per copy).

Finding: research/findings/tail.md
"""
import itertools
import math

import numpy as np
import torch as th

from az import osp
from az.estimators import modes
from az.estimators.sline import level_mass
from az.model import DATA, trace
from az.rare import models

th.set_grad_enabled(False)
DEV = "cuda"
CACHE = DATA / "cache" / "heavy_modes"
CACHE.mkdir(parents=True, exist_ok=True)
COPIES = ("s2/15", "s2/28", "s1/12", "s1/16", "s4/3", "example/0")
N_ALL = 2**31
labels, ws = models("heldout")
H = osp.H


def events(i):
    """Directions and margins of the copy's inputs with margin >= 6, from 2^31 nominal inputs; cached."""
    path = CACHE / f"ev_{i}.pt"
    if path.exists():
        return th.load(path).to(DEV)
    W = [th.tensor(a, dtype=th.float32, device=DEV) for a in ws[i]]
    g = th.Generator(device=DEV).manual_seed(11)
    out = []
    for _ in range(2 ** 9):
        z = th.randn(2**22, 10, generator=g, device=DEV)
        u = (z / z.norm(dim=1, keepdim=True)).double()
        f = modes.margin(W, u)
        out.append(th.cat([u[f >= 6], f[f >= 6, None]], 1))
    ev = th.cat(out)
    th.save(ev.cpu(), path)
    return ev


def m_of(comps, alphas, beta, u, pa):
    return th.cat([osp.mixture_weight(comps, alphas, beta, u[a:a + 8192]) for a in range(0, len(u), 8192)]).mean().item() / pa


def floor(comps, eps):
    return osp.OSP((1 - eps) * comps.pi + eps * osp.valid_keys() / 720, comps.mu, comps.sig, comps.tm, comps.tt, comps.tl)


KEYS = th.tensor([p * 100 + q * 10 + r for p, q, r in itertools.permutations(range(10), 3)], device=DEV)
TIES = {  # tie point (M, S, T) and component shape (mu, sig, tm, tt, tl) in x-space units, per tie type
    "top two": ((0.48, 0.48, 0.30), (1.4, 0.5, 0.01, 0.6, 0.6)),
    "top three": ((0.48, 0.48, 0.47), (1.4, 0.5, 0.02, 0.03, 0.6)),
    "S at T": ((0.70, 0.10, 0.10), (0.3, 0.4, 1.5, 0.03, 0.6)),
}
LOW = th.tensor([0.05, -0.05, -0.12, -0.2, -0.28, -0.36, -0.45], dtype=th.float64, device=DEV)


def tie_margins(w, point, placements=4):
    """(720,) the copy's margin for label q at the tie point of each order (p, q, r), the lower inputs placed at random."""
    W = [th.tensor(a, dtype=th.float32, device=DEV) for a in w]
    p, q, r = KEYS // 100, KEYS // 10 % 10, KEYS % 10
    g = th.Generator(device=DEV).manual_seed(0)
    out = th.zeros(720, dtype=th.float64, device=DEV)
    rows = th.arange(720, device=DEV)
    for _ in range(placements):
        free = th.ones(720, 10, dtype=th.bool, device=DEV)
        free[rows, p] = free[rows, q] = free[rows, r] = False
        x = th.zeros(720, 10, dtype=th.float64, device=DEV).scatter(1, th.rand(720, 10, generator=g, device=DEV).masked_fill(~free, 2.0).argsort(1)[:, :7], LOW.expand(720, -1))
        x[rows, p], x[rows, q], x[rows, r] = point
        u = x / x.norm(dim=1, keepdim=True)
        L = modes._forward(W, u.float()).double()
        out += (L.scatter(1, q[:, None], -th.inf).max(1).values - L.gather(1, q[:, None])[:, 0]) / placements
    return out


def atlas(w, kappa=2.0):
    parts = []
    for point, shape in TIES.values():
        pi = th.zeros(osp.KEYS, dtype=th.float64, device=DEV)
        m = tie_margins(w, point)
        pi[KEYS] = th.softmax(kappa * (m - m.max()), 0)
        parts.append(osp.OSP(pi[None], *shape))
    return osp.OSP.cat(parts)


def log_ndtr_diff(a, b):
    """log(Phi(b) - Phi(a)), a <= b."""
    hi, lo = th.where(a > 0, -a, b), th.where(a > 0, -b, a)
    lh = th.special.log_ndtr(hi)
    return lh + th.log1p(-th.exp(th.special.log_ndtr(lo) - lh))


def log_rest(comps, alphas, beta, x, j):
    """log density, under the mixture in x-space, of the nine inputs other than slot j (x_j integrated over (T, M))."""
    v, idx = x.scatter(1, j[:, None], -th.inf).sort(1, descending=True)
    p, r, M, T = idx[:, 0], idx[:, 1], v[:, 0], v[:, 1]
    top = th.stack([p, j, r], 1)
    low = ((T[:, None] - x) ** 2).scatter(1, top, 0.0)
    isl = th.ones_like(x).scatter(1, top, 0.0)
    w1, w2, w3 = 1 / comps.sig**2, 1 / comps.tm**2, 1 / comps.tt**2
    P = w1 + w2 + w3
    m = (comps.mu * w1 + M[:, None] * w2 + T[:, None] * w3) / P
    Q = (w1 * w2 * (comps.mu - M[:, None]) ** 2 + w1 * w3 * (comps.mu - T[:, None]) ** 2 + w2 * w3 * (M - T)[:, None] ** 2) / P
    lk = (th.log(comps.pi.T[p * 100 + j * 10 + r].clamp(min=1e-300)) - low @ (0.5 / comps.tl**2).T + isl @ (H - comps.tl.log()).T
          - th.log(comps.sig * math.sqrt(2 * math.pi)) + 2 * H - comps.tm.log() - comps.tt.log() - Q / 2 + 0.5 * th.log(2 * math.pi / P)
          + log_ndtr_diff(P.sqrt() * (T[:, None] - m), P.sqrt() * (M[:, None] - m)))
    lR = -4.5 * math.log(2 * math.pi) - 0.5 * (x.scatter(1, j[:, None], 0.0) ** 2).sum(1)
    return th.logsumexp(th.cat([(math.log(beta) + lR + log_ndtr_diff(T, M))[:, None], alphas.log() + lk], 1), 1), lR, T, M


def sline_value(W64, comps, alphas, x, j, c):
    """phi_9(R) int_T^M phi(s) 1_A ds / q_R(R): the estimate when S is integrated exactly; and passes per line."""
    lq, lR, T, M = log_rest(comps, alphas, modes.BETA, x, j)
    m, ops = level_mass(*W64, x, j, T, M, th.tensor([c], dtype=th.float64, device=DEV))
    return th.exp(lR - lq) * m[:, 0], ops / 2880


def short(x, j, delta=0.03):
    v = x.scatter(1, j[:, None], -th.inf).sort(1, descending=True).values
    return (v[:, 0] - v[:, 1]) < delta * (x.scatter(1, j[:, None], 0.0) ** 2).sum(1).sqrt()


def proposal(comps, alphas, n, seed=5):
    g = th.Generator(device=DEV).manual_seed(seed)
    part = th.multinomial(th.cat([th.tensor([modes.BETA], dtype=th.float64, device=DEV), alphas]), n, replacement=True, generator=g)
    z = th.randn(n, 13, generator=g, device=DEV, dtype=th.float64)
    x = comps.place((part - 1).clamp(min=0), th.rand(n, generator=g, device=DEV, dtype=th.float64), z)
    x[part == 0] = z[part == 0, :10]
    return x, x.topk(2, dim=1).indices[:, 1]


PER = (modes.TAIL_OPS + modes.DRAW_OPS + modes.GAP_OPS + modes.COMP_OPS * 20) / 2880  # passes per point, no first stage
print("Part 1: the library mixture's mismatch m, the share of E[p/q] carried by the heaviest 10% of events, their commonest"
      " orders (p, q, r), and their median gaps M - S, S - T (all events in brackets)")
data = {}
for lab in COPIES:
    i = labels.index(lab)
    ev = events(i)
    comps, alphas, _ = modes.mixture(modes.base_of(ws[i]))
    data[lab] = (i, ev, comps, alphas)
    for c in (6.0, 8.0):
        u = ev[ev[:, 10] >= c, :10]
        lr = th.cat([osp.mixture_weight(comps, alphas, modes.BETA, u[a:a + 8192]) for a in range(0, len(u), 8192)])
        o = lr.argsort(descending=True)[: len(u) // 10]
        g = osp.Gaps(u)
        k, n = th.unique(g.key[o], return_counts=True)
        top = ", ".join(f"({a // 100},{a // 10 % 10},{a % 10})" for a in k[n.argsort(descending=True)[:3]].tolist())
        print(f"  {lab:9s} c={c:g}: {len(u):6d} events, m {lr.mean().item() / (len(u) / N_ALL):7.0f}; heaviest 10% carry {lr[o].sum().item() / lr.sum().item():.2f}, "
              f"orders {top}; M - S {g.dm2[o].sqrt().median():.4f} ({g.dm2.sqrt().median():.4f}), S - T {g.dt2[o].sqrt().median():.4f} ({g.dt2.sqrt().median():.4f})", flush=True)

print("\nPart 2: m with the copy's own order law on every component (from its events at c >= 6); with K = 4 components per"
      " level refitted on half the copy's events, scored on the other half")
for lab, (i, ev, comps, alphas) in data.items():
    row = []
    pi = th.bincount(osp.Gaps(ev[:, :10]).key, minlength=osp.KEYS).double()
    own = osp.OSP(((pi + 0.5 * pi.sum() / 720) * osp.valid_keys()).expand(len(comps), -1), comps.mu, comps.sig, comps.tm, comps.tt, comps.tl)
    perm = th.randperm(len(ev), generator=th.Generator(device=DEV).manual_seed(0), device=DEV)
    fit, held = ev[perm[: len(ev) // 2]], ev[perm[len(ev) // 2:]]
    parts, als = [], []
    for c in (6.0, 8.0):
        e = fit[fit[:, 10] >= c, :10][:20000]
        cs, al = osp.em(e * th.randn(len(e), 10, generator=th.Generator(device=DEV).manual_seed(1), device=DEV, dtype=th.float64).norm(dim=1, keepdim=True), 4)
        parts.append(cs), als.append(al * 0.4)
    refit = (osp.OSP.cat(parts), th.cat(als))
    for c in (6.0, 8.0):
        pa = (ev[:, 10] >= c).sum().item() / N_ALL
        u, uh = ev[ev[:, 10] >= c, :10], held[held[:, 10] >= c, :10]
        row.append(f"c={c:g}: library {m_of(comps, alphas, 0.2, u, pa):7.0f}, own orders {m_of(own, alphas, 0.2, u, pa):7.0f}, refit {m_of(*refit, 0.2, uh, pa):7.0f}")
    print(f"  {lab:9s} " + "; ".join(row), flush=True)

print("\nPart 3: m with defensive changes and with a tie atlas (share 0.15)")
for lab, (i, ev, comps, alphas) in data.items():
    A = atlas(ws[i])
    variants = {
        "library": (comps, alphas, 0.2),
        "orders floored at 0.3 uniform, nominal 0.05": (floor(comps, 0.3), alphas / alphas.sum() * 0.95, 0.05),
        "uniform-order copies, scales x2": (osp.OSP.cat([comps, osp.OSP(osp.valid_keys().double().expand(len(comps), -1), comps.mu, 2 * comps.sig, 2 * comps.tm, 2 * comps.tt, 2 * comps.tl)]),
                                            th.cat([alphas / alphas.sum() * 0.8, alphas / alphas.sum() * 0.15]), 0.05),
        "tie atlas": (osp.OSP.cat([comps, A]), th.cat([alphas / alphas.sum() * 0.65, th.full((len(A),), 0.05, dtype=th.float64, device=DEV)]), 0.2),
    }
    out = []
    for name, (cs, al, beta) in variants.items():
        ms = [m_of(cs, al, beta, ev[ev[:, 10] >= c, :10], (ev[:, 10] >= c).sum().item() / N_ALL) for c in (6.0, 8.0)]
        out.append(f"{name} {ms[0]:.0f}/{ms[1]:.0f}")
    print(f"  {lab:9s} c = 6/8: " + "; ".join(out), flush=True)
    if lab == "s1/12":
        rank = {kind: tie_margins(ws[i], point).argsort(descending=True).argsort() for kind, (point, _) in TIES.items()}
        print("            the copy's commonest c = 6 orders (5,0,9), (5,0,7): rank of their top-two tie margin among 720: "
              + ", ".join(str(rank["top two"][(KEYS == k).nonzero()[0, 0]].item()) for k in (509, 507)))

print("\nPart 4: the exact S-line, S integrated over (T, M) at each draw (x-space): m, passes per point, and m x passes against"
      " the library's m x passes, on every point and only where the line is short ((M - T) < 0.03 |rest|)")
for lab, (i, ev, comps, alphas) in data.items():
    W64 = [th.tensor(a, dtype=th.float64, device=DEV) for a in ws[i]]
    xq, jq = proposal(comps, alphas, 2**13)
    _, lp = sline_value(W64, comps, alphas, xq, jq, 6.0)
    sh = short(xq, jq)
    passes = {"every": PER - 1 + lp.mean().item(), "short": PER + ((lp - 1) * sh).mean().item()}
    out = []
    for c in (6.0, 8.0):
        sel = ev[:, 10] >= c
        pa = sel.sum().item() / N_ALL
        u = ev[sel, :10][:8192]
        x = u * th.randn(len(u), 10, generator=th.Generator(device=DEV).manual_seed(2), device=DEV, dtype=th.float64).norm(dim=1, keepdim=True)
        j = x.topk(2, dim=1).indices[:, 1]
        plain = osp.mixture_weight(comps, alphas, modes.BETA, u)
        line = sline_value(W64, comps, alphas, x, j, c)[0]
        mixed = th.where(short(x, j), line, plain)
        m0, m1, m2 = (v.mean().item() / pa for v in (plain, line, mixed))
        out.append(f"c={c:g}: m {m0:.0f} -> {m1:.0f} every, {m2:.0f} short; m x passes {m0 * PER:.0f} -> {m1 * passes['every']:.0f}, {m2 * passes['short']:.0f}")
    print(f"  {lab:9s} passes {PER:.2f} -> {passes['every']:.2f} every, {passes['short']:.2f} short; " + "; ".join(out), flush=True)

print("\nPart 5: across the events of a copy's commonest c = 8 orders, the route gradient g = d(l_k - l_q)/dx: cosine to its mean"
      " direction (median, 10th percentile), |g| (median), and the mean route's margin over the true one (median)")
for lab in ("s2/15", "example/0"):
    i, ev, _, _ = data[lab]
    u = ev[ev[:, 10] >= 8, :10]
    g = osp.Gaps(u)
    k, n = th.unique(g.key, return_counts=True)
    for key in k[n.argsort(descending=True)[:3]].tolist():
        e = u[g.key == key]
        L, _, J = trace(ws[i], e, jacobian=True)
        q = key // 10 % 10
        j = L.scatter(1, th.full((len(e), 1), q, device=DEV), -th.inf).argmax(1)
        G = J[th.arange(len(e)), j] - J[:, q]
        mean = G.mean(0)
        cos = (G / G.norm(dim=1, keepdim=True)) @ (mean / mean.norm())
        ratio = (e @ mean) / (L.gather(1, j[:, None])[:, 0] - L[:, q])
        print(f"  {lab:9s} ({key // 100},{key // 10 % 10},{key % 10}), {len(e)} events: cos {cos.median():.3f}, {cos.quantile(0.1):.3f}; "
              f"|g| {G.norm(dim=1).median():.1f}; mean route / true {ratio.median():.2f}")
w0 = ws[labels.index("example/0")]
Lneg = np.maximum(-w0.Wi, 0) @ w0.Wo.T
print(f"  example/0: D = max_j L_j - L_9 with L = Wo relu(-Wi), the all-negative mode's slope from the weights: {np.max(np.delete(Lneg, 9)) - Lneg[9]:.1f}")
