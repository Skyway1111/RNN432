"""Which failure modes hold the confident-error tail, per base and level, and what in the weights sets each mode's margin?

Held-out copy /0 of each base (noise seeds 5000 + i), 2^28 Gaussian directions each. Per level, each event goes to the
first mode it fits: at most one positive input (and the share of those with the label at position 9); one dominant input,
|u_k| >= 0.7, by sign; the top two within 0.02 of a tie (M - S); the 2nd and 3rd within 0.02 (S - T); the rest. Then the
median tie gaps of the events against the nominal law's, and their median distance to the nearest two-value direction
(every input equal to one of two values, the vertices of the order arrangement). Deduced, one forward pass each: the
margin max_{j != q} l_j - l_q over |x| at the tie vertices -(1 - e_p)/3 (all inputs tied below the max at zero, label 9,
best p), +e_k and -e_k (one input dominant, best label), and (e_p + e_q)/sqrt2 (the top two tied, best label); and for the
first mode, the quiet-state slope D = max_j L_j - L_9 with L = Wo relu(-Wi), the logits of a last input alone, against
the measured median of f / |u_9| on that mode's events.

Finding: research/findings/tail.md
"""
import torch as th

from az.estimators.modes import margin
from az.rare import LEVELS, models

th.set_grad_enabled(False)
labels, ws = models("heldout")
DEV = "cuda"
E = th.eye(10, dtype=th.float64, device=DEV)
MODES = ("le1pos", "dom+", "dom-", "M-S<.02", "S-T<.02", "rest")


def two_value_distance(u):
    """Distance to the nearest direction whose inputs take two values: the best split of the sorted values."""
    v = u.sort(1, descending=True).values
    cs, tot = v.cumsum(1), v.sum(1)
    best = th.stack([cs[:, m - 1] ** 2 / m + (tot - cs[:, m - 1]) ** 2 / (10 - m) for m in range(1, 10)], 1).max(1).values
    return (1 - best).clamp(min=0).sqrt()


def vertex_margins(W, X):
    L = margin_all(W, X)
    return L


def margin_all(W, X):
    """(N, 10): margin over |x| for every label q at points X."""
    Wi, Wh, Wo = W
    h = th.zeros(len(X), 16, device=DEV)
    for t in range(10):
        h = th.relu(h @ Wh.T + X[:, t, None].float() * Wi)
    L = (h @ Wo.T).double()
    out = th.stack([L.index_fill(1, th.tensor([q], device=DEV), -th.inf).max(1).values - L[:, q] for q in range(10)], 1)
    return out / X.norm(dim=1, keepdim=True)


nominal = None
for i in range(0, 192, 32):
    W = [th.tensor(a, dtype=th.float32, device=DEV) for a in ws[i]]
    g = th.Generator(device=DEV).manual_seed(i)
    share = th.zeros(len(LEVELS), len(MODES) + 1, dtype=th.float64, device=DEV)
    gaps = {c: [] for c in LEVELS}
    ratio = []
    for it in range(64):
        z = th.randn(2**22, 10, generator=g, device=DEV)
        u = (z / z.norm(dim=1, keepdim=True)).double()
        v, idx = u.sort(1, descending=True)
        f = margin(W, u)
        dom = u.abs().max(1)
        pos = u.gather(1, dom.indices[:, None])[:, 0] > 0
        le1 = (u > 0).sum(1) <= 1
        m = th.stack([le1, (dom.values >= 0.7) & pos, (dom.values >= 0.7) & ~pos, v[:, 0] - v[:, 1] < 0.02, v[:, 1] - v[:, 2] < 0.02], 1)
        mode = th.where(m.any(1), m.double().argmax(1), len(MODES) - 1)
        if nominal is None and it == 0:
            nominal = (th.bincount(mode, minlength=len(MODES)).double() / len(u), th.stack([v[:, 0] - v[:, 1], v[:, 1] - v[:, 2], two_value_distance(u)], 1).median(0).values)
        for j, c in enumerate(LEVELS):
            s = f >= c
            share[j, :len(MODES)] += th.bincount(mode[s], minlength=len(MODES)).double()
            share[j, -1] += (s & le1 & (idx[:, 1] == 9)).sum()
            if len(gaps[c]) < 4:
                gaps[c].append(th.stack([v[s, 0] - v[s, 1], v[s, 1] - v[s, 2], two_value_distance(u[s])], 1))
        a = le1 & (idx[:, 1] == 9) & (f >= 4)
        ratio.append(f[a] / u[a, 9].abs())
    Ap = margin_all(W, -(1 - E) / 3)[:, 9].max().item()
    Bp, Bm = margin_all(W, E), margin_all(W, -E)
    Bp[th.arange(10), th.arange(10)] = -th.inf
    pairs = th.stack([E[p] + E[q] for p in range(10) for q in range(p + 1, 10)]) / 2**0.5
    Cm = margin_all(W, pairs)
    keep = th.stack([(E[p] + E[q]) > 0 for p in range(10) for q in range(p + 1, 10)])
    C = Cm.masked_fill(~keep, -th.inf).max().item()
    Wi, _, Wo = (th.tensor(a, dtype=th.float64, device=DEV) for a in ws[i])
    Lq = Wo @ th.relu(-Wi)
    D = (Lq[:9].max() - Lq[9]).item()
    r = th.cat(ratio)
    print(f"{labels[i]}: vertex margins A {Ap:.1f}, +e_k {Bp.max():.1f}, -e_k {Bm.max():.1f}, top-two tie {C:.1f}; "
          f"A mode: quiet-state D {D:.1f}, measured f/|u_9| median {r.median():.1f} (n={len(r)})")
    for j, c in enumerate(LEVELS):
        t = share[j, :len(MODES)].sum()
        gq = th.cat(gaps[c]).median(0).values
        print(f"   c={c:g} ({int(t)} events): " + " ".join(f"{n} {v / t:.2f}" for n, v in zip(MODES, share[j, :len(MODES)].tolist()))
              + f" (label 9: {share[j, -1] / share[j, 0].clamp(min=1):.2f}); median M-S {gq[0]:.3f}, S-T {gq[1]:.3f}, to two values {gq[2]:.2f}", flush=True)
print("nominal: " + " ".join(f"{n} {v:.3f}" for n, v in zip(MODES, nominal[0].tolist()))
      + f"; median M-S {nominal[1][0]:.3f}, S-T {nominal[1][1]:.3f}, to two values {nominal[1][2]:.2f}")
