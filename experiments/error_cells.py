"""Is the error set, organized by (family, record history) cells, few convex polytopes with few facets?

If it were, a deterministic solid-angle series per polytope would replace enumeration over a million linear regions
(parked/research/angles/exact-accuracy.md). Three suite models, 2^22 Gaussian inputs each:
  1. Coverage: how many keys hold 50 / 90 / 99% of the error mass when keyed by family, by (family, record history),
     by activation pattern, and by (cell, pattern); and Good-Turing's share of error mass in keys seen once, the mass
     an enumeration of the seen keys would miss.
  2. Convexity: for pairs of error points in one cell, the midpoint's correctness and the number of correctness
     switches along the segment (0 if the cell's error set is convex), and whether the midpoint's pattern is one of
     the pair's.
  3. Facets: per cell, the number of competitors (which position the model picks) and of activation patterns among
     the error points, error-mass weighted.

Finding: parked/research/angles/exact-accuracy.md
"""
import numpy as np
import torch as th

from az.bench import suite
from az.model import logits
from experiments.common import DEV, inputs, ranks

N = 2**22
labels, ws = suite(0.002, 32)
POW3 = th.tensor([3**i for i in range(10)], device=DEV)
HASH = th.tensor(np.random.default_rng(0).integers(1, 2**62, 160) * 2 + 1, device=DEV)


def describe(w, x):
    """Per input: family index, prediction, and a hash of the 160-bit activation pattern."""
    Wi, Wh, Wo = (th.tensor(a, device=DEV) for a in w)
    h = th.zeros(len(x), 16, dtype=th.float64, device=DEV)
    code = th.zeros(len(x), dtype=th.int64, device=DEV)
    for t in range(10):
        z = h @ Wh.T + x[:, t, None] * Wi
        on = z > 0
        code += (on.long() * HASH[16 * t : 16 * t + 16]).sum(1)
        h = z * on
    srt = ranks(x)[0]
    return srt[:, 0] * 100 + srt[:, 1] * 10 + srt[:, 2], (h @ Wo.T).argmax(1), code


def history(x):
    """Record-history code: per step, whether x_t is a new running max (2), a new running 2nd max (1), or neither."""
    M = th.full((len(x),), -1e9, dtype=th.float64, device=DEV)
    S = M.clone()
    rec = th.zeros(len(x), dtype=th.int64, device=DEV)
    for t in range(10):
        newmax = x[:, t] > M
        new2 = ~newmax & (x[:, t] > S)
        rec += (newmax.long() * 2 + new2.long()) * POW3[t]
        S = th.where(newmax, M, th.where(new2, x[:, t], S))
        M = th.where(newmax, x[:, t], M)
    return rec


def coverage(key, label):
    """Keys needed for 50 / 90 / 99% of the error points, and the share of error points whose key appears once."""
    _, inv, cnt = th.unique(key, return_inverse=True, return_counts=True)
    cs = cnt.sort(descending=True).values.cumsum(0).double() / len(key)
    need = [int((cs < f).sum().item()) + 1 for f in (0.5, 0.9, 0.99)]
    print(f"    {label:28s} keys {len(cnt):8d}; for 50 / 90 / 99% of error mass {need[0]:7d} {need[1]:7d} {need[2]:7d}; "
          f"seen once {(cnt[inv] == 1).double().mean().item():.3f}")
    return inv, cnt


def per_cell_count(inv_cell, n_cells, key):
    """For each error point, how many distinct values of key its cell's error points take."""
    _, inv2 = th.unique(inv_cell * 2**40 + key % 2**40, return_inverse=True)
    first = th.zeros(inv2.max().item() + 1, dtype=th.int64, device=DEV).scatter_(0, inv2, th.arange(len(inv2), device=DEV))
    k = th.zeros(n_cells, dtype=th.int64, device=DEV).index_add_(0, inv_cell[first], th.ones(len(first), dtype=th.int64, device=DEV))
    return k[inv_cell].double()


def main(i):
    w = ws[i]
    x = inputs(N, seed=11)
    fam, pred, pat = describe(w, x)
    rec = history(x)
    q = ranks(x)[0][:, 1]
    err = pred != q
    cell = fam * 59049 + rec
    print(f"{labels[i]}: error rate {err.double().mean().item():.4f}")
    print("  1. coverage of the error mass")
    e = err.nonzero().squeeze(1)
    coverage(fam[e], "family")
    inv_cell, cnt_cell = coverage(cell[e], "(family, record history)")
    coverage(pat[e], "pattern")
    coverage(cell[e] * 1000003 + pat[e], "(cell, pattern)")

    print("  2. convexity: pairs of error points in one cell")
    g = th.Generator(device=DEV).manual_seed(5)
    order = th.argsort(cell[e] * 2**22 + th.randint(0, 2**22, (len(e),), generator=g, device=DEV))
    a, b = e[order[:-1]], e[order[1:]]
    same = cell[a] == cell[b]
    a, b = a[same], b[same]
    lam = th.linspace(0, 1, 65, device=DEV, dtype=th.float64)
    mid = (x[a] + x[b]) / 2
    mid_fam, mid_pred, mid_pat = describe(w, mid)
    assert (mid_fam == fam[a]).all() and (history(mid) == rec[a]).all()
    switches = []
    for c in range(0, len(a), 2**16):
        seg = x[a[c : c + 2**16]][:, None] + lam[None, :, None] * (x[b[c : c + 2**16]] - x[a[c : c + 2**16]])[:, None]
        ok_seg = logits([w], seg.reshape(1, -1, 10))[0].argmax(1).reshape(-1, 65) == q[a[c : c + 2**16], None]
        switches.append((ok_seg[:, 1:] != ok_seg[:, :-1]).sum(1))
    switches = th.cat(switches)
    print(f"    pairs {len(a)}; midpoint wrong {(mid_pred != q[a]).double().mean().item():.3f}; segment switches 0 / 2 / more "
          f"{(switches == 0).double().mean().item():.3f} / {(switches == 2).double().mean().item():.3f} / "
          f"{(switches > 2).double().mean().item():.3f}; pair shares a pattern {(pat[a] == pat[b]).double().mean().item():.3f}; "
          f"midpoint pattern is a's or b's {((mid_pat == pat[a]) | (mid_pat == pat[b])).double().mean().item():.3f}")
    print(f"    pairs with different patterns and no switch along the segment: "
          f"{((switches == 0) & (pat[a] != pat[b])).double().mean().item():.3f} of pairs")

    print("  3. facets: per cell, among its error points, error-mass weighted")
    for key, label in ((pred[e], "competitors"), (pat[e], "patterns")):
        k = per_cell_count(inv_cell, len(cnt_cell), key)
        print(f"    {label:12s} per cell: mean {k.mean().item():.2f}; share of error mass in cells with 1 / <= 3 / <= 10 / > 10: "
              f"{(k == 1).double().mean().item():.3f} / {(k <= 3).double().mean().item():.3f} / {(k <= 10).double().mean().item():.3f} / "
              f"{(k > 10).double().mean().item():.3f}")


if __name__ == "__main__":
    for i in (0, 32, 160):
        main(i)
