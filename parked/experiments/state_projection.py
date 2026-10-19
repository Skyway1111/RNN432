"""Can a forward pass over state laws, keyed by the running top three and the activation pattern, carry the decision?

A deduction-projection estimator steps the law of v = (h, running top-three values) forward and projects it, per cell, onto
a Gaussian (inactive neurons exactly 0). Two measurements on Gaussian inputs:
  1. How many cells there are: per step, cells (key = positions of the running top three, x the 16-bit pattern) holding
     mass above 1e-6, and cells per key. 2^23 inputs, two models.
  2. The projection's own error, with the deduction exact: inputs stepped exactly, but after a step each input's v is
     replaced by a draw from its cell's Gaussian (moments over the cell's inputs, cells of 30 or more). Accuracy minus the
     unprojected accuracy on the same 2^21 inputs, projecting at every step (keys only; keys x patterns) or once after
     step 3, 6 or 9. A control redraws v from the cell's own inputs instead of the Gaussian, which keeps the cell's law.

Finding: research/angles/spike-slab-mixture.md
"""
import torch as th

from az.bench import suite
from experiments.common import DEV

th.set_grad_enabled(False)
labels, ws = suite(0.002, 32)
BITS = 2 ** th.arange(16, device=DEV)


def weights(w):
    return (th.tensor(a, device=DEV, dtype=th.float64) for a in w)


print("1. cells (top-three key x activation pattern) with mass above 1e-6, by step")
for i in (32, 0):
    Wi, Wh, _ = weights(ws[i])
    g = th.Generator(device=DEV).manual_seed(1)
    x = th.randn(2**23, 10, generator=g, device=DEV, dtype=th.float64)
    h = th.zeros(2**23, 16, dtype=th.float64, device=DEV)
    row = []
    for t in range(10):
        h = th.relu(h @ Wh.T + x[:, t, None] * Wi)
        k = min(3, t + 1)
        key = (x[:, : t + 1].topk(k, 1).indices * th.tensor([100, 10, 1][:k], device=DEV)).sum(1)
        _, cnt = th.unique(key * 65536 + ((h > 0).long() * BITS).sum(1), return_counts=True)
        big = int((cnt > 2**23 * 1e-6).sum())
        row.append(f"{big} ({big / th.unique(key).numel():.0f}/key)")
    print(f"  {labels[i]:10s} " + ", ".join(row))


def run(w, x, g, mode, at=None):
    """Accuracy on x with v projected per cell after the steps in `at` (all but the last if None)."""
    Wi, Wh, Wo = weights(w)
    n = len(x)
    h = th.zeros(n, 16, dtype=th.float64, device=DEV)
    top = th.full((n, 3), -1e9, dtype=th.float64, device=DEV)
    pos = th.full((n, 3), -1, dtype=th.long, device=DEV)
    for t in range(10):
        h = th.relu(h @ Wh.T + x[:, t, None] * Wi)
        o = th.cat([top, x[:, t, None]], 1).argsort(1, descending=True)[:, :3]
        top = th.cat([top, x[:, t, None]], 1).gather(1, o)
        pos = th.cat([pos, th.full((n, 1), t, dtype=th.long, device=DEV)], 1).gather(1, o)
        if mode == "none" or t == 9 or (at is not None and t not in at):
            continue
        k = min(t + 1, 3)
        cell = ((pos[:, 0] + 1) * 11 + pos[:, 1] + 1) * 11 + pos[:, 2] + 1
        if mode != "keys":
            cell = cell * 65536 + ((h > 0).long() * BITS).sum(1)
        _, inv, cnt = th.unique(cell, return_inverse=True, return_counts=True)
        v = th.cat([h, top[:, :k]], 1)
        if mode == "control":  # redraw from the cell's own inputs: the exact cell law
            o1 = th.argsort(inv.double() + th.rand(n, generator=g, device=DEV, dtype=th.float64))
            o2 = th.argsort(inv.double() + th.rand(n, generator=g, device=DEV, dtype=th.float64))
            v[o1] = v[o2].clone()
        else:
            d = v.shape[1]
            s1 = th.zeros(len(cnt), d, dtype=th.float64, device=DEV).index_add_(0, inv, v)
            s2 = th.zeros(len(cnt), d, d, dtype=th.float64, device=DEV)
            for c in range(0, n, 2**18):
                s2.index_add_(0, inv[c : c + 2**18], v[c : c + 2**18, :, None] * v[c : c + 2**18, None, :])
            mean = s1 / cnt[:, None]
            cov = s2 / cnt[:, None, None] - mean[:, :, None] * mean[:, None, :]
            ev, vec = th.linalg.eigh(th.nan_to_num((cov + cov.transpose(1, 2)) / 2).cpu())
            L = (vec * ev.clamp(min=0).sqrt()[:, None, :]).to(DEV)
            new = mean[inv] + th.einsum("nij,nj->ni", L[inv], th.randn(n, d, generator=g, device=DEV, dtype=th.float64))
            v = th.where((cnt[inv] >= 30)[:, None], new, v)
        h, top[:, :k] = v[:, :16], v[:, 16:]
    return ((h @ Wo.T).argmax(1) == pos[:, 1]).double().mean().item()


print("\n2. projection error in accuracy (projected - exact on the same 2^21 inputs)")
for i in (32, 0, 160):
    g = th.Generator(device=DEV).manual_seed(5)
    x = th.randn(2**21, 10, generator=g, device=DEV, dtype=th.float64)
    exact = run(ws[i], x, g, "none")
    every = {m: run(ws[i], x, g, m) - exact for m in ("control", "keys", "keys x patterns")}
    once = [run(ws[i], x, g, "keys x patterns", at={t}) - exact for t in (2, 5, 8)]
    print(f"  {labels[i]:10s} every step: control {every['control']:+.4f}, keys {every['keys']:+.4f}, keys x patterns "
          f"{every['keys x patterns']:+.4f}; keys x patterns once after step 3, 6, 9: " + ", ".join(f"{e:+.4f}" for e in once), flush=True)
