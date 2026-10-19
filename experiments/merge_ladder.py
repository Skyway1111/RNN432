"""Certified merging in a route mixture: how many components does a given accuracy error cost?

Routes with different linear maps cannot merge exactly (parked/research/angles/route-mixture.md). The cheapest merge with a
measurable error: within each (family, record history) cell, force every gate whose deviation from the cell's majority
is rarer than tau to the majority, and keep the rest as the input's own bits. The components are the distinct
(cell, reduced pattern) keys that remain, and the error is the merged model's accuracy against the true one, on the
same 2^22 inputs. Any certificate would bound this error from above, so the ladder bounds the component count a
certified mixture needs from below. Three suite models.

Finding: parked/research/angles/route-mixture.md
"""
import torch as th

from az.bench import suite
from experiments.common import DEV, inputs, ranks
from experiments.error_cells import HASH, describe, history

N = 2**22
labels, ws = suite(0.002, 32)
TAUS = (0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.0)


def patterns(w, x):
    """(N, 10, 16) gate bits."""
    Wi, Wh, _ = (th.tensor(a, device=DEV) for a in w)
    h = th.zeros(len(x), 16, dtype=th.float64, device=DEV)
    P = []
    for t in range(10):
        z = h @ Wh.T + x[:, t, None] * Wi
        P.append(z > 0)
        h = th.relu(z)
    return th.stack(P, 1)


def forced(w, x, mask):
    """Prediction with the gates replaced by mask (N, 10, 16)."""
    Wi, Wh, Wo = (th.tensor(a, device=DEV) for a in w)
    h = th.zeros(len(x), 16, dtype=th.float64, device=DEV)
    for t in range(10):
        h = (h @ Wh.T + x[:, t, None] * Wi) * mask[:, t]
    return (h @ Wo.T).argmax(1)


for i in (0, 32, 160):
    w = ws[i]
    x = inputs(N, seed=11)
    fam, pred, _ = describe(w, x)
    q = ranks(x)[0][:, 1]
    acc = (pred == q).double().mean().item()
    cell = fam * 59049 + history(x)
    _, inv, cnt = th.unique(cell, return_inverse=True, return_counts=True)
    P = patterns(w, x).reshape(N, 160)
    rate = th.zeros(len(cnt), 160, dtype=th.float64, device=DEV).index_add_(0, inv, P.double()) / cnt[:, None]
    major = rate > 0.5
    dev = th.where(major, 1 - rate, rate)  # per cell and bit, the share of the cell's inputs off the majority
    print(f"{labels[i]}: accuracy {acc:.4f}; cells {len(cnt)}; bits off the majority per input, mean "
          f"{(P != major[inv]).double().sum(1).mean().item():.1f}")
    print(f"  {'tau':>6} {'components':>10} {'for 99% mass':>12} {'merged bits':>11} {'acc error':>9}")
    for tau in TAUS:
        keep = (dev[inv] >= tau) if tau > 0 else th.ones_like(P)
        reduced = th.where(keep, P, major[inv])
        key = (reduced.long() * HASH).sum(1) + cell * 1000003
        _, cnt2 = th.unique(key, return_counts=True)
        cs = cnt2.sort(descending=True).values.cumsum(0).double() / N
        err = (forced(w, x, reduced.reshape(N, 10, 16).double()) == q).double().mean().item() - acc
        print(f"  {tau:6.3f} {len(cnt2):10d} {int((cs < 0.99).sum().item()) + 1:12d} {(~keep).double().mean().item() * 160:11.1f} {err:+9.4f}")
