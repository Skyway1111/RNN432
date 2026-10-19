"""Is the phase wheel a clock: does the readout's dominant plane assign positions equally spaced angles, and does the state rotate to match?

The readout W_oh's top two singular directions span a plane. Each position's readout row has an angle in it, and so does
the state h_10 of a two-impulse input (Keane: x_p = 2, x_q = 1, the rest 0). The state's angle should land on its position's.

Finding: research/findings/circuit.md
"""
import numpy as np
import torch as th

from az.model import NAMES, load, trace

np.set_printoptions(precision=0, suppress=True, linewidth=160)


def unwrap(deg):
    return np.degrees(np.unwrap(np.radians(deg)))


for name in NAMES:
    w = load(name)
    U, s, Vt = np.linalg.svd(w.Wo)
    plane = Vt[:2]  # (2, 16): the readout plane in state space
    row_ang = np.degrees(np.arctan2(U[:, 1] * s[1], U[:, 0] * s[0]))
    print(f"{name}: W_oh singular values {s[:4].round(1)}; share of ||W_oh||^2 in the top plane {(s[:2] ** 2).sum() / (s ** 2).sum():.2f}")
    print(f"  readout row angle by position: {row_ang.round(0)}  steps: {np.diff(unwrap(row_ang)).round(0)}")
    hits, errs = 0, []
    for q in range(10):
        angs = []
        for p in range(10):
            if p == q:
                continue
            x = th.zeros(1, 10, dtype=th.float64)
            x[0, p], x[0, q] = 2.0, 1.0
            L, pre, _ = trace(w, x)
            h = th.relu(pre[0, -1]).numpy()
            v = plane @ h
            angs.append(np.degrees(np.arctan2(v[1], v[0])))
            hits += int(L[0].argmax().item() == q)
        d = np.abs(((np.array(angs) - row_ang[q] + 180) % 360) - 180)
        errs.append(d.mean())
        print(f"  q={q}: state angle over p: mean {np.degrees(np.arctan2(np.sin(np.radians(angs)).mean(), np.cos(np.radians(angs)).mean())):+5.0f}, "
              f"readout row {row_ang[q]:+5.0f}, |diff| mean {d.mean():3.0f} max {d.max():3.0f}")
    print(f"  two-impulse accuracy {hits / 90:.3f}; mean |state angle - row angle| {np.mean(errs):.0f} deg")
