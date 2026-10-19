"""Is the decision a phase in the readout plane, and is the residual a sum of per-step pushes on that phase?

h_10 = sum over t of P_t W_hi x_t (exact; `contributions` in common.py), so its projection onto the readout plane is a
sum of ten 2-D vectors: the 2nd max sets the hand, every other input pushes it. Asks (A) how many readout directions the
decision needs, (B) how the hand turns step by step on a two-impulse input, (C) how large the pushes are by rank,
(D) whether a step's push per unit input is fixed by its offset from the 2nd-max event, and (E) where errors sit in the plane.

Finding: research/findings/circuit.md
"""
import numpy as np
import torch as th

from az.model import NAMES, load, trace
from experiments.common import DEV, contributions, inputs, ranks

np.set_printoptions(precision=2, suppress=True, linewidth=160)


def wrap(deg):
    return (deg + 180) % 360 - 180


def angle(v):
    return th.rad2deg(th.atan2(v[..., 1], v[..., 0]))


def r2_by_group(u, key):
    """Share of u's variance (any trailing shape) explained by the group means of key."""
    vals, inv, cnt = th.unique(key, return_inverse=True, return_counts=True)
    flat = u.reshape(len(u), -1)
    s = th.zeros(len(vals), flat.shape[1], dtype=u.dtype, device=u.device).index_add_(0, inv, flat)
    mean = s / cnt[:, None]
    return 1 - ((flat - mean[inv]) ** 2).sum() / ((flat - flat.mean(0)) ** 2).sum()


x = inputs(2**18, seed=21)
srt, ys = ranks(x)
p, q, r = srt[:, 0], srt[:, 1], srt[:, 2]

print("(A) decision with the readout truncated to its top k singular directions: agreement with the full model, accuracy")
for name in NAMES:
    w = load(name)
    U, s, Vt = np.linalg.svd(w.Wo)
    L, pre, _ = trace(w, x)
    h = th.relu(pre[:, -1])
    full = L.argmax(1)
    out = []
    for k in range(1, 5):
        Wk = th.tensor((U[:, :k] * s[:k]) @ Vt[:k], device=DEV)
        pk = (h @ Wk.T).argmax(1)
        out.append(f"k={k}: {(pk == full).double().mean():.3f} / {(pk == q).double().mean():.3f}")
    print(f"  {name}: full accuracy {(full == q).double().mean():.4f};  " + "  ".join(out))

w = load("example")
U, s, Vt = np.linalg.svd(w.Wo)
plane = th.tensor(Vt[:2], device=DEV)
row_ang = th.tensor(np.degrees(np.arctan2(U[:, 1] * s[1], U[:, 0] * s[0])), device=DEV)

print("\n(B) example, two-impulse inputs: the hand's angle and radius in the plane after each step (neurons on, canonical core order 2 4 7 6 1 8 then satellites)")
order = [2, 4, 7, 6, 1, 8, 0, 3, 5, 9, 10, 11, 12, 13, 14, 15]
for pp, qq in [(1, 5), (6, 2)]:
    xx = th.zeros(1, 10, dtype=th.float64, device=DEV)
    xx[0, pp], xx[0, qq] = 2.0, 1.0
    _, pre, _ = trace(w, xx)
    print(f"  p={pp}, q={qq}; readout angle of q {row_ang[qq]:+.0f}")
    for t in range(10):
        v = plane @ th.relu(pre[0, t])
        on = "".join("1" if pre[0, t, n] > 0 else "." for n in order)
        print(f"    after step {t}: angle {angle(v):+5.0f}  radius {v.norm():6.1f}  on {on[:6]} {on[6:]}")

C = contributions(w, x)
assert th.allclose(C.sum(1), th.relu(trace(w, x)[1][:, -1]), atol=1e-8)
V = C @ plane.T  # (N, 10, 2): the plane vector each step contributes
by_rank = V.gather(1, srt[..., None].expand(-1, -1, 2))  # rank k at index k-1
hand = by_rank[:, 1]
rest = V.sum(1) - hand
total = V.sum(1)
print("\n(C) example, Gaussian inputs: plane vectors by rank")
print("  mean |v| by rank 1..10:", by_rank.norm(dim=2).mean(0).cpu().numpy())
print(f"  |hand| mean {hand.norm(dim=1).mean():.1f}; |rest| mean {rest.norm(dim=1).mean():.1f}; |total| mean {total.norm(dim=1).mean():.1f}")
d_hand = wrap(angle(hand) - row_ang[q])
d_total = wrap(angle(total) - row_ang[q])
print(f"  hand angle minus q's readout angle: mean {d_hand.mean():+.1f}, sd {d_hand.std():.1f}, |.| > 22.5 deg share {(d_hand.abs() > 22.5).double().mean():.3f}")
print(f"  total angle minus q's readout angle: mean {d_total.mean():+.1f}, sd {d_total.std():.1f}, |.| > 22.5 deg share {(d_total.abs() > 22.5).double().mean():.3f}")
unit = hand / hand.norm(dim=1, keepdim=True)
radial = (rest * unit).sum(1)
tang = rest[:, 0] * -unit[:, 1] + rest[:, 1] * unit[:, 0]
print(f"  rest along the hand: mean {radial.mean():+.1f} sd {radial.std():.1f};  across it: mean {tang.mean():+.1f} sd {tang.std():.1f}")
for k in [1, 3, 4, 5]:
    vk = by_rank[:, k - 1]
    tk = vk[:, 0] * -unit[:, 1] + vk[:, 1] * unit[:, 0]
    print(f"  rank {k}: push across the hand mean {tk.mean():+.1f} sd {tk.std():.1f}")

print("\n(D) example: push per unit input at step t, u_t = plane P_t W_hi, explained by what?")
unit_push = th.where(x.abs()[..., None] > 1e-9, V / x[..., None], th.zeros_like(V))
steps = th.arange(10, device=DEV)[None].expand(len(x), -1)
offset = steps - q[:, None]
after_p = (steps > p[:, None]).long()
mask = steps != q[:, None]
u, off, ap, sgn = unit_push[mask], offset[mask], after_p[mask], (x > 0).long()[mask]
rank_t = srt.argsort(1)[mask]  # rank of the input at step t
print(f"  R2 of u_t by offset t-q: {r2_by_group(u, off):.3f};  by (offset, after p): {r2_by_group(u, off * 2 + ap):.3f};  "
      f"by (offset, after p, sign): {r2_by_group(u, (off * 2 + ap) * 2 + sgn):.3f};  by (offset, after p, rank): {r2_by_group(u, (off * 2 + ap) * 10 + rank_t):.3f}")
late = off > 0
print(f"  steps after q only: by offset {r2_by_group(u[late], off[late]):.3f};  by (offset, after p): {r2_by_group(u[late], (off * 2 + ap)[late]):.3f};  "
      f"by (offset, after p, sign): {r2_by_group(u[late], ((off * 2 + ap) * 2 + sgn)[late]):.3f}")
print("  mean |u_t| by offset t-q = 1..9: ", np.array([u[off == d].norm(dim=1).mean().item() for d in range(1, 10)]))
print("  mean |u_t| by offset t-q = -9..-1:", np.array([u[off == d].norm(dim=1).mean().item() for d in range(-9, 0)]))

print("\n(E) example: errors in the plane")
L = trace(w, x)[0]
pred = L.argmax(1)
err = pred != q
qr = err & (pred == r)
d_r = wrap(angle(total) - row_ang[r])
print(f"  error rate {err.double().mean():.4f}; of errors, predicted r {qr.sum() / err.sum():.2f}")
print(f"  on q-vs-r errors: |total angle - q's| mean {d_total[qr].abs().mean():.0f}, |total angle - r's| mean {d_r[qr].abs().mean():.0f}, closer to r {(d_r[qr].abs() < d_total[qr].abs()).double().mean():.2f}")
print(f"  on correct inputs: |total angle - q's| mean {d_total[~err].abs().mean():.0f}")
print(f"  |total| on errors {total[err].norm(dim=1).mean():.1f} vs correct {total[~err].norm(dim=1).mean():.1f}")
