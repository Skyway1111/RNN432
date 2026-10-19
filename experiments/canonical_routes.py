"""Does one deductive reference route per (p, q) reproduce the neurons' coefficients and the votes measured on Gaussian inputs?

Reference inputs: the mean magnitude set with M at p, S at q, and the other eight values in K fixed orders. Their routes are
exact, and each gives a linear map A (16 x 10) from inputs to h_10. Compares the reference coefficients on M and S, and the
self-votes, with the Gaussian means per (p, q); then applies each (p, q)'s mean reference map to Gaussian inputs as a surrogate.
Example model.

Finding: research/findings/votes.md
"""
import numpy as np
import torch as th

from az.model import load, trace
from experiments.common import DEV, contributions, inputs, order_stat_means, ranks

np.set_printoptions(precision=2, suppress=True, linewidth=200)
w = load("example")
Wo = th.tensor(w.Wo, device=DEV)
K = 8
ystar = th.tensor(order_stat_means(), dtype=th.float64, device=DEV)
rng = np.random.default_rng(0)
orders = [np.arange(8)] + [rng.permutation(8) for _ in range(K - 1)]  # fixed orders of the eight lower values

pairs = [(a, b) for a in range(10) for b in range(10) if a != b]
ref = th.zeros(len(pairs), K, 10, dtype=th.float64, device=DEV)
for i, (a, b) in enumerate(pairs):
    rest = [t for t in range(10) if t not in (a, b)]
    for k, o in enumerate(orders):
        ref[i, k, a], ref[i, k, b] = ystar[0], ystar[1]
        ref[i, k, rest] = ystar[2:][th.as_tensor(o, device=DEV)]
flat = ref.reshape(-1, 10)
A_ref = (contributions(w, flat) / flat[..., None]).reshape(len(pairs), K, 10, 16)  # coefficient of x_t in each neuron
ref_ok = trace(w, flat)[0].argmax(1).reshape(len(pairs), K) == th.tensor([b for _, b in pairs], device=DEV)[:, None]
print(f"reference inputs correct: {ref_ok.double().mean():.3f}")

x = inputs(2**20, seed=24)
srt, ys = ranks(x)
p, q = srt[:, 0], srt[:, 1]
key = p * 10 + q
C = contributions(w, x) / x[..., None]  # (N, 10, 16)
pair_index = th.full((100,), -1, dtype=th.long, device=DEV)
for i, (a, b) in enumerate(pairs):
    pair_index[a * 10 + b] = i
g = pair_index[key]
cnt = th.zeros(len(pairs), dtype=th.float64, device=DEV).index_add_(0, g, th.ones(len(x), dtype=th.float64, device=DEV))


def group_mean(v):
    s = th.zeros(len(pairs), *v.shape[1:], dtype=v.dtype, device=DEV).index_add_(0, g, v)
    return s / cnt.view(-1, *[1] * (v.dim() - 1))


cM = C.gather(1, p[:, None, None].expand(-1, 1, 16))[:, 0]
cS = C.gather(1, q[:, None, None].expand(-1, 1, 16))[:, 0]
meas_M, meas_S = group_mean(cM), group_mean(cS)  # (90, 16)
pa, qa = th.tensor([a for a, _ in pairs], device=DEV), th.tensor([b for _, b in pairs], device=DEV)
ref_M = A_ref.mean(1).gather(1, pa[:, None, None].expand(-1, 1, 16))[:, 0]
ref_S = A_ref.mean(1).gather(1, qa[:, None, None].expand(-1, 1, 16))[:, 0]


def r2(a, b):
    return (1 - ((a - b) ** 2).sum(0) / ((b - b.mean(0)) ** 2).sum(0)).cpu().numpy()


print("R2 across the 90 (p, q) pairs of the reference coefficient vs the Gaussian mean, per neuron")
print("  on M:", r2(ref_M, meas_M))
print("  on S:", r2(ref_S, meas_S))
sv_M_ref, sv_M_meas = (ref_M @ Wo.T).gather(1, pa[:, None])[:, 0], (meas_M @ Wo.T).gather(1, pa[:, None])[:, 0]
sv_S_ref, sv_S_meas = (ref_S @ Wo.T).gather(1, qa[:, None])[:, 0], (meas_S @ Wo.T).gather(1, qa[:, None])[:, 0]
print(f"self-vote of the max over the 90 pairs: R2 {r2(sv_M_ref[:, None], sv_M_meas[:, None])[0]:.2f}; reference mean {sv_M_ref.mean():+.1f}, measured mean {sv_M_meas.mean():+.1f}")
print(f"self-vote of the 2nd max:              R2 {r2(sv_S_ref[:, None], sv_S_meas[:, None])[0]:.2f}; reference mean {sv_S_ref.mean():+.1f}, measured mean {sv_S_meas.mean():+.1f}")
print("  the max's self-vote by p, reference:", np.array([sv_M_ref[pa == t].mean().item() for t in range(10)]))
print("                           measured:", np.array([sv_M_meas[pa == t].mean().item() for t in range(10)]))
spread_ref = A_ref.std(1).mean().item()
spread_gauss = (C - group_mean(C)[g]).pow(2).mean().sqrt().item()
print(f"coefficient spread across the K reference orders {spread_ref:.2f} vs across Gaussian inputs within a pair {spread_gauss:.2f}")

A_mean = A_ref.mean(1)  # (90, 10, 16)
h_sur = th.einsum("nt,nti->ni", x, A_mean[g])
pred_sur = (h_sur @ Wo.T).argmax(1)
pred = trace(w, x)[0].argmax(1)
print(f"surrogate 'each (p, q)'s reference linear map': agreement with the model {(pred_sur == pred).double().mean():.4f}, accuracy {(pred_sur == q).double().mean():.4f}, model {(pred == q).double().mean():.4f}")
A_g = group_mean(C)
h_fit = th.einsum("nt,nti->ni", x, A_g[g])
pred_fit = (h_fit @ Wo.T).argmax(1)
print(f"surrogate 'each (p, q)'s mean Gaussian map':    agreement {(pred_fit == pred).double().mean():.4f}, accuracy {(pred_fit == q).double().mean():.4f}")
near = (ys[:, 1] - ys[:, 2] < 0.1) | (ys[:, 0] - ys[:, 1] < 0.1)
print(f"reference surrogate agreement near a tie {(pred_sur == pred)[near].double().mean():.3f}, elsewhere {(pred_sur == pred)[~near].double().mean():.3f}")
