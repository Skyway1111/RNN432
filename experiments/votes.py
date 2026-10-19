"""What does each neuron hold, and how does each input vote, written as exact coefficients on the inputs?

The model is piecewise linear with no biases, so h_10[n] = sum_t c[n, t](x) x_t and l_i = sum_t J[i, t](x) x_t exactly, with
c = P_t W_hi the per-unit contribution (`contributions` / x_t) and J = W_oh c. Averaging the coefficients by the rank of x_t
gives each neuron a formula in M, S, T, ... and each rank a vote vector over positions; the sd says how route-dependent it is.
Example model.

Finding: research/findings/circuit.md
"""
import numpy as np
import torch as th

from az.model import load
from experiments.common import DEV, contributions, inputs, ranks

np.set_printoptions(precision=1, suppress=True, linewidth=200)
w = load("example")
x = inputs(2**18, seed=23)
srt, ys = ranks(x)
p, q, r = srt[:, 0], srt[:, 1], srt[:, 2]
C = contributions(w, x) / x[..., None]  # (N, 10 steps, 16): coefficient of x_t in each neuron
by_rank = C.gather(1, srt[..., None].expand(-1, -1, 16))  # (N, 10 ranks, 16)
Wo = th.tensor(w.Wo, device=DEV)

print("(1) each neuron's coefficient on the input of rank k (1 = max, 2 = 2nd max, 3 = 3rd): mean, and sd in brackets")
print("     rank:   1 (M)          2 (S)          3 (T)          4              5            6-10 summed")
for n in range(16):
    m, s = by_rank[:, :, n].mean(0).cpu().numpy(), by_rank[:, :, n].std(0).cpu().numpy()
    low = by_rank[:, 5:, n].sum(1)
    print(f"  n{n:2d}: " + "  ".join(f"{m[k]:+6.1f} [{s[k]:4.1f}]" for k in range(5)) + f"  {low.mean():+6.1f} [{low.std():4.1f}]   mean h {(by_rank[:, :, n] * ys).sum(1).mean():5.1f}")

J = C @ Wo.T  # (N, 10 steps, 10 positions): vote of x_t per unit for each position
steps = th.arange(10, device=DEV)[None].expand(len(x), -1)
self_vote = J.gather(2, steps[..., None])[..., 0]  # (N, 10 steps)
vote_q = J.gather(2, q[:, None, None].expand(-1, 10, 1))[..., 0]
vote_p = J.gather(2, p[:, None, None].expand(-1, 10, 1))[..., 0]
vote_r = J.gather(2, r[:, None, None].expand(-1, 10, 1))[..., 0]
rank_of_step = srt.argsort(1)  # (N, 10 steps): rank index of x_t
print("\n(2) votes per unit input, by the input's rank: for its own position, for q, for p, for r, and the mean over the other positions")
for k in range(10):
    m = rank_of_step == k
    others = (J.sum(2) - self_vote)[m] / 9
    print(f"  rank {k + 1:2d}: self {self_vote[m].mean():+6.1f} [{self_vote[m].std():5.1f}]  for q {vote_q[m].mean():+6.1f}  for p {vote_p[m].mean():+6.1f}  "
          f"for r {vote_r[m].mean():+6.1f}  others {others.mean():+6.1f}")

print("\n(3) the max's self-vote by its position, and by whether it came before or after q")
m1 = rank_of_step == 0
for t in range(10):
    mt = m1 & (steps == t)
    before = mt & (p < q)[:, None]
    after = mt & (p > q)[:, None]
    print(f"  p = {t}: self {self_vote[mt].mean():+6.1f}  before q {self_vote[before].mean():+6.1f}  after q {self_vote[after].mean():+6.1f}")

print("\n(4) the 2nd max's self-vote by its position, and by case")
m2 = rank_of_step == 1
for t in range(10):
    mt = m2 & (steps == t)
    dem = mt & (q < p)[:, None]
    run = mt & (q > p)[:, None]
    print(f"  q = {t}: self {self_vote[mt].mean():+6.1f} [{self_vote[mt].std():5.1f}]  demoted {self_vote[dem].mean():+6.1f}  runner-up {self_vote[run].mean():+6.1f}")


def r2_by_group(u, key):
    vals, inv, cnt = th.unique(key, return_inverse=True, return_counts=True)
    s = th.zeros(len(vals), dtype=u.dtype, device=u.device).index_add_(0, inv, u)
    mean = s / cnt
    return (1 - ((u - mean[inv]) ** 2).sum() / ((u - u.mean()) ** 2).sum()).item()


sv2 = self_vote[m2]
print(f"\n(5) variance of the 2nd max's self-vote explained by q: {r2_by_group(sv2, q):.3f}; by (q, p): {r2_by_group(sv2, q * 10 + p):.3f}; "
      f"by (q, p, r): {r2_by_group(sv2, (q * 10 + p) * 10 + r):.3f}")
sv1 = self_vote[m1]
print(f"    the max's self-vote: by p {r2_by_group(sv1, p):.3f}; by (p, q) {r2_by_group(sv1, p * 10 + q):.3f}; by (p, q, r) {r2_by_group(sv1, (p * 10 + q) * 10 + r):.3f}")
