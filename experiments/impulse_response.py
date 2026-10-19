"""Are the neurons' tuning bumps over the max's position the impulse responses of the gated recurrence?

A single impulse x = e_p leaves every earlier state at zero, so the route is canonical and h_10 is exactly the neuron
profile for a max of size 1 at age 10 - p. Compares those profiles, and the votes W_oh h_10 they imply, with the mean
coefficients measured on Gaussian inputs (age_code, votes). Then the same for the 2nd max from x = 2 e_p + e_q.
Example model.

Finding: research/findings/votes.md
"""
import numpy as np
import torch as th

from az.model import load, trace
from experiments.common import DEV, contributions, inputs, ranks

np.set_printoptions(precision=1, suppress=True, linewidth=200)
w = load("example")
Wo = th.tensor(w.Wo, device=DEV)

x = inputs(2**20, seed=24)  # same inputs as age_code
srt, ys = ranks(x)
p, q = srt[:, 0], srt[:, 1]
C = contributions(w, x) / x[..., None]
cM = C.gather(1, p[:, None, None].expand(-1, 1, 16))[:, 0]
cS = C.gather(1, q[:, None, None].expand(-1, 1, 16))[:, 0]
measured_M = th.stack([cM[p == t].mean(0) for t in range(10)])  # (10 positions, 16)
measured_S = th.stack([cS[q == t].mean(0) for t in range(10)])

single = th.eye(10, dtype=th.float64, device=DEV)
imp_M = th.relu(trace(w, single)[1][:, -1])  # (10, 16): response to a unit max at position p
print("impulse response (a unit input at position p) vs the mean coefficient on M at p, per neuron: R2 across positions, and both profiles")
r2s = []
for n in range(16):
    a, b = imp_M[:, n], measured_M[:, n]
    r2 = 1 - ((a - b) ** 2).sum() / ((b - b.mean()) ** 2).sum()
    r2s.append(r2.item())
    print(f"  n{n:2d}: R2 {r2:+.2f}  impulse {a.cpu().numpy()}  measured {b.cpu().numpy()}")
print(f"  median R2 {np.median(r2s):.2f}")
vote_imp = (imp_M @ Wo.T).diagonal()  # self-vote of a unit max at each position
vote_meas = (measured_M @ Wo.T).diagonal()
print(f"  the max's self-vote by p: from impulses {vote_imp.cpu().numpy()}\n                            measured mean  {vote_meas.cpu().numpy()}")

print("\n2nd max: response to x = 2 e_p + e_q minus the max's own impulse response, vs the mean coefficient on S at q")
two = th.zeros(90, 10, dtype=th.float64, device=DEV)
pairs = [(a, b) for a in range(10) for b in range(10) if a != b]
for i, (a, b) in enumerate(pairs):
    two[i, a], two[i, b] = 2.0, 1.0
h2 = th.relu(trace(w, two)[1][:, -1])
imp_S = th.zeros(10, 16, dtype=th.float64, device=DEV)
for i, (a, b) in enumerate(pairs):
    imp_S[b] += (h2[i] - 2 * imp_M[a]) / 9  # average over p of the part attributable to the 2nd max
r2s = []
for n in range(16):
    a, b = imp_S[:, n], measured_S[:, n]
    r2 = 1 - ((a - b) ** 2).sum() / ((b - b.mean()) ** 2).sum()
    r2s.append(r2.item())
    print(f"  n{n:2d}: R2 {r2:+.2f}  two-impulse {a.cpu().numpy()}  measured {b.cpu().numpy()}")
print(f"  median R2 {np.median(r2s):.2f}")
print(f"  the 2nd max's self-vote by q: from two impulses {(imp_S @ Wo.T).diagonal().cpu().numpy()}\n                               measured mean    {(measured_S @ Wo.T).diagonal().cpu().numpy()}")

print("\nhow the max's impulse response is built: neurons on at each step after a unit impulse at p = 2 (canonical order 2 4 7 6 1 8 | 0 3 5 9 10 11 12 13 14 15), and h")
order = [2, 4, 7, 6, 1, 8, 0, 3, 5, 9, 10, 11, 12, 13, 14, 15]
_, pre, _ = trace(w, single[2:3])
for t in range(2, 10):
    h = th.relu(pre[0, t])
    print(f"  after step {t}: on {''.join('1' if h[n] > 0 else '.' for n in order[:6])} {''.join('1' if h[n] > 0 else '.' for n in order[6:])}   h {h[order].cpu().numpy()}")
