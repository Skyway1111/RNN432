"""Are the route-dependent coefficients set by the record history, and do satellites hold the max as a function of its age?

Each neuron's coefficient on M (and on S) is exact per input (`contributions`). Groups them by the max's position p, by (p, q),
and by the record history (at each step: new max, new 2nd max, or neither), and reports the variance explained with
held-out groups. Also each neuron's mean coefficient on M by p: an age profile if the satellites are a delay line.
Example model.

Finding: research/findings/circuit.md
"""
import numpy as np
import torch as th

from az.model import load
from experiments.common import DEV, contributions, inputs, ranks

np.set_printoptions(precision=1, suppress=True, linewidth=200)
w = load("example")
N = 2**20
x = inputs(N, seed=24)
srt, ys = ranks(x)
p, q, r = srt[:, 0], srt[:, 1], srt[:, 2]
C = contributions(w, x) / x[..., None]
cM = C.gather(1, p[:, None, None].expand(-1, 1, 16))[:, 0]  # (N, 16): coefficient on M in each neuron
cS = C.gather(1, q[:, None, None].expand(-1, 1, 16))[:, 0]
Wo = th.tensor(w.Wo, device=DEV)
self_vote_M = (cM @ Wo.T).gather(1, p[:, None])[:, 0]
self_vote_S = (cS @ Wo.T).gather(1, q[:, None])[:, 0]

M_run = th.full((N,), -1e9, dtype=th.float64, device=DEV)
S_run = M_run.clone()
ev = []
for t in range(10):
    newmax = x[:, t] > M_run
    new2 = ~newmax & (x[:, t] > S_run)
    ev.append(newmax.long() * 2 + new2.long())
    S_run = th.where(newmax, M_run, th.where(new2, x[:, t], S_run))
    M_run = th.where(newmax, x[:, t], M_run)
history = (th.stack(ev, 1) * th.tensor([3**i for i in range(10)], device=DEV)).sum(1)


def r2_heldout(u, key):
    """Variance of u (N, d) explained by group means of key, fit on even rows and scored on odd rows; groups unseen in the fit predict the global mean."""
    fit, test = th.arange(N, device=DEV) % 2 == 0, th.arange(N, device=DEV) % 2 == 1
    vals, inv = th.unique(key, return_inverse=True)
    cnt = th.zeros(len(vals), dtype=u.dtype, device=DEV).index_add_(0, inv[fit], th.ones(fit.sum(), dtype=u.dtype, device=DEV))
    s = th.zeros(len(vals), u.shape[1], dtype=u.dtype, device=DEV).index_add_(0, inv[fit], u[fit])
    mean = th.where(cnt[:, None] > 0, s / cnt.clamp(min=1)[:, None], u[fit].mean(0))
    pred = mean[inv[test]]
    return (1 - ((u[test] - pred) ** 2).sum() / ((u[test] - u[test].mean(0)) ** 2).sum()).item()


keys = {"p": p, "(p, q)": p * 10 + q, "(p, q, r)": (p * 10 + q) * 10 + r, "history": history, "history + (p, q, r)": history * 1000 + (p * 10 + q) * 10 + r}
print(f"distinct record histories: {len(th.unique(history))}")
print("held-out variance of each neuron's coefficient on M explained by:", " | ".join(keys))
for n in range(16):
    print(f"  n{n:2d}: " + "  ".join(f"{r2_heldout(cM[:, n : n + 1], k):.2f}" for k in keys.values()))
print("same for the coefficient on S:")
for n in range(16):
    print(f"  n{n:2d}: " + "  ".join(f"{r2_heldout(cS[:, n : n + 1], k):.2f}" for k in keys.values()))
print("the max's self-vote: " + "  ".join(f"{r2_heldout(self_vote_M[:, None], k):.2f}" for k in keys.values()))
print("the 2nd max's self-vote: " + "  ".join(f"{r2_heldout(self_vote_S[:, None], k):.2f}" for k in keys.values()))

print("\nmean coefficient on M by the max's position p = 0..9 (an age profile), per neuron")
for n in range(16):
    prof = np.array([cM[p == t, n].mean().item() for t in range(10)])
    print(f"  n{n:2d}: {prof}")
print("\nmean coefficient on S by the 2nd max's position q = 0..9, per neuron")
for n in range(16):
    prof = np.array([cS[q == t, n].mean().item() for t in range(10)])
    print(f"  n{n:2d}: {prof}")
