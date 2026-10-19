"""Is the gap-wedge oracle's +0.02 bias the cost of fixing the seven lower magnitudes: does varying their order and spread remove it?

Same design as wedge_feasibility (all 720 families, 12 x 12 gap grid, mean 0), with the lower seven values in 4 fixed orders
and at 3 spreads about T (0.6, 1.0, 1.5 times the mean set's). Reports the oracle's bias and RMSE for each variant set.

Finding: research/angles/tie-wedge.md
"""
import numpy as np
import torch as th
from scipy.stats import norm

from az.bench import suite, truth
from az.model import logits
from experiments.common import DEV, order_stat_means

G = np.array([0.01, 0.02, 0.04, 0.07, 0.1, 0.15, 0.22, 0.32, 0.45, 0.65, 0.95, 1.5])
edges = np.concatenate([[0], (G[1:] + G[:-1]) / 2, [np.inf]])
fine = np.arange(0.0025, 3.0, 0.005)
y3 = np.arange(-7, 4, 0.005)
base = 720 * norm.pdf(y3) * norm.cdf(y3) ** 7
F = np.zeros((len(fine), len(fine)))
for i, g1 in enumerate(fine):
    F[i] = (base[None, :] * norm.pdf(y3[None, :] + g1 + fine[:, None]) * norm.pdf(y3[None, :] + fine[:, None])).sum(1)
bi = np.searchsorted(edges, fine, side="right") - 1
mass = np.zeros((len(G), len(G)))
np.add.at(mass, (bi[:, None].repeat(len(fine), 1), bi[None, :].repeat(len(fine), 0)), F)
mass /= mass.sum()

ystar = order_stat_means()
fams = [(p, q, r) for p in range(10) for q in range(10) for r in range(10) if len({p, q, r}) == 3]
rng = np.random.default_rng(0)
orders = [np.arange(7)] + [rng.permutation(7) for _ in range(3)]
spreads = [0.6, 1.0, 1.5]
variants = [(o, s) for o in range(4) for s in spreads]
X = np.zeros((len(variants), len(fams), len(G), len(G), 10))
T = ystar[2]
for vi, (o, s) in enumerate(variants):
    low = T - s * (T - ystar[3:])
    for k, (p, q, r) in enumerate(fams):
        rest = [t for t in range(10) if t not in (p, q, r)]
        v = np.zeros(10)
        v[rest] = low[orders[o]]
        for i, g1 in enumerate(G):
            for j, g2 in enumerate(G):
                v[r], v[q], v[p] = T, T + g2, T + g2 + g1
                X[vi, k, i, j] = v
q_true = np.array([q for _, q, _ in fams])
Xt = th.tensor(X.reshape(-1, 10), dtype=th.float32, device=DEV)
qt = th.tensor(q_true, device=DEV).repeat(len(variants)).repeat_interleave(len(G) * len(G))
labels, ws = suite(0.002, 32)
p_true = truth(0.002, 32)
wrong = np.zeros((len(ws), len(variants), len(fams), len(G), len(G)), dtype=bool)
with th.no_grad():
    for s0 in range(0, len(ws), 8):
        batch = ws[s0 : s0 + 8]
        out = []
        for c in range(0, len(Xt), 2**17):
            xb = Xt[c : c + 2**17][None].expand(len(batch), -1, -1)
            out.append(logits(batch, xb).argmax(-1) != qt[c : c + 2**17][None])
        wrong[s0 : s0 + len(batch)] = th.cat(out, 1).cpu().numpy().reshape(len(batch), len(variants), len(fams), len(G), len(G))


def report(name, vidx):
    e = wrong[:, vidx].mean((1, 2))  # (models, g1, g2)
    est = 1 - (e * mass[None]).sum((1, 2))
    print(f"  {name:42s} bias {np.mean(est - p_true):+.4f}  rmse {np.sqrt(np.mean((est - p_true) ** 2)):.4f}")


print("oracle gap-wedge (720 families, 12x12 gaps, mean 0) with the lower seven values varied:")
report("1 order, spread 1.0 (as before)", [1])
report("4 orders, spread 1.0", [1, 4, 7, 10])
report("1 order, spreads 0.6 / 1.0 / 1.5", [0, 1, 2])
report("4 orders x 3 spreads", list(range(12)))
for s, name in enumerate(["spread 0.6", "spread 1.0", "spread 1.5"]):
    report(f"4 orders, {name}", [s, 3 + s, 6 + s, 9 + s])
