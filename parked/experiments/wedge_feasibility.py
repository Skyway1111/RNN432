"""Can a gap-wedge estimator beat sampling at low budgets: error = gap density x fraction of designed configurations wrong?

Inputs are designed, not sampled: one per (p, q, r) family with the seven lower values at the mean magnitude set in a fixed
order, on a grid over the two top gaps (M - S, S - T) and the input mean, which are independent of the ordering. The gap
density and the mean's density are exact order-statistic facts. The estimate is 1 - sum over the grid of density x share of
families wrong. Oracle: all 720 families and the full grid; then systematic family subsets and sub-grids, each with its op
cost, against sampling's RMSE at the same ops, on the 192-model suite.

Finding: research/angles/tie-wedge.md
"""
import numpy as np
import torch as th
from scipy.stats import norm

from az.bench import suite, truth
from az.model import SAMPLE_OPS, logits
from experiments.common import DEV, order_stat_means

G = np.array([0.01, 0.02, 0.04, 0.07, 0.1, 0.15, 0.22, 0.32, 0.45, 0.65, 0.95, 1.5])  # grid for each gap
CM = np.array([-0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9])  # grid for the input mean
N_IN = 10

# bin masses of the gap pair (Y1 - Y2, Y2 - Y3) for 10 standard normals, on a fine lattice
edges = np.concatenate([[0], (G[1:] + G[:-1]) / 2, [np.inf]])
fine = np.arange(0.0025, 3.0, 0.005)
y3 = np.arange(-7, 4, 0.005)
base = 720 * norm.pdf(y3) * norm.cdf(y3) ** 7
F = np.zeros((len(fine), len(fine)))
for i, g1 in enumerate(fine):
    F[i] = (base[None, :] * norm.pdf(y3[None, :] + g1 + fine[:, None]) * norm.pdf(y3[None, :] + fine[:, None])).sum(1) * 0.005
F *= 0.005 ** 2
bi = np.searchsorted(edges, fine, side="right") - 1
mass_g = np.zeros((len(G), len(G)))
np.add.at(mass_g, (bi[:, None].repeat(len(fine), 1), bi[None, :].repeat(len(fine), 0)), F)
mass_g /= mass_g.sum()
cedges = np.concatenate([[-np.inf], (CM[1:] + CM[:-1]) / 2, [np.inf]])
mass_c = np.diff(norm.cdf(cedges / np.sqrt(0.1)))
print(f"gap bin masses sum to 1 after normalising from {F.sum():.4f}; P(S-T < 0.085) {mass_g[:, :5].sum():.3f}; mean bins {mass_c.round(3)}")

# designed inputs: family (p, q, r) x g1 x g2 x c
ystar = order_stat_means()
fams = [(p, q, r) for p in range(10) for q in range(10) for r in range(10) if len({p, q, r}) == 3]
X = np.zeros((len(fams), len(G), len(G), len(CM), N_IN))
for k, (p, q, r) in enumerate(fams):
    rest = [t for t in range(10) if t not in (p, q, r)]
    v = np.zeros(N_IN)
    v[rest] = ystar[3:]
    T = ystar[2]
    for i, g1 in enumerate(G):
        for j, g2 in enumerate(G):
            v[r], v[q], v[p] = T, T + g2, T + g2 + g1
            X[k, i, j] = v[None, :] + CM[:, None]
q_true = np.array([q for _, q, _ in fams])
Xt = th.tensor(X.reshape(-1, N_IN), dtype=th.float32, device=DEV)
qt = th.tensor(q_true, device=DEV).repeat_interleave(len(G) * len(G) * len(CM))

labels, ws = suite(0.002, 32)
p_true = truth(0.002, 32)
wrong = np.zeros((len(ws), len(fams), len(G), len(G), len(CM)), dtype=bool)
with th.no_grad():
    for s in range(0, len(ws), 8):
        batch = ws[s : s + 8]
        out = []
        for c in range(0, len(Xt), 2**17):
            xb = Xt[c : c + 2**17][None].expand(len(batch), -1, -1)
            out.append(logits(batch, xb).argmax(-1) != qt[c : c + 2**17][None])
        wrong[s : s + len(batch)] = th.cat(out, 1).cpu().numpy().reshape(len(batch), len(fams), len(G), len(G), len(CM))
print(f"designed inputs {len(Xt)} per model; oracle error on the mean-set family inputs (g = typical, c = 0): {wrong[:, :, 8, 8, 3].mean():.4f}")

W3 = mass_g[:, :, None] * mass_c[None, None, :]  # (g1, g2, c) weights
rows = []


def score(name, fam_idx, gi, gj, ci, weights):
    e = wrong[:, fam_idx][:, :, gi][:, :, :, gj][:, :, :, :, ci].mean(1)  # (models, |gi|, |gj|, |ci|)
    est = 1 - (e * weights[None]).sum((1, 2, 3))
    ops = len(fam_idx) * len(gi) * len(gj) * len(ci) * SAMPLE_OPS
    rmse = np.sqrt(np.mean((est - p_true) ** 2))
    samp = np.sqrt(np.mean(p_true * (1 - p_true)) / max(ops // SAMPLE_OPS, 1))
    rows.append((name, len(fam_idx), ops, rmse, np.mean(est - p_true), samp, rmse**2 / samp**2))


all_g = list(range(len(G)))
for K in [720, 180, 72, 36, 18, 9]:
    fam_idx = list(range(0, 720, 720 // K))[:K]
    score("full grid 12x12x7", fam_idx, all_g, all_g, all_g[:7], W3)
    w2 = (mass_g[:, :, None] * np.array([[[1.0]]]))
    score("gaps only 12x12, c=0", fam_idx, all_g, all_g, [3], w2)
    wg2 = mass_g.sum(0)[None, :, None]
    score("S-T only 12, M-S typical, c=0", fam_idx, [8], all_g, [3], wg2)
    sub = [1, 4, 6, 8]  # 0.02, 0.1, 0.22, 0.45
    sub_edges = np.array([0, 0.05, 0.15, 0.3, np.inf])
    wsub = np.array([mass_g.sum(0)[(G >= lo) & (G < hi)].sum() for lo, hi in zip(sub_edges[:-1], sub_edges[1:])])[None, :, None]
    score("S-T only 4 points, M-S typical, c=0", fam_idx, [8], sub, [3], wsub)
    wsub2 = (mass_g[np.ix_(sub, sub)] / mass_g[np.ix_(sub, sub)].sum())[:, :, None]
    score("gaps 4x4, c=0", fam_idx, sub, sub, [3], wsub2)

print(f"\nsuite: {len(ws)} models, truth mean {p_true.mean():.4f} sd {p_true.std():.4f}")
print(f"{'design':38s} {'fams':>5s} {'ops':>9s} {'rmse':>7s} {'bias':>8s} {'samp':>7s} {'mse/samp':>8s}")
for name, K, ops, rmse, bias, samp, ratio in rows:
    print(f"{name:38s} {K:5d} {ops:9.2e} {rmse:7.4f} {bias:+8.4f} {samp:7.4f} {ratio:8.3f}")
