"""How few deterministic evaluations recover the margin's mean and variance over the lower seven inputs at a near-tie?

Same configurations as margin_gaussianity (720 families x 6 x 6 gaps, T at its mean). Designs over the seven lower inputs,
each at the truncated mean plus or minus the truncated sd: the centre point alone; the 8-run Hadamard (Plackett-Burman) design;
centre plus 8; the 15-run star (centre and two axial points per input); the 16-run fold-over. Each design's sample mean and sd
give Phi(-mu / sd); the empirical reference is 256 draws of the lower inputs. Scored by the mass-weighted near-tie error and
by the per-cell RMS error of P(wrong), on six models.

Finding: research/angles/margin-gaussian.md
"""
import numpy as np
import torch as th
from scipy.linalg import hadamard
from scipy.stats import norm

from az.bench import suite
from az.model import load, trace
from experiments.common import DEV, order_stat_means

np.set_printoptions(precision=4, suppress=True, linewidth=200)
th.set_grad_enabled(False)
G = np.array([0.01, 0.03, 0.07, 0.15, 0.3, 0.6])
DRAWS = 256
ystar = order_stat_means()
T = ystar[2]
mu_T = -norm.pdf(T) / norm.cdf(T)
sd_T = np.sqrt(1 - T * norm.pdf(T) / norm.cdf(T) - (norm.pdf(T) / norm.cdf(T)) ** 2)
fams = [(p, q, r) for p in range(10) for q in range(10) for r in range(10) if len({p, q, r}) == 3]
freet = th.tensor([[t for t in range(10) if t not in f] for f in fams], device=DEV)
qt = th.tensor([q for _, q, _ in fams], device=DEV)

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
W = mass[:, :, None] / 720

H8 = hadamard(8)[:, 1:]  # 8 runs x 7 factors, +-1
star = np.vstack([np.zeros(7), np.eye(7), -np.eye(7)])
designs = {
    "centre (1 run)": np.zeros((1, 7)),
    "Hadamard 8": H8.astype(float),
    "centre + Hadamard 8 (9)": np.vstack([np.zeros(7), H8]),
    "star (15)": star,
    "fold-over 16": np.vstack([H8, -H8]).astype(float),
}


def margins(L, q):
    other = L.clone()
    other[th.arange(len(L), device=DEV), q] = -1e9
    return L[th.arange(len(L), device=DEV), q] - other.max(1).values


gen = th.Generator(device=DEV).manual_seed(3)
labels, ws = suite(0.002, 32)
models = [("example", load("example")), ("s0", load("s0")), ("s4", load("s4")), (labels[0], ws[0]), (labels[32], ws[32]), (labels[160], ws[160])]
print(f"{'model':10s} " + " ".join(f"{k:>24s}" for k in ["empirical"] + list(designs)))
for name, w in models:
    emp = np.zeros((len(G), len(G), 720))
    pred = {k: np.zeros_like(emp) for k in designs}
    for i, g1 in enumerate(G):
        for j, g2 in enumerate(G):
            top = th.zeros(720, 10, dtype=th.float64, device=DEV)
            for k, (p, q, r) in enumerate(fams):
                top[k, r], top[k, q], top[k, p] = T, T + g2, T + g2 + g1
            u = th.rand(720, DRAWS, 7, generator=gen, device=DEV, dtype=th.float64)
            low = th.tensor(norm.ppf(u.cpu().numpy() * norm.cdf(T)), device=DEV)
            x = top[:, None, :].expand(-1, DRAWS, -1).clone()
            x.scatter_(2, freet[:, None, :].expand(-1, DRAWS, -1), low)
            m = margins(trace(w, x.reshape(-1, 10))[0], qt.repeat_interleave(DRAWS)).reshape(720, DRAWS)
            emp[i, j] = (m < 0).double().mean(1).cpu().numpy()
            for dname, D in designs.items():
                lowd = th.tensor(mu_T + sd_T * D, dtype=th.float64, device=DEV)  # (runs, 7)
                xd = top[:, None, :].expand(-1, len(D), -1).clone()
                xd.scatter_(2, freet[:, None, :].expand(-1, len(D), -1), lowd[None].expand(720, -1, -1))
                md = margins(trace(w, xd.reshape(-1, 10))[0], qt.repeat_interleave(len(D))).reshape(720, len(D))
                mu = md.mean(1)
                sd = md.std(1) if len(D) > 1 else th.full_like(mu, 1e-9)
                if dname == "star (15)":  # a star design's spread is per-axis: variance = sum over axes of half the squared axial differences
                    sd = (((md[:, 1:8] - md[:, 8:15]) / 2) ** 2).sum(1).sqrt()
                    mu = md[:, 0] + ((md[:, 1:8] + md[:, 8:15]) / 2 - md[:, 0:1]).sum(1)
                pred[dname][i, j] = norm.cdf(-(mu / sd.clamp(min=1e-9)).cpu().numpy())
    e_emp = (W * emp).sum()
    cells = {k: (np.sqrt(np.mean((v - emp) ** 2)), (W * v).sum()) for k, v in pred.items()}
    print(f"{name:10s} {e_emp:24.4f} " + " ".join(f"{v[1]:+.4f} (cell rms {v[0]:.3f})".rjust(24) for v in cells.values()))
print("\ncolumns: mass-weighted near-tie error over the grid (T at its mean); in brackets the RMS error of P(wrong) over the 25,920 cells")
