"""At a near-tie, is the margin Gaussian over the seven lower inputs, with mean and variance deducible from the weights?

Configuration = family (p, q, r) and the top three magnitudes; the seven lower inputs are i.i.d. N(0,1) truncated above at T,
one per free position, which is their exact conditional law. For each configuration on a grid over the two top gaps:
  empirical: P(wrong), the margin's mean and sd over 256 draws of the lower inputs;
  deductive: the margin and its gradient at the symmetric reference input (every free position at the truncated mean),
             giving mu and sigma^2 = Var(truncated) x sum of squared coefficients on the free positions, from one pass.
Tests P(wrong) = Phi(-mu / sigma) with deductive moments, and with oracle (empirical) moments to isolate Gaussianity.
Aggregates over families and the exact gap density into near-tie error mass, the quantity an estimator would need.

Finding: research/angles/margin-gaussian.md
"""
import numpy as np
import torch as th
from scipy.stats import norm

from az.bench import suite
from az.model import load, trace
from experiments.common import DEV, order_stat_means

np.set_printoptions(precision=3, suppress=True, linewidth=200)
th.set_grad_enabled(False)
G = np.array([0.01, 0.03, 0.07, 0.15, 0.3, 0.6])
DRAWS = 256
ystar = order_stat_means()
T = ystar[2]
mu_T = -norm.pdf(T) / norm.cdf(T)
var_T = 1 - T * norm.pdf(T) / norm.cdf(T) - (norm.pdf(T) / norm.cdf(T)) ** 2
fams = [(p, q, r) for p in range(10) for q in range(10) for r in range(10) if len({p, q, r}) == 3]
free = np.array([[t for t in range(10) if t not in f] for f in fams])  # (720, 7)

# gap bin masses (same lattice as wedge_feasibility)
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

gen = th.Generator(device=DEV).manual_seed(3)
labels, ws = suite(0.002, 32)
models = [("example", load("example")), ("s0", load("s0")), ("s4", load("s4")), (labels[0], ws[0]), (labels[32], ws[32]), (labels[160], ws[160])]
qt = th.tensor([q for _, q, _ in fams], device=DEV)
freet = th.tensor(free, device=DEV)


def margins(L, q):
    other = L.clone()
    other[th.arange(len(L), device=DEV), q] = -1e9
    return L[th.arange(len(L), device=DEV), q] - other.max(1).values


for name, w in models:
    emp_wrong = np.zeros((len(G), len(G), 720))
    emp_mu = np.zeros_like(emp_wrong)
    emp_sd = np.zeros_like(emp_wrong)
    ded_mu = np.zeros_like(emp_wrong)
    ded_sd = np.zeros_like(emp_wrong)
    skew = []
    for i, g1 in enumerate(G):
        for j, g2 in enumerate(G):
            top = th.zeros(720, 10, dtype=th.float64, device=DEV)
            for k, (p, q, r) in enumerate(fams):
                top[k, r], top[k, q], top[k, p] = T, T + g2, T + g2 + g1
            # deductive: one pass with the Jacobian at the symmetric reference
            ref = top.clone()
            ref.scatter_(1, freet, th.full((720, 7), mu_T, dtype=th.float64, device=DEV))
            L, _, J = trace(w, ref, jacobian=True)  # J: (720, 10 logits, 10 inputs)
            comp = L.clone()
            comp[th.arange(720, device=DEV), qt] = -1e9
            c = comp.argmax(1)
            m_ref = margins(L, qt)
            Jm = J[th.arange(720, device=DEV), qt] - J[th.arange(720, device=DEV), c]  # (720, 10): d margin / d x
            coef_free = Jm.gather(1, freet)
            ded_mu[i, j] = m_ref.cpu().numpy()
            ded_sd[i, j] = (var_T * (coef_free ** 2).sum(1)).sqrt().cpu().numpy()
            # empirical: 256 draws of the lower inputs per family
            u = th.rand(720, DRAWS, 7, generator=gen, device=DEV, dtype=th.float64)
            low = th.tensor(norm.ppf(u.cpu().numpy() * norm.cdf(T)), device=DEV)  # truncated normals below T
            x = top[:, None, :].expand(-1, DRAWS, -1).clone()
            x.scatter_(2, freet[:, None, :].expand(-1, DRAWS, -1), low)
            Lx = trace(w, x.reshape(-1, 10))[0]
            m = margins(Lx, qt.repeat_interleave(DRAWS)).reshape(720, DRAWS)
            emp_wrong[i, j] = (m < 0).double().mean(1).cpu().numpy()
            emp_mu[i, j] = m.mean(1).cpu().numpy()
            emp_sd[i, j] = m.std(1).cpu().numpy()
            z = (m - m.mean(1, keepdim=True)) / m.std(1, keepdim=True).clamp(min=1e-9)
            skew.append((z ** 3).mean(1).abs().mean().item())
    p_ded = norm.cdf(-ded_mu / np.maximum(ded_sd, 1e-9))
    p_orc = norm.cdf(-emp_mu / np.maximum(emp_sd, 1e-9))
    W = mass[:, :, None] / 720
    err_emp, err_ded, err_orc = (W * emp_wrong).sum(), (W * p_ded).sum(), (W * p_orc).sum()
    print(f"\n{name}: near-tie error mass over the grid (all gaps, mass-weighted): empirical {err_emp:.4f}, Gaussian with oracle moments {err_orc:.4f}, "
          f"Gaussian with deductive moments {err_ded:.4f}   (gaps < 0.1 only: {(W[:3, :3] * emp_wrong[:3, :3]).sum():.4f} / {(W[:3, :3] * p_orc[:3, :3]).sum():.4f} / {(W[:3, :3] * p_ded[:3, :3]).sum():.4f})")
    print(f"  mean |skewness| of the margin over the lower inputs {np.mean(skew):.2f}; sd ratio deductive/empirical median {np.median(ded_sd / np.maximum(emp_sd, 1e-9)):.2f}; "
          f"mean offset (deductive - empirical) median {np.median(ded_mu - emp_mu):+.2f}, mad {np.median(np.abs(ded_mu - emp_mu)):.2f}")
    print("  P(wrong) by gap cell (rows S-T, cols M-S), family-averaged: empirical | oracle Gaussian | deductive Gaussian")
    for j, g2 in enumerate(G):
        print(f"    S-T {g2:4.2f}: " + " ".join(f"{emp_wrong[i, j].mean():.3f}" for i in range(len(G))) + " | " + " ".join(f"{p_orc[i, j].mean():.3f}" for i in range(len(G))) + " | " + " ".join(f"{p_ded[i, j].mean():.3f}" for i in range(len(G))))
    bins = [0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 1.01]
    print("  calibration over all 25,920 configurations, binned by the deductive prediction: [lo, hi) share | mean predicted | mean empirical")
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (p_ded >= lo) & (p_ded < hi)
        if sel.any():
            print(f"    [{lo:.2f}, {hi:.2f}) {sel.mean():.3f} | {p_ded[sel].mean():.3f} | {emp_wrong[sel].mean():.3f}")
