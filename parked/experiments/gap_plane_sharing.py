"""Can a few gap nodes per (family, T) stand for the whole (M - S, S - T) plane in gauss_margin?

Within a route the logits are affine in the inputs, and with T and the design's lower values fixed the inputs are affine in
the gaps (g1, g2) = (M - S, S - T). So each design run's logit differences l_q - l_j might be near-affine over the plane,
and a few nodes might give them everywhere.
Reference: the 8-run design at every point of a fine gap lattice (24 x 24, denser near 0), Phi(-mu / sd) integrated against the
exact gap density given T. Cheap: the design at a few nodes, one quantity interpolated over the lattice (bilinear on the node
tensor grid, extrapolated linearly from the end segments), then the same integral. Quantities: each run's logit differences
(the margin is their minimum, so competitor switches stay exact); the margin's mu and sd; the probit mu / sd; P(wrong).
Six suite models (copy 0 of each base), T at its 3%, 10%, 50% and 90% quantiles. Reported per scheme: the cheap estimate's
accuracy given T minus the reference's, mean and rms over models and T.

Finding: research/angles/margin-gaussian.md
"""
import numpy as np
import torch as th
from scipy.stats import norm

from az.bench import suite
from az.estimators.gauss_margin import FAMS, FREE, H8 as DESIGN, Q
from az.model import logits
from experiments.common import DEV

th.set_grad_enabled(False)
labels, ws = suite(0.002, 32)
models = [(labels[i * 32], ws[i * 32]) for i in range(6)]
pos = th.tensor(FAMS, device=DEV)
freet = th.tensor(FREE, device=DEV)
qt = th.tensor(Q, device=DEV)
LAT = np.concatenate([np.linspace(0.0025, 0.1, 10), np.linspace(0.13, 0.6, 8), [0.75, 1.0, 1.3, 1.7, 2.2, 3.0]])  # 24 points
T_QUANT = {"T 3%": -0.11, "T 10%": 0.126, "T 50%": 0.647, "T 90%": 1.196}  # quantiles of the 3rd largest of 10
NODES = {"3x3": [0.01, 0.1, 0.6], "4x4": [0.005, 0.04, 0.2, 0.8], "5x5": [0.005, 0.03, 0.1, 0.3, 1.0],
         "6x6": [0.004, 0.02, 0.06, 0.15, 0.4, 1.2]}
TARGETS = ("logit diffs", "mu, sd", "probit", "P(wrong)")


def diffs(w, t, g1, g2):
    """(len(g1), families, runs, 10) logit differences l_q - l_j for the design at gap points (g1[i], g2[i])."""
    a = norm.pdf(t) / norm.cdf(t)
    low = th.tensor(-a + np.sqrt(1 - t * a - a * a) * DESIGN, device=DEV)  # (8, 7)
    n = len(g1)
    x = th.zeros(n, len(FAMS), len(DESIGN), 10, dtype=th.float64, device=DEV)
    x.scatter_(3, freet[None, :, None].expand(n, -1, len(DESIGN), -1), low[None, None].expand(n, len(FAMS), -1, -1).contiguous())
    g1, g2 = th.tensor(g1, device=DEV), th.tensor(g2, device=DEV)
    top = th.stack([t + g2 + g1, t + g2, th.full_like(g1, t)], 1)  # (n, 3): M, S, T
    x.scatter_(3, pos[None, :, None].expand(n, -1, len(DESIGN), -1), top[:, None, None].expand(-1, len(FAMS), len(DESIGN), -1).contiguous())
    L = logits([w], x.reshape(1, -1, 10))[0].reshape(n, len(FAMS), len(DESIGN), 10)
    return L.gather(3, qt[None, :, None, None].expand(n, -1, len(DESIGN), 1)) - L


def moments(d):
    """Margin mean and sd over runs from logit differences (..., families, runs, 10)."""
    m = d.scatter(-1, qt[None, :, None, None].expand(*d.shape[:-1], 1), 1e9).min(-1).values
    return m.mean(-1), m.std(-1).clamp(min=1e-9)


def bilinear(nodes, f):
    """f on the node tensor grid (k, k, ...) interpolated to the lattice points (G1, G2)."""
    def axis(x):
        i = np.clip(np.searchsorted(nodes, x) - 1, 0, len(nodes) - 2)
        u = th.tensor((x - nodes[i]) / (nodes[i + 1] - nodes[i]), device=DEV).view(-1, *[1] * (f.dim() - 2))
        return i, u
    i, u = axis(G1)
    j, v = axis(G2)
    return (1 - u) * (1 - v) * f[i, j] + u * (1 - v) * f[i + 1, j] + (1 - u) * v * f[i, j + 1] + u * v * f[i + 1, j + 1]


G1, G2 = (a.ravel() for a in np.meshgrid(LAT, LAT, indexing="ij"))
dl = np.gradient(LAT)
err = {(n, k): [] for n in NODES for k in TARGETS}
for name, w in models:
    for tname, t in T_QUANT.items():
        dens = th.tensor(norm.pdf(t + G2) * norm.pdf(t + G2 + G1) * np.outer(dl, dl).ravel(), device=DEV)
        dens /= dens.sum()
        mu, sd = moments(diffs(w, t, G1, G2))
        ref = (dens * th.special.ndtr(-mu / sd).mean(1)).sum().item()
        for nname, nodes in NODES.items():
            nodes = np.array(nodes)
            N1, N2 = (a.ravel() for a in np.meshgrid(nodes, nodes, indexing="ij"))
            dn = diffs(w, t, N1, N2)
            mu_n, sd_n = moments(dn)
            k = len(nodes)
            grid = lambda f: f.reshape(k, k, *f.shape[1:])  # noqa: E731
            pw = {
                "logit diffs": lambda: (lambda m: th.special.ndtr(-m[0] / m[1]))(moments(bilinear(nodes, grid(dn)))),
                "mu, sd": lambda: th.special.ndtr(-bilinear(nodes, grid(mu_n)) / bilinear(nodes, grid(sd_n)).clamp(min=1e-9)),
                "probit": lambda: th.special.ndtr(bilinear(nodes, grid(-mu_n / sd_n))),
                "P(wrong)": lambda: bilinear(nodes, grid(th.special.ndtr(-mu_n / sd_n))).clamp(0, 1),
            }
            for tg in TARGETS:
                err[nname, tg].append(ref - (dens * pw[tg]().mean(1)).sum().item())
        print(f"{name:10s} {tname:6s} ref error given T {ref:.4f}; 5x5 errors " +
              " ".join(f"{tg} {err['5x5', tg][-1]:+.5f}" for tg in TARGETS), flush=True)
print("\ncheap minus reference accuracy given T: mean, rms over 6 models x 4 values of T")
print(f"{'nodes':6s} " + " ".join(f"{tg:>20s}" for tg in TARGETS))
for nname in NODES:
    print(f"{nname:6s} " + " ".join(f"{np.mean(err[nname, tg]):+.5f} {np.sqrt(np.mean(np.square(err[nname, tg]))):.5f}".rjust(20) for tg in TARGETS))
