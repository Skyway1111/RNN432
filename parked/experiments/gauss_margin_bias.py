"""Where does gauss_margin's positive bias come from: the grid, the Gaussian presumption, or the design's moments?

Twelve suite models (copies 0 and 1 of each base), at gauss_margin levels (7, 8) and (12, 12). Per cell of the (T, g1, g2)
grid, four error masses, each summed over families:
  truth:   2^27 Gaussian inputs binned into the cells, the share wrong;
  node:    mass x P(wrong) at the cell's node, 256 draws of the lower inputs per family (truncated normal below T);
  oracle:  mass x Phi(-mu / sd) with the moments of those draws;
  design:  mass x Phi(-mu / sd) with the estimator's design moments (gauss_margin.moments), without the cell spread.
So truth - node is the grid's error, node - oracle the presumption's, oracle - design the design's.
Errors are in accuracy units (estimate - truth), so positive means the estimate is too high.

Finding: research/angles/margin-gaussian.md
"""
import numpy as np
import torch as th

from az.bench import suite
from az.estimators.gauss_margin import FAMS, FREE, GAP_EDGES, Q, moments, quadrature, t_edges
from az.model import logits
from experiments.common import DEV

th.set_grad_enabled(False)
DRAWS = 256
N_TRUTH, CHUNK = 2**27, 2**20
labels, ws = suite(0.002, 32)
pick = [i * 32 + j for i in range(6) for j in range(2)]
models = [ws[i] for i in pick]
freet = th.tensor(FREE, device=DEV)
qt = th.tensor(Q, device=DEV)
pos, fam = th.tensor(FAMS, device=DEV), th.arange(len(FAMS), device=DEV)


def margin(L, q):
    other = L.clone()
    other[th.arange(len(L), device=DEV), q] = -1e9
    return L[th.arange(len(L), device=DEV), q] - other.max(1).values


def truth_cells(n_t, n_g):
    """(models, cells) error mass per cell, from N_TRUTH Gaussian inputs shared by all models."""
    te = th.tensor(t_edges(n_t)[1:-1], device=DEV)
    ge = th.tensor(GAP_EDGES[n_g][1:-1], device=DEV, dtype=th.float64)
    g = th.Generator(device=DEV).manual_seed(11)
    err = th.zeros(len(models), n_t * n_g * n_g, dtype=th.float64, device=DEV)
    for _ in range(N_TRUTH // CHUNK):
        x = th.randn(CHUNK, 10, generator=g, device=DEV, dtype=th.float64)
        top = x.topk(3, 1)
        M, S, T = top.values.T
        cell = (th.bucketize(T, te) * n_g + th.bucketize(M - S, ge)) * n_g + th.bucketize(S - T, ge)
        wrong = (logits(models, x[None].expand(len(models), -1, -1)).argmax(-1) != top.indices[:, 1]).double()
        err.index_add_(1, cell, wrong)
    return (err / N_TRUTH).cpu().numpy()


def node_cells(w, nodes):
    """(cells, families) P(wrong) and oracle Phi(-mu/sd) at each node, from DRAWS truncated-normal draws of the lower inputs."""
    g = th.Generator(device=DEV).manual_seed(5)
    emp, orc = [], []
    for t, g1, g2 in nodes:
        x = th.zeros(len(FAMS), DRAWS, 10, dtype=th.float64, device=DEV)
        u = th.rand(len(FAMS), DRAWS, 7, generator=g, device=DEV, dtype=th.float64)
        low = th.special.ndtri(u * th.special.ndtr(th.tensor(t, dtype=th.float64)))
        x.scatter_(2, freet[:, None].expand(-1, DRAWS, -1), low)
        x[fam, :, pos[:, 2]], x[fam, :, pos[:, 1]], x[fam, :, pos[:, 0]] = t, t + g2, t + g2 + g1
        m = margin(logits([w], x.reshape(1, -1, 10))[0], qt.repeat_interleave(DRAWS)).reshape(len(FAMS), DRAWS)
        emp.append((m < 0).double().mean(1))
        orc.append(th.special.ndtr(-m.mean(1) / m.std(1).clamp(min=1e-9)))
    return th.stack(emp).cpu().numpy(), th.stack(orc).cpu().numpy()


for level in [(7, 8), (12, 12)]:
    n_t, n_g = level
    mass, nodes = quadrature(*level)
    tru = truth_cells(*level)
    rows = []
    by_t = np.zeros((4, n_t))
    by_g1 = np.zeros((4, n_g))
    by_g2 = np.zeros((4, n_g))
    for i, w in enumerate(models):
        emp, orc = node_cells(w, nodes)
        mu, sd = moments(w, nodes)
        des = th.special.ndtr(-mu / sd).cpu().numpy()
        cells = np.stack([tru[i], mass * emp.mean(1), mass * orc.mean(1), mass * des.mean(1)])  # error mass per cell
        rows.append(cells.sum(1))
        c = cells.reshape(4, n_t, n_g, n_g)
        by_t += c.sum((2, 3)) / len(models)
        by_g1 += c.sum((1, 3)) / len(models)
        by_g2 += c.sum((1, 2)) / len(models)
    rows = np.array(rows)  # (models, 4) error masses
    print(f"\nlevel {level}: accuracy error (estimate - truth) per model; truth = 1 - binned error mass")
    print(f"{'model':12s} {'truth acc':>9s} {'node':>8s} {'oracle':>8s} {'design':>8s}   grid  presump  design-moments")
    for k, i in enumerate(pick):
        e = rows[k]
        print(f"{labels[i]:12s} {1 - e[0]:9.4f} {e[0] - e[1]:+8.4f} {e[0] - e[2]:+8.4f} {e[0] - e[3]:+8.4f}   "
              f"{e[0] - e[1]:+.4f} {e[1] - e[2]:+.4f} {e[2] - e[3]:+.4f}")
    d = rows[:, :1] - rows[:, 1:]
    print(f"{'mean':12s} {1 - rows[:, 0].mean():9.4f} " + " ".join(f"{v:+8.4f}" for v in d.mean(0)) +
          f"   rms {np.sqrt((d ** 2).mean(0))}")
    for name, b, edges in [("T bin", by_t, None), ("M-S bin", by_g1, GAP_EDGES[n_g]), ("S-T bin", by_g2, GAP_EDGES[n_g])]:
        print(f"  by {name} (mean over models): error mass truth | accuracy error from grid | presumption | design moments")
        for j in range(b.shape[1]):
            lab = f"[{edges[j]:.3f}, {edges[j + 1]:.3f})" if edges else f"{j}"
            print(f"    {lab:16s} {b[0, j]:.5f} | {b[0, j] - b[1, j]:+.5f} | {b[1, j] - b[2, j]:+.5f} | {b[2, j] - b[3, j]:+.5f}")
