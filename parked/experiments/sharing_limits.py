"""How much work can gauss_margin share: across the lower inputs, across cells of (T, g1, g2), across families?

  1. The most the Gaussian presumption can remove: the share of the correctness indicator's variance left inside
     (family, node) at gauss_margin's (12, 12) nodes, E[pi (1 - pi)] / (p (1 - p)) with pi = P(wrong | family, node) from
     256 truncated-normal draws of the lower inputs. Whatever is not inside must come from the quadrature over the rest.
  2. How large a cell the presumption tolerates: real Gaussian inputs binned by (family, cell), and Phi(-mu / sd) with the
     cell's own margin moments (an oracle: no design error) against the cell's share wrong. Cells: one per family, 3 x 3 and
     5 x 5 gap cells, 3 T cells x 5 x 5, and gauss_margin's (7, 8) level; the same with a mean linear in (T, g1, g2) fitted
     per cell; and scale-free cells, (sign T) x 8 x 8 bins of the gaps over |T| with the margin over |T|.
  3. Whether a balanced subset of families stands for all 720: the 90 families (p, q, L[p][q]) of an idempotent Latin
     square, which balance every pair of positions, on the true per-family error rates (same inputs as 2).
Six suite models, copy 0 of each base; 2^25 inputs per model for 2 and 3.

Finding: research/angles/margin-gaussian.md
"""
import numpy as np
import torch as th

from az.bench import suite
from az.estimators.gauss_margin import FAMS, GAP_EDGES, margins, quadrature, t_edges
from az.model import logits
from experiments.common import DEV

th.set_grad_enabled(False)
labels, ws = suite(0.002, 32)
pick = [k * 32 for k in range(6)]
models = [ws[i] for i in pick]
inf = np.inf
FIDX = th.full((1000,), -1, dtype=th.long, device=DEV)
for k, (p, q, r) in enumerate(FAMS):
    FIDX[p * 100 + q * 10 + r] = k

print("1. share of the indicator's variance left inside (family, node) at the (12, 12) nodes")
mass, nodes = quadrature(12, 12)
gen = th.Generator(device=DEV).manual_seed(2)
for i in pick:
    within = pbar = 0.0
    for c in range(len(nodes)):
        t = nodes[c, 0]
        u = th.rand(1, 720, 256, 7, generator=gen, device=DEV, dtype=th.float64)
        pi = (margins(ws[i], nodes[c : c + 1], th.special.ndtri(u * th.special.ndtr(th.tensor(t))))[0] < 0).double().mean(1)
        within += mass[c] * (pi * (1 - pi) * 256 / 255).mean().item()
        pbar += mass[c] * pi.mean().item()
    print(f"  {labels[i]:10s} P(wrong) {pbar:.4f}, share inside {within / (pbar * (1 - pbar)):.3f}", flush=True)


def batches(n=2**25, chunk=2**20):
    """Shared Gaussian inputs: family index, T, M - S, S - T, and each model's margin (models, chunk)."""
    g = th.Generator(device=DEV).manual_seed(321)
    for _ in range(n // chunk):
        x = th.randn(chunk, 10, generator=g, device=DEV, dtype=th.float64)
        top = x.topk(3, 1)
        M, S, T = top.values.T
        L = logits(models, x[None].expand(len(models), -1, -1))
        q = top.indices[:, 1][None, :, None].expand(len(models), -1, 1)
        m = L.gather(2, q)[..., 0] - L.scatter(2, q, -1e9).max(2).values
        yield FIDX[top.indices[:, 0] * 100 + top.indices[:, 1] * 10 + top.indices[:, 2]], T, M - S, S - T, m


def bins(e, v):
    return th.bucketize(v, th.tensor(np.asarray(e[1:-1], float), device=DEV))


def tensor_cells(te, ge):
    nt, ng = len(te) - 1, len(ge) - 1
    return nt * ng * ng, lambda T, g1, g2: (bins(te, T) * ng + bins(ge, g1)) * ng + bins(ge, g2)


PARTS = {"one cell": tensor_cells([-inf, inf], [0, inf]),
         "gaps 3 x 3": tensor_cells([-inf, inf], [0, 0.05, 0.3, inf]),
         "gaps 5 x 5": tensor_cells([-inf, inf], [0, 0.02, 0.06, 0.15, 0.4, inf]),
         "T 3 x gaps 5 x 5": tensor_cells([-inf, 0.3, 0.9, inf], [0, 0.02, 0.06, 0.15, 0.4, inf]),
         "level (7, 8)": tensor_cells(t_edges(7), GAP_EDGES[8])}
REL = [0, 0.015, 0.04, 0.08, 0.15, 0.3, 0.6, 1.5, inf]
n_cell = {k: v[0] for k, v in PARTS.items()}
stat = {k: th.zeros(len(models), 720 * n, 4, dtype=th.float64, device=DEV) for k, n in n_cell.items()}
stat["scale-free 2 x 8 x 8"] = th.zeros(len(models), 720 * 128, 4, dtype=th.float64, device=DEV)
lin = {k: (th.zeros(len(models), 720 * n_cell[k], 4, 4, dtype=th.float64, device=DEV),
           th.zeros(len(models), 720 * n_cell[k], 4, dtype=th.float64, device=DEV)) for k in list(PARTS)[:4]}
fam_wrong = th.zeros(len(models), 720, dtype=th.float64, device=DEV)
fam_n = th.zeros(720, dtype=th.float64, device=DEV)
for f, T, g1, g2, m in batches():
    fam_n.index_add_(0, f, th.ones_like(T))
    fam_wrong.index_add_(1, f, (m < 0).double())
    X = th.stack([th.ones_like(T), T, g1, g2], 1)
    for k, (n, cell) in PARTS.items():
        idx = f * n + cell(T, g1, g2)
        stat[k].index_add_(1, idx, th.stack([th.ones_like(m), m, m * m, (m < 0).double()], 2))
        if k in lin:
            lin[k][0].index_add_(1, idx, (X[:, :, None] * X[:, None, :])[None].expand(len(models), -1, -1, -1).contiguous())
            lin[k][1].index_add_(1, idx, X[None] * m[..., None])
    ms = m / T.abs()
    idx = f * 128 + ((T > 0).long() * 8 + bins(REL, g1 / T.abs())) * 8 + bins(REL, g2 / T.abs())
    stat["scale-free 2 x 8 x 8"].index_add_(1, idx, th.stack([th.ones_like(ms), ms, ms * ms, (ms < 0).double()], 2))

print("\n2. oracle Gaussian over each cell: estimate - empirical accuracy per model, then mean and rms")
total = fam_n.sum()
for k, s in stat.items():
    n = s[..., 0].clamp(min=1)
    mu = s[..., 1] / n
    sd = (s[..., 2] / n - mu ** 2).clamp(min=1e-18).sqrt()
    d = ((s[..., 3].sum(1) - (s[..., 0] * th.special.ndtr(-mu / sd)).sum(1)) / total).cpu().numpy()
    print(f"  {k:22s} cells per family {s.shape[1] // 720:4d}: " + " ".join(f"{e:+.4f}" for e in d) +
          f"   mean {d.mean():+.4f} rms {np.sqrt((d ** 2).mean()):.4f}")
beta = {k: th.linalg.solve(A + 1e-9 * th.eye(4, device=DEV, dtype=th.float64), b[..., None])[..., 0] for k, (A, b) in lin.items()}
sd_lin = {k: ((stat[k][..., 2] - (beta[k] * lin[k][1]).sum(-1)) / stat[k][..., 0].clamp(min=1)).clamp(min=1e-18).sqrt() for k in lin}
pw = {k: th.zeros(len(models), dtype=th.float64, device=DEV) for k in lin}
for f, T, g1, g2, m in batches():
    X = th.stack([th.ones_like(T), T, g1, g2], 1)
    for k in lin:
        n, cell = PARTS[k]
        idx = f * n + cell(T, g1, g2)
        pw[k] += th.special.ndtr(-(beta[k][:, idx] * X[None]).sum(-1) / sd_lin[k][:, idx]).sum(1)
for k in lin:
    d = ((stat[k][..., 3].sum(1) - pw[k]) / total).cpu().numpy()
    print(f"  {k + ', linear mean':22s} cells per family {n_cell[k]:4d}: " + " ".join(f"{e:+.4f}" for e in d) +
          f"   mean {d.mean():+.4f} rms {np.sqrt((d ** 2).mean()):.4f}")

print("\n3. Latin 90-family subset: its mean error rate minus all 720 families', on the true per-family rates")
L = [[None] * 10 for _ in range(10)]
for a in range(10):
    L[a][a] = a


def fill(k, cells=[(a, b) for a in range(10) for b in range(10) if a != b]):
    if k == len(cells):
        return True
    a, b = cells[k]
    for v in range(10):
        if all(L[a][j] != v for j in range(10)) and all(L[j][b] != v for j in range(10)):
            L[a][b] = v
            if fill(k + 1):
                return True
            L[a][b] = None
    return False


assert fill(0)
latin = FIDX[th.tensor([a * 100 + b * 10 + L[a][b] for a in range(10) for b in range(10) if a != b], device=DEV)]
rate = fam_wrong / fam_n
d = (rate[:, latin].mean(1) - rate.mean(1)).cpu().numpy()
print("  " + " ".join(f"{labels[i]} {e:+.4f}" for i, e in zip(pick, d)) + f"   rms {np.sqrt((d ** 2).mean()):.4f}")
