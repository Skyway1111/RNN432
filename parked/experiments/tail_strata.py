"""How much does stratifying on the failure modes' features buy, and does it grow with the level?

Part 1, strata on mechanistic features whose masses are model-free: the order (p, q) of the top two (90 cells, exact
masses); (p, q) with the number of positive inputs (capped at 4); the 2nd largest's value S and the two tie gaps M - S
and S - T of the direction, each in 16 quantile bins of the nominal law; and (p, q) with S and M - S. Four held-out copies
per base, 2^26 directions each; the split-half Neyman gain P / ((sum w sqrt(pA)) (sum w pB / sqrt(pA))) with the
allocation from one half and the cell rates from the other (pA floored at P / 100), so the score is not fitted in sample.
Part 2, deduction inside one mode: the exact stratum "at most one positive input, label at position 9" (mass 11/10240,
sampled exactly by sign patterns and an exchangeable swap), with 5% of the nominal law as a defensive share, on the two
bases whose tail it holds (example, s3); the per-sample gain over sampling on 2^22 points.

Finding: research/findings/tail.md
"""
import torch as th

from az.estimators.modes import margin
from az.rare import LEVELS, models, truth

th.set_grad_enabled(False)
labels, ws = models("heldout")
P = truth("heldout")
DEV = "cuda"
NB = 16
z0 = th.randn(2**20, 10, generator=th.Generator(device=DEV).manual_seed(0), device=DEV, dtype=th.float64)
v0 = (z0 / z0.norm(dim=1, keepdim=True)).sort(1, descending=True).values
EDGES = [th.quantile(a, th.linspace(0, 1, NB + 1, device=DEV, dtype=th.float64)[1:-1]) for a in (v0[:, 1], v0[:, 0] - v0[:, 1], v0[:, 1] - v0[:, 2])]


def keys(u):
    v, idx = u.sort(1, descending=True)
    pq = idx[:, 0] * 10 + idx[:, 1]
    s, d1, d2 = (th.bucketize(a.contiguous(), e) for a, e in zip((v[:, 1], v[:, 0] - v[:, 1], v[:, 1] - v[:, 2]), EDGES))
    npos = (u > 0).sum(1).clamp(max=4)
    return {"(p,q)": (pq, 100), "(p,q) x positives": (pq * 5 + npos, 500), "S x (M-S) x (S-T)": ((s * NB + d1) * NB + d2, NB**3),
            "(p,q) x S x (M-S)": ((pq * NB + s) * NB + d1, 100 * NB**2)}


print(f"Part 1: split-half Neyman gain of strata, levels c = {LEVELS}")
for b in range(6):
    acc = {}
    for i in range(32 * b, 32 * b + 4):
        W = [th.tensor(a, dtype=th.float32, device=DEV) for a in ws[i]]
        g = th.Generator(device=DEV).manual_seed(50 + i)
        for it in range(16):
            x = th.randn(2**22, 10, generator=g, device=DEV, dtype=th.float64)
            u = x / x.norm(dim=1, keepdim=True)
            f = margin(W, u)
            for name, (k, n) in keys(u).items():
                C = acc.setdefault(name, th.zeros(2, len(LEVELS) + 1, n, dtype=th.float64, device=DEV))
                C[it % 2, 0] += th.bincount(k, minlength=n).double()
                for j, c in enumerate(LEVELS):
                    C[it % 2, j + 1] += th.bincount(k[f >= c], minlength=n).double()
    print(f"  {labels[32 * b].split('/')[0]}")
    for name, C in acc.items():
        w = C[:, 0].sum(0) / C[:, 0].sum()
        row = []
        for j in range(len(LEVELS)):
            pA, pB = C[0, j + 1] / C[0, 0].clamp(min=1), C[1, j + 1] / C[1, 0].clamp(min=1)
            Pt = (w * pB).sum()
            pA = pA.clamp(min=Pt.item() / 100)
            row.append((Pt / ((w * pA.sqrt()).sum() * (w * pB / pA.sqrt()).sum())).item())
        print(f"    {name:20s} " + " ".join(f"{v:6.1f}" for v in row), flush=True)

print("Part 2: exact stratum (at most one positive, label 9) with 5% nominal; per-sample gain per level")
M = 11 / 1024 / 10
for i in (0, 128):
    W = [th.tensor(a, dtype=th.float32, device=DEV) for a in ws[i]]
    g = th.Generator(device=DEV).manual_seed(1)
    n = 2**22
    nom = th.rand(n, generator=g, device=DEV) < 0.05
    y = th.randn(n, 10, generator=g, device=DEV, dtype=th.float64)
    a = y.abs()
    j = th.randint(-1, 10, (n,), generator=g, device=DEV)
    s = -a
    r = th.arange(n, device=DEV)
    s[r[j >= 0], j[j >= 0]] = a[r[j >= 0], j[j >= 0]]
    second = s.argsort(1, descending=True)[:, 1]
    s9, sk = s[:, 9].clone(), s[r, second].clone()
    s[r, second], s[:, 9] = s9, sk
    x = th.where(nom[:, None], y, s)
    u = x / x.norm(dim=1, keepdim=True)
    inside = ((u > 0).sum(1) <= 1) & (u.argsort(1)[:, -2] == 9)
    lr = 1 / (0.05 + 0.95 * inside.double() / M)
    f = margin(W, u)
    p = th.tensor(P[i], device=DEV)
    h = th.stack([(f >= c) * lr for c in LEVELS], 1)
    gain = ((1 - p) / p) / (((h - p) ** 2).mean(0) / p**2)
    print(f"  {labels[i]}: estimate / truth " + " ".join(f"{v:.2f}" for v in (h.mean(0) / p).tolist()) + "; gain " + " ".join(f"{v:.1f}" for v in gain.tolist()))
