"""Does integrating some inputs exactly along lines help a deterministic estimator?

Correctness is piecewise constant along any line, so an exact 1-D integral replaces a quadrature direction. Four checks:
  1. The share of the indicator's variance left after integrating the last one or two inputs exactly, Var(E[1 | prefix])
     / (p (1 - p)), with 200 or 40 x 40 equal-mass nodes standing in for the exact integral; 4,096 prefixes, four models.
  2. Great circles: correctness depends on direction only, so a circle through 0 is integrated exactly by its sectors.
     Sectors per circle (gate, ordering or decision changes over 20,000 angles) and the sd of the correct fraction over
     120 circles give the MSE per op against sampling at two passes per sector; four models.
  3. A deterministic prefix design: a rank-1 lattice over x_0..x_8 (multiplier n x 0.618) and x_9 on 200 equal-mass nodes,
     costed at two passes per prefix, scored on the whole suite. A lattice is equidistribution, the edge of the rules.
  4. Inside gauss_margin: families with max(p, q, r) = 8, so x_9 is a lower input. P(wrong) per node from the 16-run
     design, from an order-statistic design over the six prefix lower inputs (8 or 16 runs) with x_9 integrated on 64
     equal-mass nodes, and from 512 draws of all seven; each on levels (12, 12), (7, 8), (5, 6). If the suffix integral
     smoothed the integrand, its coarse-minus-fine grid error would shrink. Three models, 56 families.

Finding: research/angles/exact-suffix.md
"""
import numpy as np
import torch as th
from scipy.stats import norm

from az.bench import suite, truth
from az.estimators.gauss_margin import FAMS, FREE, H8, Q, grid, lower_inputs, truncated
from az.model import SAMPLE_OPS, logits, trace
from experiments.common import DEV

th.set_grad_enabled(False)
labels, ws = suite(0.002, 32)
p_true = truth(0.002, 32)
gen = th.Generator(device=DEV).manual_seed(7)


def correct(w, x):
    return (logits([w], x[None])[0].argmax(1) == x.argsort(1)[:, -2]).double()


print("1. share of the indicator's variance left after integrating the last k inputs exactly")
for i in (0, 32, 64, 96):
    out = []
    for k, n in ((1, 200), (2, 40)):
        z = th.tensor(norm.ppf((np.arange(n) + 0.5) / n), device=DEV)
        P = th.randn(4096, 10 - k, generator=gen, device=DEV, dtype=th.float64)
        S = th.cartesian_prod(*([z] * k)).reshape(-1, k)
        x = th.cat([P[:, None].expand(-1, len(S), -1), S[None].expand(len(P), -1, -1)], 2).reshape(-1, 10)
        out.append(correct(ws[i], x).reshape(len(P), len(S)).mean(1).var().item() / (p_true[i] * (1 - p_true[i])))
    print(f"  {labels[i]:10s} last input {out[0]:.3f}, last two {out[1]:.3f}")

print("\n2. great circles: sectors per circle, sd of the correct fraction, MSE per op against sampling at 2 passes per sector")
ang = th.linspace(0, 2 * np.pi, 20001, device=DEV, dtype=th.float64)[:-1]
for i in (0, 32, 64, 96):
    fr, sec = [], []
    for _ in range(120):
        U = th.randn(2, 10, generator=gen, device=DEV, dtype=th.float64)
        x = th.cos(ang)[:, None] * U[0] + th.sin(ang)[:, None] * U[1]
        L, pre, _ = trace(ws[i], x)
        pat, order = (pre > 0).reshape(len(ang), -1), x.argsort(1)
        sec.append(((pat != pat.roll(1, 0)).any(1) | (order != order.roll(1, 0)).any(1)).sum().item())
        fr.append((L.argmax(1) == order[:, -2]).double().mean().item())
    fr, sec = np.array(fr), np.array(sec)
    print(f"  {labels[i]:10s} sectors {sec.mean():.0f}, sd {fr.std():.3f}, MSE ratio {fr.var() * 2 * sec.mean() / (p_true[i] * (1 - p_true[i])):.1f}")

print("\n3. lattice prefixes x exact last input, whole suite")
z = th.tensor(norm.ppf((np.arange(200) + 0.5) / 200), device=DEV)
for n in (127, 509, 2039):
    a = int(round(n * 0.6180339887))
    U = ((np.arange(n)[:, None] * np.array([pow(a, j, n) for j in range(9)])) % n + 0.5) / n
    x = th.cat([th.tensor(norm.ppf(U), device=DEV)[:, None].expand(-1, 200, -1), z[None, :, None].expand(n, -1, 1)], 2).reshape(-1, 10)
    e = np.array([correct(w, x).mean().item() for w in ws])
    ops = 2 * n * SAMPLE_OPS
    rmse, rs = np.sqrt(np.mean((e - p_true) ** 2)), np.sqrt(np.mean(p_true * (1 - p_true)) / (2 * n))
    print(f"  {n:5d} prefixes: ops {ops:.1e}, rmse {rmse:.4f}, bias {np.mean(e - p_true):+.4f}, sampling {rs:.4f}, mse/samp {(rmse / rs) ** 2:.2f}")

print("\n4. gauss_margin cells, families with x_9 a lower input: error mass at (12, 12) and coarse minus fine")
fams = [k for k, f in enumerate(FAMS) if max(f) == 8][::3]
pos, free, qt = (th.tensor(np.asarray(a)[fams], device=DEV) for a in (FAMS, FREE, Q))
F, NS = len(fams), 64


def os_design(t, n, R):
    """(cells, R, n): the order-statistic design for n i.i.d. N(0,1) truncated above at t."""
    v = th.special.ndtri(th.special.ndtr(t)[:, None] * th.tensor(((5 * np.arange(R)) % R + 0.5) / R, device=DEV) ** (1 / n))
    mu, sd = truncated(v)
    top = np.arange(R) % n
    rest = th.tensor(np.array([[j for j in range(n) if j != s] for s in top]), device=DEV)
    low = th.zeros(len(t), R, n, dtype=th.float64, device=DEV)
    low.scatter_(2, rest.expand(len(t), -1, -1), mu[..., None] + sd[..., None] * th.tensor(np.vstack([H8, -H8])[:R, : n - 1], device=DEV))
    return low.scatter_(2, th.tensor(top, device=DEV)[None, :, None].expand(len(t), -1, 1), v[..., None])


def marg(w, nd, low):
    C, R = len(nd), low.shape[2]
    x = th.zeros(C, F, R, 10, dtype=th.float64, device=DEV)
    x.scatter_(3, free[None, :, None].expand(C, -1, R, -1), low.expand(C, F, R, 7).contiguous())
    top = th.stack([nd[:, 0] + nd[:, 2] + nd[:, 1], nd[:, 0] + nd[:, 2], nd[:, 0]], 1)
    x.scatter_(3, pos[None, :, None].expand(C, -1, R, -1), top[:, None, None].expand(-1, F, R, -1).contiguous())
    L = logits([w], x.reshape(1, -1, 10))[0].reshape(C, F, R, 10)
    qq = qt[None, :, None, None].expand(C, -1, R, 1)
    return L.gather(3, qq)[..., 0] - L.scatter(3, qq, -1e9).max(3).values


def integrands(w, nd):
    t = nd[:, 0]
    m = marg(w, nd, lower_inputs(t)[:, None])
    out = {"16-run design": th.special.ndtr(-m.mean(2) / m.std(2, correction=0).clamp(min=1e-9))}
    us = th.special.ndtri(th.special.ndtr(t)[:, None] * th.tensor((np.arange(NS) + 0.5) / NS, device=DEV))
    for R in (8, 16):
        low = th.cat([os_design(t, 6, R)[:, :, None].expand(-1, -1, NS, -1), us[:, None, :, None].expand(-1, R, -1, 1)], 3)
        out[f"suffix exact, {R} runs"] = (marg(w, nd, low.reshape(len(t), 1, R * NS, 7)).reshape(len(t), F, R, NS) < 0).double().mean((2, 3))
    u = th.rand(len(t), F, 512, 7, generator=gen, device=DEV, dtype=th.float64)
    out["512 draws"] = (marg(w, nd, th.special.ndtri(u * th.special.ndtr(t)[:, None, None, None])) < 0).double().mean(2)
    return out


for i in (0, 32, 128):
    tot = {}
    for lv in ((12, 12), (7, 8), (5, 6)):
        mass, nodes, _ = grid(lv)
        acc = {}
        for c in range(0, len(nodes), 4):
            for k, v in integrands(ws[i], th.tensor(nodes[c : c + 4], device=DEV)).items():
                acc[k] = acc.get(k, 0.0) + float((th.tensor(mass[c : c + 4], device=DEV)[:, None] * v).sum() / F)
        tot[lv] = acc
    fine = tot[(12, 12)]
    print(f"  {labels[i]:10s} " + "; ".join(f"{k}: {fine[k]:.4f}, (7, 8) {tot[(7, 8)][k] - fine[k]:+.4f}, (5, 6) {tot[(5, 6)][k] - fine[k]:+.4f}" for k in fine), flush=True)
