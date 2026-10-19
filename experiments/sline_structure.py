"""Why do exact S-lines work, what do they cost, and which part of az/estimators/sline.py needs its quasi-random sequence?

  1. The error set along an S-line (x_q from T to M, everything else fixed; Gaussian inputs as anchors, 400 grid points per
     line, three models): its share at the T end, the M end and the interior; correctness switches per line; and the
     share of the indicator's variance left after integrating S exactly.
  2. The exact piecewise-linear integral (sline.sline) against 2,000 equal-mass nodes on 4,000 lines (random family,
     T, M and lower inputs), and its multiply-adds per line in passes.
  3. Component isolation on the whole suite, 3 x 3 and 6 x 6 strata per family: stratum positions from the Kronecker
     sequence or at conditional medians, lower inputs from the Kronecker sequence or from gauss_margin's 16-run design
     (run index rotating with family and stratum).

Finding: parked/research/angles/family-lattice.md
"""
import numpy as np
import torch as th
from scipy.stats import norm

from az.bench import suite, truth
from az.estimators.gauss_margin import FAMS, FREE, Q, lower_inputs
from az.estimators.sline import sline
from az.model import SAMPLE_OPS, logits
from experiments.common import DEV

th.set_grad_enabled(False)
ALPHA = np.array([np.sqrt(p) % 1 for p in (2, 3, 5, 7, 11, 13, 17, 19, 23)])  # the first lattice: a per-family Kronecker sequence
labels, ws = suite(0.002, 32)
P = truth(0.002, 32)


def weights(w):
    return [th.tensor(a, device=DEV, dtype=th.float64) for a in w]


print("1. the error set along S-lines")
for i in (32, 0, 160):
    g = th.Generator(device=DEV).manual_seed(3)
    x = th.randn(4096, 10, generator=g, device=DEV, dtype=th.float64)
    top = x.topk(3, 1)
    M, S, T = top.values.T
    q = top.indices[:, 1]
    s = T[:, None] + (M - T)[:, None] * (th.arange(400, device=DEV, dtype=th.float64) + 0.5) / 400
    dens = th.exp(-s ** 2 / 2)
    dens /= dens.sum(1, keepdim=True)
    X = x[:, None].expand(-1, 400, -1).clone()
    X.scatter_(2, q[:, None, None].expand(-1, 400, 1), s[..., None])
    wrong = (logits([ws[i]], X.reshape(1, -1, 10))[0].argmax(1).reshape(-1, 400) != q[:, None]).double()
    first_ok = (wrong == 0).double().argmax(1)
    last_ok = 399 - (wrong.flip(1) == 0).double().argmax(1)
    allwrong = wrong.min(1).values == 1
    idx = th.arange(400, device=DEV)[None]
    t_end = th.where(allwrong[:, None], wrong, (idx < first_ok[:, None]).double() * wrong)
    m_end = th.where(allwrong[:, None], 0 * wrong, (idx > last_ok[:, None]).double() * wrong)
    err = (dens * wrong).sum(1)
    tot, pp = err.mean().item(), 1 - P[i]
    sw = (wrong[:, 1:] != wrong[:, :-1]).sum(1)
    print(f"  {labels[i]:10s} error share at the T end {(dens * t_end).sum(1).mean().item() / tot:.2f}, M end "
          f"{(dens * m_end).sum(1).mean().item() / tot:.2f}; lines with 0, 1, 2, >2 switches "
          + ", ".join(f"{(sel).double().mean():.2f}" for sel in (sw == 0, sw == 1, sw == 2, sw > 2))
          + f"; variance left {err.var().item() / (pp * (1 - pp)):.2f}")

print("\n2. exact integral against 2,000 nodes, and its cost")
rng = np.random.default_rng(1)
for i in (32, 0, 160):
    fam = rng.integers(720, size=4000)
    T = norm.ppf(rng.uniform(0.02, 0.98, 4000)) * 0.45 + 0.65
    M = norm.ppf(norm.cdf(T) + (1 - norm.cdf(T)) * np.sqrt(rng.uniform(0, 1, 4000)))
    X = np.zeros((4000, 10))
    for k in range(4000):
        p_, _, r_ = FAMS[fam[k]]
        X[k, FREE[fam[k]]] = norm.ppf(norm.cdf(T[k]) * rng.uniform(0, 1, 7))
        X[k, r_], X[k, p_] = T[k], M[k]
    Xt, qt, Tt, Mt = (th.tensor(a, device=DEV) for a in (X, Q[fam], T, M))
    pc, ops = sline(*weights(ws[i]), Xt, qt, Tt, Mt)
    us = (th.arange(2000, device=DEV, dtype=th.float64) + 0.5) / 2000
    Sg = th.special.ndtri(th.special.ndtr(Tt)[:, None] + (th.special.ndtr(Mt) - th.special.ndtr(Tt))[:, None] * us)
    ref = []
    for c in range(0, 4000, 200):
        xx = Xt[c : c + 200, None].expand(-1, 2000, -1).clone()
        xx[th.arange(200, device=DEV), :, qt[c : c + 200]] = Sg[c : c + 200]
        ref.append((logits([ws[i]], xx.reshape(1, -1, 10))[0].argmax(1).reshape(200, 2000) == qt[c : c + 200, None]).double().mean(1))
    d = pc - th.cat(ref)
    print(f"  {labels[i]:10s} max |exact - nodes| {d.abs().max():.4f}, mean {d.mean():+.5f}; passes per line mean "
          f"{ops.mean() / SAMPLE_OPS:.2f}, 90th percentile {ops.quantile(0.9) / SAMPLE_OPS:.1f}")

print("\n3. which part needs the Kronecker sequence (whole suite)")
t = np.linspace(-5, 5, 200001)
cdf_t = np.cumsum(360 * norm.pdf(t) * norm.cdf(t) ** 7 * (1 - norm.cdf(t)) ** 2)
cdf_t /= cdf_t[-1]
for kt in (3, 6):
    n = kt * kt
    f, s = np.arange(720)[:, None, None], np.arange(n)[None, :, None]
    for tm, zs in (("Kronecker", "Kronecker"), ("medians", "Kronecker"), ("Kronecker", "16-run design"), ("medians", "16-run design")):
        u = (f * ALPHA + s * 0.3819660113 + np.arange(9) * 0.123) % 1.0
        if tm == "medians":
            u[..., :2] = 0.5
        T = np.interp((np.arange(n) // kt + u[..., 0]) / kt, cdf_t, t)
        M = norm.ppf(norm.cdf(T) + (1 - norm.cdf(T)) * np.sqrt((np.arange(n) % kt + u[..., 1]) / kt))
        if zs == "Kronecker":
            z = norm.ppf(norm.cdf(T)[..., None] * u[..., 2:])
        else:
            runs = lower_inputs(th.tensor(T.reshape(-1), device=DEV)).cpu().numpy().reshape(720, n, 16, 7)
            z = np.take_along_axis(runs, ((f[..., 0] + 5 * s[..., 0]) % 16)[..., None, None].repeat(7, 3), 2)[:, :, 0]
        X = np.zeros((720, n, 10))
        for k, (p_, _, r_) in enumerate(FAMS):
            X[k, :, FREE[k]] = z[k].T
            X[k, :, r_], X[k, :, p_] = T[k], M[k]
        args = [th.tensor(a, device=DEV) for a in (X.reshape(-1, 10), np.repeat(Q, n), T.reshape(-1), M.reshape(-1))]
        e = np.array([[v.mean().item() if j == 0 else v.sum().item() for j, v in enumerate(sline(*weights(w), *args))] for w in ws])
        err, ops = e[:, 0] - P, e[:, 1].mean()
        mse_s = np.mean(P * (1 - P)) / (ops / SAMPLE_OPS)
        print(f"  {kt} x {kt}, positions {tm:9s}, lower inputs {zs:13s}: ops {ops:.2e}, rmse {np.sqrt(np.mean(err ** 2)):.4f}, "
              f"bias {err.mean():+.4f}, mse/samp {np.mean(err ** 2) / mse_s:.2f}", flush=True)
