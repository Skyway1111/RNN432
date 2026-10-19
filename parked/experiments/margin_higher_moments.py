"""Is the Gaussian presumption's floor the margin's shape, and do its skewness and kurtosis remove it?

At gauss_margin's (7, 8) nodes, every family, the seven lower inputs i.i.d. N(0,1) truncated above at T. P(wrong) from
the empirical share of 2048 draws against laws fitted to moments: the Gaussian (two moments), and the Edgeworth expansion
of the distribution function with the skewness, then also the excess kurtosis. First with the draws' own moments (the
oracle: what the law alone costs), then with moments from gauss_margin's 16-run design and from a 64-run order-statistic
design (the same construction with 64 quantiles of the largest lower input). Six models, copy 0 of each base; accuracy
error = estimate - empirical, mass-weighted.

Finding: research/angles/margin-gaussian.md
"""
import numpy as np
import torch as th

from az.bench import suite
from az.estimators.gauss_margin import H8, margins, quadrature, truncated
from experiments.common import DEV

th.set_grad_enabled(False)
labels, ws = suite(0.002, 32)
mass, nodes = quadrature(7, 8)
gen = th.Generator(device=DEV).manual_seed(9)


def p_wrong(m, order):
    """P(margin < 0) from the moments of m (..., runs): Gaussian (order 2) or Edgeworth with skewness (3) and kurtosis (4)."""
    mu, sd = m.mean(-1), m.std(-1, correction=0).clamp(min=1e-9)
    z = (m - mu[..., None]) / sd[..., None]
    sk, ku = (z ** 3).mean(-1), (z ** 4).mean(-1) - 3
    u = -mu / sd
    c = 0 * u if order == 2 else sk / 6 * (u ** 2 - 1)
    if order == 4:
        c = c + ku / 24 * (u ** 3 - 3 * u) + sk ** 2 / 72 * (u ** 5 - 10 * u ** 3 + 15 * u)
    return (th.special.ndtr(u) - th.exp(-u ** 2 / 2) / np.sqrt(2 * np.pi) * c).clamp(0, 1)


def os64(t):
    """64-run order-statistic design: the largest lower input at quantile ((5 j mod 64) + 1/2) / 64 of its law in slot
    j mod 7, the other six at mean +- sd below it with signs from H8 and its mirror, cycled."""
    q = th.tensor(((5 * np.arange(64)) % 64 + 0.5) / 64, device=DEV)
    v = th.special.ndtri(th.special.ndtr(t)[:, None] * q ** (1 / 7))
    mu, sd = truncated(v)
    top = np.arange(64) % 7
    rest = th.tensor(np.array([[i for i in range(7) if i != s] for s in top]), device=DEV)
    signs = th.tensor(np.tile(np.vstack([H8, -H8])[:, :6], (4, 1)), device=DEV)
    low = th.zeros(len(t), 64, 7, dtype=th.float64, device=DEV)
    low.scatter_(2, rest.expand(len(t), -1, -1), mu[..., None] + sd[..., None] * signs)
    return low.scatter_(2, th.tensor(top, device=DEV)[None, :, None].expand(len(t), -1, 1), v[..., None])


names = [f"{src} {law}" for src in ("oracle", "16-run", "64-run") for law in ("Gaussian", "Edgeworth 3", "Edgeworth 4")]
res = []
for i in [k * 32 for k in range(6)]:
    acc = np.zeros(len(names) + 1)
    for c in range(len(nodes)):
        nd = nodes[c : c + 1]
        t = th.tensor(nd[:, 0], device=DEV)
        u = th.rand(1, 720, 2048, 7, generator=gen, device=DEV, dtype=th.float64)
        me = margins(ws[i], nd, th.special.ndtri(u * th.special.ndtr(t)[:, None, None, None]))[0]
        runs = [me, margins(ws[i], nd)[0], margins(ws[i], nd, os64(t))[0]]
        acc += mass[c] * np.array([(me < 0).double().mean().item()] + [p_wrong(m, o).mean().item() for m in runs for o in (2, 3, 4)])
    res.append(acc[0] - acc[1:])
    print(f"{labels[i]:10s} " + "  ".join(f"{n} {e:+.4f}" for n, e in zip(names, res[-1])), flush=True)
res = np.array(res)
print("\nestimate - empirical accuracy over 6 models: mean, rms")
for n, col in zip(names, res.T):
    print(f"  {n:22s} {col.mean():+.5f} {np.sqrt((col ** 2).mean()):.5f}")
