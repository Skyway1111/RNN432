"""Why do central designs over the lower inputs miss the margin's spread, and does placing their maximum by its law fix it?

At gauss_margin's (7, 8) nodes, every family, the seven lower inputs i.i.d. N(0,1) truncated above at T. Three measurements:
  1. How much of the margin's variance over the lower inputs is linear in them, additive (a piecewise-linear function of each
     input, knots at its quintiles), and seen by the 8-run Hadamard design (1/n), as shares of the empirical variance over
     512 draws, mass-weighted over (node, family) pairs with 0.02 < P(wrong) < 0.98. Two models, every third node.
  2. The largest lower input near T: truncate the lower inputs at T - delta for both the design and the draws. If H8's
     error and low sd come from that input nearing T, they shrink as delta grows. Two models.
  3. The estimator's 16-run order-statistic design (gauss_margin.lower_inputs: the maximum of the seven at quantiles of its
     law (Phi(v)/Phi(T))^7, the other six at mean +- sd below it, 1/n) against H8 (1/(n-1), the earlier estimator) and the
     oracle (Phi with the draws' moments), as estimate - empirical accuracy over 1024 draws. Six models, copy 0 of each base.

Finding: research/angles/margin-gaussian.md
"""
import numpy as np
import torch as th
from scipy.stats import norm

from az.bench import suite
from az.estimators.gauss_margin import H8, margins, quadrature, truncated
from experiments.common import DEV

th.set_grad_enabled(False)
labels, ws = suite(0.002, 32)
mass, nodes = quadrature(7, 8)
gen = th.Generator(device=DEV).manual_seed(9)


def draws(t, cut, n):
    """(1, 720, n, 7) lower inputs i.i.d. N(0,1) truncated above at cut."""
    u = th.rand(1, 720, n, 7, generator=gen, device=DEV, dtype=th.float64)
    return th.special.ndtri(u * norm.cdf(cut))


def h8(cut):
    mu, sd = (a.item() for a in truncated(th.tensor(cut, dtype=th.float64)))
    return th.tensor(mu + sd * H8, device=DEV)[None]


def wrong(m, pop):
    return th.special.ndtr(-m.mean(-1) / m.std(-1, correction=0 if pop else 1).clamp(min=1e-9))


print("1. shares of the margin's variance over the lower inputs (mass-weighted over pairs with 0.02 < P(wrong) < 0.98)")
for i in (32, 0):
    S = []
    for c in range(0, len(nodes), 3):
        nd = nodes[c : c + 1]
        t = nd[0, 0]
        z = draws(t, t, 512)
        me = margins(ws[i], nd, z)[0]
        pe = (me < 0).double().mean(1)
        sel = (pe > 0.02) & (pe < 0.98)
        if not sel.any():
            continue
        Z, M = z[0, sel], me[sel]
        knots = th.tensor(norm.ppf(np.array([0.2, 0.4, 0.6, 0.8]) * norm.cdf(t)), device=DEV)
        lin = th.cat([th.ones_like(Z[..., :1]), Z], -1)
        add = th.cat([lin] + [th.relu(Z - k) for k in knots], -1)
        fit = lambda X: (X @ th.linalg.lstsq(X, M[..., None]).solution)[..., 0].var(1)  # noqa: E731
        vd = margins(ws[i], nd, h8(t))[0][sel].var(1, correction=0)
        S.append(th.stack([M.var(1), fit(lin), fit(add), vd, th.full_like(vd, mass[c])], 1))
    S = th.cat(S)
    share = lambda k: ((S[:, k] / S[:, 0]) * S[:, 4]).sum().item() / S[:, 4].sum().item()  # noqa: E731
    print(f"  {labels[i]:10s} linear {share(1):.3f}  additive {share(2):.3f}  H8 design {share(3):.3f}")

print("\n2. lower inputs truncated at T - delta for design and draws: estimate - empirical accuracy; median sd ratio H8/empirical")
for i in (32, 0):
    for delta in (0.0, 0.15, 0.4):
        acc, rs = np.zeros(4), []
        for c in range(len(nodes)):
            nd = nodes[c : c + 1]
            t = nd[0, 0]
            me = margins(ws[i], nd, draws(t, t - delta, 1024))[0]
            md = margins(ws[i], nd, h8(t - delta))[0]
            pe = (me < 0).double().mean(1)
            sel = (pe > 0.02) & (pe < 0.98)
            rs.append((md.std(1, correction=0) / me.std(1))[sel])
            acc += mass[c] * np.array([pe.mean().item(), wrong(md, True).mean().item(), wrong(md, False).mean().item(),
                                       wrong(me, False).mean().item()])
        print(f"  {labels[i]:10s} delta {delta:.2f}: H8 1/n {acc[0] - acc[1]:+.4f}, H8 1/(n-1) {acc[0] - acc[2]:+.4f}, "
              f"oracle {acc[0] - acc[3]:+.4f}; sd ratio {th.cat(rs).median().item():.3f}")

print("\n3. estimate - empirical accuracy at the (7, 8) nodes: H8 1/(n-1) | order-statistic 16 runs | oracle")
res = []
for i in [k * 32 for k in range(6)]:
    acc = np.zeros(4)
    for c in range(len(nodes)):
        nd = nodes[c : c + 1]
        t = nd[0, 0]
        me = margins(ws[i], nd, draws(t, t, 1024))[0]
        acc += mass[c] * np.array([(me < 0).double().mean().item(), wrong(margins(ws[i], nd, h8(t))[0], False).mean().item(),
                                   wrong(margins(ws[i], nd)[0], True).mean().item(), wrong(me, False).mean().item()])
    res.append(acc[0] - acc[1:])
    print(f"  {labels[i]:10s} " + " | ".join(f"{e:+.4f}" for e in res[-1]), flush=True)
res = np.array(res)
print(f"  {'mean':10s} " + " | ".join(f"{e:+.4f}" for e in res.mean(0)))
print(f"  {'rms':10s} " + " | ".join(f"{e:.4f}" for e in np.sqrt((res ** 2).mean(0))))
