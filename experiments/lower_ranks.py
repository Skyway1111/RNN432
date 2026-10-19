"""Which lower inputs does the tie decision depend on nonlinearly, and can a presumption add their exact effects?

At the tie S = T, for configurations (family, T, M) from the exact laws and draws of the seven lower inputs each:
  1. A deduced ladder: the k largest lower inputs exact (k = 0..5), the rest entering linearly through the route at
     their conditional mean below the smallest exact one. Also a time split: the lower inputs before the first of M, S,
     T exact, the later ones linear. Per prediction, the share of draws whose sign of the tie margin m_T it gets right,
     and its P(m_T < 0) against the draws' (bias, per-configuration rms).
  2. The additive presumption: m_T = m(z*) + sum_i [m(z* with z_i actual) - m(z*)], each lower input's exact 1-D effect
     at an anchor z* (all seven at their conditional mean; or at the expected order statistics in slot order), so the
     law of m_T would be a convolution of seven exact 1-D laws. Its P(m_T < 0) bias with standard error and rms, and
     its bias in E[relu(-m_T)], the first-order wedge's numerator; the second-order (pairwise) version for comparison.
  3. Where the interaction lives: the pairwise terms' variance by the time distance between the two inputs' slots.
  4. Whether the interaction follows from the two 1-D effects: among pairs whose effects both lower the margin, the
     share with a positive pairwise term and its mean against the effects' sum; the R^2 of the pairwise term on
     [g_i g_j, g_i^2 + g_j^2, |g_i| + |g_j|].

Finding: research/findings/errors.md, parked/research/angles/additive-presumption.md
"""
import itertools

import numpy as np
import torch as th

from az.bench import suite
from az.estimators.gauss_margin import FAMS, FREE
from az.estimators.lattice import _cdf_t
from experiments.common import DEV

labels, ws = suite(0.002, 32)
FAMS_T, FREE_T = th.tensor(FAMS, device=DEV), th.tensor(FREE, device=DEV)
PAIRS = list(itertools.combinations(range(7), 2))
OSQ = th.tensor(np.arange(7, 0, -1) / 8.0, device=DEV)
t_grid, cdf_t = _cdf_t()


def place(fam, M, T, z):
    x = th.zeros(len(fam), 10, device=DEV, dtype=th.float64)
    x.scatter_(1, FREE_T[fam], z)
    return x.scatter_(1, FAMS_T[fam], th.stack([M, T, T], 1))


def jac(W, x):
    """Logits and their Jacobian in x (the exact linear map of x's route)."""
    Wi, Wh, Wo = W
    h = th.zeros(len(x), 16, device=DEV, dtype=th.float64)
    J = th.zeros(len(x), 16, 10, device=DEV, dtype=th.float64)
    for t in range(10):
        z = h @ Wh.T + x[:, t, None] * Wi
        on = (z > 0).double()
        J = th.einsum("ij,njk->nik", Wh, J)
        J[:, :, t] += Wi
        J, h = J * on[:, :, None], z * on
    return h @ Wo.T, th.einsum("oi,nik->nok", Wo, J)


def margin(L, q):
    return L.gather(1, q[:, None])[:, 0] - L.scatter(1, q[:, None], -1e9).max(1).values


def configs(rng, nc, nd):
    fam = th.tensor(rng.integers(0, 720, nc), device=DEV)
    T = th.tensor(np.interp(rng.random(nc), cdf_t, t_grid), device=DEV)
    PT = th.special.ndtr(T)
    M = th.special.ndtri(PT + (1 - PT) * th.tensor(np.sqrt(rng.random(nc)), device=DEV))
    z = th.special.ndtri(PT[:, None, None] * th.tensor(rng.random((nc, nd, 7)), device=DEV)).reshape(-1, 7)
    return fam, T, M, z


def chunks(fn, n, *a):
    return th.cat([fn(*(b[c:c + 2**18] for b in a)) for c in range(0, n, 2**18)])


def mean_below(v):
    return -th.exp(-v ** 2 / 2) / np.sqrt(2 * np.pi) / th.special.ndtr(v)


def stats(pr, m, nc):
    pr, m = pr.reshape(nc, -1), m.reshape(nc, -1)
    d = (pr < 0).double().mean(1) - (m < 0).double().mean(1)
    return ((pr < 0) == (m < 0)).double().mean().item(), d.mean().item(), d.std().item() / np.sqrt(nc), d.pow(2).mean().sqrt().item()


rng = np.random.default_rng(0)
for i in (32, 0, 160, 64, 96, 128):
    W = [th.tensor(a, device=DEV, dtype=th.float64) for a in ws[i]]
    nc, nd = 2048, 256
    fam, T, M, z = configs(rng, nc, nd)
    famr, Tr, Mr = (a.repeat_interleave(nd) for a in (fam, T, M))
    q = FAMS_T[famr][:, 1]
    x = place(famr, Mr, Tr, z)
    m = chunks(lambda xx, qq: margin(jac(W, xx)[0], qq), len(x), x, q)
    print(f"{labels[i]}: P(m_T < 0) {(m < 0).double().mean():.4f}, E[relu(-m_T)] {th.relu(-m).mean():.3f}")

    print("  1. exact set, the rest linear through the route at their conditional mean below it")
    order = z.argsort(1, descending=True)
    early = FREE_T[famr] < FAMS_T[famr].min(1, keepdim=True).values
    sets = {f"{k} largest": th.zeros_like(early).scatter_(1, order[:, :k], True) for k in range(6)}
    sets["those before the first of M, S, T"] = early
    for name, ex in sets.items():
        floor = th.where(ex, z, th.full_like(z, np.inf)).min(1).values.clamp(max=Tr)
        zr = th.where(ex, z, mean_below(floor)[:, None])
        pr = chunks(lambda xr, xx, qq: margin(th.einsum("nij,nj->ni", jac(W, xr)[1], xx), qq), len(x), place(famr, Mr, Tr, zr), x, q)
        a, b, se, r = stats(pr, m, nc)
        print(f"    {name:36s} sign agreement {a:.3f}, P(m_T<0) bias {b:+.4f} (se {se:.4f}), per-config rms {r:.3f}")

    print("  2. additive presumption: each lower input's exact 1-D effect at the anchor, summed")
    mt = lambda zz: chunks(lambda ff, mm, tt, zq, qq: margin(jac(W, place(ff, mm, tt, zq))[0], qq), len(zz), famr, Mr, Tr, zz, q)  # noqa: E731
    anchors = {"conditional mean": mean_below(T)[:, None].expand(-1, 7),
               "order statistics": th.special.ndtri(th.special.ndtr(T)[:, None] * OSQ)}
    for name, za in anchors.items():
        zs = za.repeat_interleave(nd, 0).contiguous()
        m0 = mt(zs)
        one = []
        for k in range(7):
            zk = zs.clone()
            zk[:, k] = z[:, k]
            one.append(mt(zk) - m0)
        h1 = m0 + sum(one)
        a, b, se, r = stats(h1, m, nc)
        rel = (th.relu(-h1).mean() / th.relu(-m).mean() - 1).item()
        print(f"    anchor {name:17s} first order:  sign agreement {a:.3f}, P bias {b:+.4f} (se {se:.4f}), rms {r:.3f}; "
              f"E[relu(-m_T)] {rel:+.3f} relative")
        if name == "order statistics" and i in (32, 0, 160):
            sub = th.arange(len(x), device=DEV).reshape(nc, nd)[:512].reshape(-1)  # second order on a quarter
            h2, by_dist, tot, trip = h1[sub].clone(), np.zeros(10), 0.0, []
            slot = FREE_T[famr[sub]]
            for u, v in PAIRS:
                zuv = zs[sub].clone()
                zuv[:, [u, v]] = z[sub][:, [u, v]]
                g = chunks(lambda ff, mm, tt, zq, qq: margin(jac(W, place(ff, mm, tt, zq))[0], qq), len(sub),
                           famr[sub], Mr[sub], Tr[sub], zuv, q[sub]) - m0[sub] - one[u][sub] - one[v][sub]
                h2 += g
                tot += g.pow(2).sum().item()
                by_dist += th.bincount((slot[:, v] - slot[:, u]).abs(), weights=g.pow(2), minlength=10).cpu().numpy()
                trip.append(th.stack([g, one[u][sub], one[v][sub]], 1))
            gab, ga, gb = th.cat(trip).T
            neg = (ga < -0.1) & (gb < -0.1)
            X = th.stack([ga * gb, ga ** 2 + gb ** 2, ga.abs() + gb.abs(), th.ones_like(ga)], 1)
            fit = X @ th.linalg.lstsq(X, gab[:, None]).solution
            r2 = 1 - ((gab[:, None] - fit) ** 2).sum() / ((gab - gab.mean()) ** 2).sum()
            a, b, se, r = stats(h2, m[sub], 512)
            rel = (th.relu(-h2).mean() / th.relu(-m[sub]).mean() - 1).item()
            print(f"    anchor {name:17s} second order: sign agreement {a:.3f}, P bias {b:+.4f} (se {se:.4f}), rms {r:.3f}; "
                  f"E[relu(-m_T)] {rel:+.3f} relative")
            print("  3. share of the pairwise terms' variance by the time distance between the two slots, 1..9: "
                  + " ".join(f"{v / tot:.2f}" for v in by_dist[1:]))
            print(f"  4. pairs whose effects both lower m_T by > 0.1: {neg.double().mean():.3f} of pairs; their pairwise term "
                  f"> 0 in {(gab[neg] > 0).double().mean():.2f}, mean {gab[neg].mean():+.3f} against the effects' sum "
                  f"{(ga + gb)[neg].mean():+.3f}; R^2 of the pairwise term on the two effects {r2:.3f}")
    print(flush=True)
