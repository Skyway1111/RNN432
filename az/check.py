"""Gate: doc structure, then estimator math against brute force. python -m az.check [--fix]"""
import sys

import numpy as np
from scipy.stats import multivariate_normal

from az import docs, gauss


def close(name, got, want, tol):
    err = float(np.max(np.abs(np.asarray(got) - np.asarray(want))))
    print(f"{'ok ' if err <= tol else 'FAIL'} {name}: max err {err:.2e} (tol {tol:.0e})")
    return err <= tol


def check_osp():
    """Gap proposals: K9 against its closed forms; a nominal-plus-concentrated mixture against exact probabilities; the
    closed-form angular density against quadrature of the ray integral."""
    import math

    import torch as th

    from az import osp

    s = th.tensor([0.0, 80.0, -80.0], dtype=th.float64, device=osp.DEV)
    want = [math.log(2**4 * 24), 3200 + 0.5 * math.log(2 * math.pi) + 9 * math.log(80) + math.log1p(36 / 6400), math.log(math.factorial(9)) - 10 * math.log(80)]
    ok = close("logk9 vs closed forms", osp.logk9(s).cpu().numpy(), want, 1e-3)
    pi = osp.valid_keys().double() * (th.arange(osp.KEYS, device=osp.DEV) % 100 // 10 == 9)
    comp = osp.OSP(pi, -0.5, 0.1, 0.5, 0.05, th.linspace(0.1, 0.6, 10))  # label at 9, near the all-negative vertex
    gen = th.Generator(device=osp.DEV).manual_seed(0)
    n = 2**20
    x = th.cat([th.randn(n, 10, generator=gen, device=osp.DEV, dtype=th.float64), comp.sample(n, gen)])
    u = x / x.norm(dim=1, keepdim=True)
    lr = osp.mixture_weight(comp, th.tensor([0.5], dtype=th.float64, device=osp.DEV), 0.5, u)
    got = [lr.mean().item(), (lr * (u.argsort(1)[:, -2] == 9)).mean().item(), (lr * ((u > 0).sum(1) <= 1)).mean().item()]
    ok &= close("osp mixture: E[LR], P(label 9), P(<= 1 positive)", got, [1.0, 0.1, 11 / 1024], 3e-3)
    # the closed-form angular density of a three-component stack against the ray integral of its density in x
    rng = th.Generator(device=osp.DEV).manual_seed(1)
    stack = osp.OSP(th.rand(3, osp.KEYS, generator=rng, device=osp.DEV, dtype=th.float64) * osp.valid_keys(), [0.6, -0.4, 0.0],
                    [0.3, 0.1, 0.5], [0.8, 0.05, 1.5], [0.02, 0.4, 0.2], th.rand(3, 10, generator=rng, device=osp.DEV, dtype=th.float64) + 0.1)
    u = th.cat([stack.sample(16, gen, k) for k in range(3)])
    u = u / u.norm(dim=1, keepdim=True)
    r = th.linspace(1e-3, 40, 4000, dtype=th.float64, device=osp.DEV)
    lx = th.stack([stack.log_x(osp.Gaps(ri * u)) for ri in r]) + 9 * th.log(r)[:, None, None]
    quad = th.logsumexp(lx, 0) + math.log(r[1].item() - r[0].item())
    return ok & close("osp log angular density vs ray quadrature", stack.log_ang(osp.Gaps(u)).cpu().numpy(), quad.cpu().numpy(), 1e-3)


def check_ce():
    """Gaussian components of the ce control: closed-form angular density against the ray integral, and E[LR] = 1 and
    P(label 9) = 0.1 on a nominal-plus-Gaussian mixture."""
    import math

    import torch as th

    from az import osp
    from az.estimators.ce import Gauss, outer

    gen = th.Generator(device=osp.DEV).manual_seed(2)
    A = th.randn(3, 10, 10, generator=gen, device=osp.DEV, dtype=th.float64) * 0.3
    g = Gauss(th.randn(3, 10, generator=gen, device=osp.DEV, dtype=th.float64), A @ A.mT + 0.1 * th.eye(10, device=osp.DEV, dtype=th.float64))
    u = th.randn(48, 10, generator=gen, device=osp.DEV, dtype=th.float64)
    u = u / u.norm(dim=1, keepdim=True)
    r = th.linspace(1e-3, 40, 4000, dtype=th.float64, device=osp.DEV)
    lx = th.stack([g.log_x(ri * u, outer(ri * u)) for ri in r]) + 9 * th.log(r)[:, None, None]
    quad = th.logsumexp(lx, 0) + math.log(r[1].item() - r[0].item())
    ok = close("ce log angular density vs ray quadrature", g.log_ang(u).cpu().numpy(), quad.cpu().numpy(), 1e-3)
    n = 2**20
    part = th.repeat_interleave(th.arange(4, device=osp.DEV), th.tensor([n, n // 2, n // 4, n // 4], device=osp.DEV))
    x = g.place(part, th.randn(len(part), 10, generator=gen, device=osp.DEV, dtype=th.float64))
    u = x / x.norm(dim=1, keepdim=True)
    lq = th.log(th.tensor([0.5, 0.25, 0.125, 0.125], dtype=th.float64, device=osp.DEV))
    lr = th.exp(-math.log(osp.AREA) - th.logsumexp(lq + th.cat([th.full((len(u), 1), -math.log(osp.AREA), dtype=th.float64, device=osp.DEV), g.log_ang(u)], 1), 1))
    got = [lr.mean().item(), (lr * (u.argsort(1)[:, -2] == 9)).mean().item()]
    return ok & close("ce mixture: E[LR], P(label 9)", got, [1.0, 0.1], 3e-3)


def main():
    errs = docs.lint(fix="--fix" in sys.argv)
    for e in errs:
        print("FAIL", e)
    print(f"{'ok ' if not errs else 'FAIL'} docs: {len(errs)} problems")
    rng = np.random.default_rng(0)
    ok = not errs
    h, k = rng.normal(size=200) * 1.5, rng.normal(size=200) * 1.5
    r = np.concatenate([rng.uniform(-0.999, 0.999, 190), [0.95, -0.95, 0.99, -0.99, 0.9999, -0.9999, 0.5, 0, 0.93, -0.93]])
    want = [multivariate_normal(cov=[[1, ri], [ri, 1]]).cdf([-hi, -ki]) for hi, ki, ri in zip(h, k, r)]
    ok &= close("bvnu vs scipy", gauss.bvnu(h, k, r), want, 1e-6)

    d, n = 5, 2_000_000
    A = rng.normal(size=(d, d))
    S, mu = A @ A.T, rng.normal(size=d)
    z = rng.multivariate_normal(mu, S, size=n)
    m, C, _ = gauss.relu_moments(mu, S)
    rz = np.maximum(z, 0)
    ok &= close("relu mean vs MC", m, rz.mean(0), 0.01 * np.sqrt(S.diagonal().max()))
    ok &= close("relu cov vs MC", C, np.cov(rz.T), 0.02 * S.diagonal().max())

    a = rng.normal(size=d)
    m0, mu2, S2 = gauss.truncate(1.0, mu, S, a)
    sel = z @ a > 0
    ok &= close("truncate mass", m0, sel.mean(), 2e-3)
    ok &= close("truncate mean", mu2, z[sel].mean(0), 0.01 * np.sqrt(S.diagonal().max()))
    ok &= close("truncate cov", S2, np.cov(z[sel].T), 0.02 * S.diagonal().max())
    ok &= check_osp()
    ok &= check_ce()
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
