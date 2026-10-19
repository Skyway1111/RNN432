"""Does scale-freeness give the last input's sweep a second exact dimension for free?

Correctness depends only on the input's direction, so with the prefix x_0..x_8 = r w the pair (r, x_9) integrates in
closed form: s = x_9 / r has density proportional to (1 + s^2)^-5, i.e. t_9 / 3. The plain sweep integrates x_9 at the
prefix's own r (az/estimators/sweep.py); the radial one integrates (r, x_9) at its direction w, with the same kinks and
the same cost. On 2^20 Gaussian inputs, four suite models: the variance of the indicator, the sweep and the radial sweep,
overall and within the 720 families.

Finding: research/angles/family-lattice.md
"""
import torch as th
from scipy.special import stdtr

from az.bench import suite
from az.estimators.gauss_margin import FAMS
from az.estimators.sweep import interval, prefix, sweep
from experiments.common import DEV

labels, ws = suite(0.002, 32)
FAMS_T = th.tensor(FAMS, device=DEV)
KEY = th.full((1000,), -1, dtype=th.long, device=DEV)
KEY[FAMS_T[:, 0] * 100 + FAMS_T[:, 1] * 10 + FAMS_T[:, 2]] = th.arange(720, device=DEV)


def radial(W, X, fam, q):
    """P(correct) over (r, x_9) at each prefix's direction: the sweep's intervals in s = x_9 / r under t_9 / 3."""
    Xu = X / X[:, :9].norm(dim=1, keepdim=True)
    lo, hi = interval(Xu, fam, 9, list(range(9)))
    h = prefix(W, Xu, 9)
    Wi, Wh, Wo = W
    A = h @ Wh.T
    rt = -A / Wi
    ok = (rt > lo[:, None]) & (rt < hi[:, None])
    bounds = th.cat([lo[:, None], th.where(ok, rt, hi[:, None]).sort(1).values, hi[:, None]], 1)
    plo, phi = bounds[:, :-1], bounds[:, 1:]
    on = ((A[:, None] + Wi * ((plo + phi) / 2)[..., None]) > 0).double()
    Lc, Ld = (on * A[:, None]) @ Wo.T, (on * Wi) @ Wo.T
    qq = q[:, None, None].expand(-1, 17, 1)
    a, b = Lc.gather(2, qq) - Lc, Ld.gather(2, qq) - Ld
    other = th.ones_like(a, dtype=th.bool).scatter_(2, qq, False)
    root = -a / b
    lo_c = th.maximum(plo, th.where(other & (b > 0), root, -th.inf).max(2).values)
    hi_c = th.minimum(phi, th.where(other & (b < 0), root, th.inf).min(2).values)
    good = ~(other & (b == 0) & (a <= 0)).any(2) & (phi > plo)
    F = lambda v: th.tensor(stdtr(9, 3 * v.cpu().numpy()), device=DEV)  # noqa: E731
    return ((F(hi_c) - F(lo_c)).clamp(min=0) * good).sum(1) / (F(hi) - F(lo))


g = th.Generator(device=DEV).manual_seed(5)
for i in (32, 0, 160, 64):
    W = [th.tensor(a, device=DEV, dtype=th.float64) for a in ws[i]]
    x = th.randn(2**20, 10, generator=g, device=DEV, dtype=th.float64)
    o = x.argsort(1, descending=True)[:, :3]
    fam = KEY[o[:, 0] * 100 + o[:, 1] * 10 + o[:, 2]]
    q = o[:, 1]
    out = {"indicator": [], "sweep": [], "radial sweep": []}
    for c in range(0, len(x), 2**17):
        X, f, qq = x[c:c + 2**17], fam[c:c + 2**17], q[c:c + 2**17]
        out["indicator"].append(((prefix(W, X, 10) @ W[2].T).argmax(1) == qq).double())
        lo, hi = interval(X, f, 9, list(range(9)))
        out["sweep"].append(sweep(W, prefix(W, X, 9), qq, lo, hi)[0])
        out["radial sweep"].append(radial(W, X, f, qq))
    row = []
    for k, v in out.items():
        v = th.cat(v)
        fm = th.zeros(720, device=DEV, dtype=th.float64).index_add_(0, fam, v) / th.bincount(fam, minlength=720)
        row.append(f"{k}: mean {v.mean():.4f}, variance {v.var():.4f}, within families {(v - fm[fam]).pow(2).mean():.4f}")
    print(f"{labels[i]}  " + " | ".join(row), flush=True)
