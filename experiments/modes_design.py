"""Which parts of the modes estimator carry its advantage? Ablations scored by the harness on the held-out suite.

Each row is `az.rare.run("modes", "heldout", ...)` on the budget ladder 3e6 to 1e10 with one change from the settled
design without the order stage (az/estimators/modes.py with ADAPT = 0: K = 8, 4, 4, 4 components at levels 2, 4, 6, 8,
shares 0.6, 0.2, 0.1, 0.1, BETA = 0.2):
the T gap tied to the lower inputs' pooled scale (the proposal without its T-end wedge); one lower-gap scale for all
positions; independent draws by component, multinomial counts and pseudo-random normals instead of fixed counts and a
shifted lattice; K = 8 at every level; BETA = 0.1 and 0.3.

Finding: research/angles/mode-mixture.md
"""
import math
import zlib

import numpy as np
import torch as th

from az import osp
from az.estimators import modes
from az.rare import LEVELS, run

B = [3e6, 1e7, 3e7, 1e8, 3e8, 1e9, 3e9, 1e10]
fit0, tail0 = osp.fit, modes.tail


def fit_tied_t(x, w, smooth=0.5):
    c = fit0(x, w, smooth)
    pi, mu, sig, tm, _, tl = c.params()
    return osp.OSP(pi, mu, sig, tm, float(np.sqrt(np.mean(tl**2))), tl)


def fit_scalar_lower(x, w, smooth=0.5):
    c = fit0(x, w, smooth)
    pi, mu, sig, tm, tt, tl = c.params()
    return osp.OSP(pi, mu, sig, tm, tt, float(np.sqrt(np.mean(tl**2))))


def tail_random(w, budget):
    comps, alphas, _ = modes.mixture(modes.base_of(w))
    per = modes.TAIL_OPS + 20 + modes.GAP_OPS + modes.COMP_OPS * len(comps)
    n = max(int(budget // per), 1)
    W = [th.tensor(a, dtype=th.float32, device=modes.DEV) for a in w]
    gen = th.Generator(device=modes.DEV).manual_seed(zlib.crc32(np.ascontiguousarray(w.Wh).tobytes()))
    probs = th.cat([th.tensor([modes.BETA], dtype=th.float64, device=modes.DEV), alphas])
    lv = th.tensor(LEVELS, dtype=th.float64, device=modes.DEV)
    s = th.zeros(len(LEVELS), dtype=th.float64, device=modes.DEV)
    for start in range(0, n, 2**20):
        m = min(2**20, n - start)
        k = th.multinomial(probs, m, replacement=True, generator=gen)
        x = th.randn(m, 10, generator=gen, device=modes.DEV, dtype=th.float64)
        for j in range(len(comps)):
            sel = k == j + 1
            if sel.any():
                x[sel] = comps.sample(int(sel.sum()), gen, j)
        u = x / x.norm(dim=1, keepdim=True)
        s += ((modes.margin(W, u)[:, None] >= lv) * osp.mixture_weight(comps, alphas, modes.BETA, u)[:, None]).sum(0)
    return (s / n).cpu().numpy(), n * per + 6 * 432


L = (2.0, 4.0, 6.0, 8.0)
ROWS = [("settled", {}), ("T gap at the lowers' scale", {"fit": fit_tied_t}), ("one lower-gap scale", {"fit": fit_scalar_lower}),
        ("random draws", {"tail": tail_random}), ("K = 8 at every level", {"K": dict(zip(L, (8, 8, 8, 8)))}),
        ("BETA = 0.1", {"BETA": 0.1}), ("BETA = 0.3", {"BETA": 0.3})]
modes.ADAPT = 0.0
K0, BETA0 = dict(modes.K), modes.BETA
print("advantage at c = " + ", ".join(f"{c:g}" for c in LEVELS))
for name, ch in ROWS:
    osp.fit, modes.tail = ch.get("fit", fit0), ch.get("tail", tail0)
    modes.K, modes.BETA = ch.get("K", K0), ch.get("BETA", BETA0)
    modes.library.cache_clear()
    r = run("modes", "heldout", B)
    print(f"  {name:28s} " + " ".join(f"{a:7.2f}" if math.isfinite(a) else "    inf" for a in r["advantage"]), flush=True)
