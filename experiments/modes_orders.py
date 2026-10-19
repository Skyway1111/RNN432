"""Does giving the modes estimator each copy's own orders pay, by learning them from its first stage or predicting them
from the weights? And how much does one realization's score vary?

Each row is the harness on the held-out suite (budgets 3e6 to 1e10) for the settled design (az/estimators/modes.py with
ADAPT = 0) and three variants, over seed salts 1-5 drawn with the same seeds per salt, so the comparison is paired:
learned, ADAPT = 0.3, the first stage's soft events re-estimating every component's orders; predicted, each level-4, 6, 8
component's pi times exp(0.5 (f_copy - f_base)) at its mean point placed at all 720 orders, run when the budget covers
ten times the 720 passes per component; both. Each (variant, salt) result is cached in data/cache/modes_orders/; a run
computes missing ones for up to 15 minutes, so rerun until the table is complete.

Finding: research/angles/mode-mixture.md
"""
import json
import math
import time

import numpy as np
import torch as th

from az.estimators import modes
from az.model import DATA
from az.rare import LEVELS, TAIL_OPS, run

B = [3e6, 1e7, 3e7, 1e8, 3e8, 1e9, 3e9, 1e10]
CACHE = DATA / "cache" / "modes_orders"
CACHE.mkdir(parents=True, exist_ok=True)
SALTS = (1, 2, 3, 4, 5)
mixture0, tail0 = modes.mixture, modes.tail
M = math.sqrt(2 / math.pi)
KEYS = th.arange(1000, device=modes.DEV)
KEYS = KEYS[(KEYS // 100 != KEYS // 10 % 10) & (KEYS // 100 != KEYS % 10) & (KEYS // 10 % 10 != KEYS % 10)]
ORD = th.stack([KEYS // 100, KEYS // 10 % 10, KEYS % 10], 1)


def at_orders(comps, k):
    _, mu, _, tm, tt, tl = comps.params(k)
    T = mu - tt * M
    x = (T - th.as_tensor(tl, device=modes.DEV) * M).expand(720, 10).clone()
    i = th.arange(720, device=modes.DEV)
    x[i, ORD[:, 0]], x[i, ORD[:, 1]], x[i, ORD[:, 2]] = mu + tm * M, mu, T
    return x / x.norm(dim=1, keepdim=True)


def tail_predicted(w, budget):
    b = modes.base_of(w)
    comps, alphas, levels = mixture0(b)
    W = [th.tensor(a, dtype=th.float32, device=modes.DEV) for a in w]
    W0 = [th.tensor(a, dtype=th.float32, device=modes.DEV) for a in modes._bases()[b]]
    pi, fixed = comps.pi.clone(), 0
    for c in (4.0, 6.0, 8.0):
        idx = (levels == c).nonzero()[:, 0].tolist()
        cost = 720 * TAIL_OPS * len(idx)
        if budget < 10 * (fixed + cost):
            break
        for j in idx:
            d = 0.5 * (modes.margin(W, at_orders(comps, j)) - modes.margin(W0, at_orders(comps, j)))
            pi[j, KEYS] *= th.exp(d - d.max())
        fixed += cost
    comps = modes.osp.OSP(pi, comps.mu, comps.sig, comps.tm, comps.tt, comps.tl)
    modes.mixture = lambda _: (comps, alphas, levels)
    try:
        e, ops = tail0(w, budget - fixed)
    finally:
        modes.mixture = mixture0
    return e, ops + fixed


VARIANTS = {"settled": (0.0, tail0), "learned": (0.3, tail0), "predicted": (0.0, tail_predicted), "both": (0.3, tail_predicted)}
start = time.time()
for name, (adapt, tail) in VARIANTS.items():
    for salt in SALTS:
        path = CACHE / f"{name}_{salt}.json"
        if path.exists() or time.time() - start > 900:
            continue
        modes.ADAPT, modes.SALT, modes.tail = adapt, salt, tail
        path.write_text(json.dumps(run("modes", "heldout", B)["advantage"]))
modes.tail = tail0
print("advantage at c = " + ", ".join(f"{c:g}" for c in LEVELS) + ", per salt and geometric mean over salts " + ", ".join(map(str, SALTS)))
for name in VARIANTS:
    got = [json.loads((CACHE / f"{name}_{s}.json").read_text()) for s in SALTS if (CACHE / f"{name}_{s}.json").exists()]
    if len(got) < len(SALTS):
        print(f"  {name}: {len(got)} of {len(SALTS)} salts done; rerun")
        continue
    a = np.array(got)
    for s, row in zip(SALTS, a):
        print(f"  {name:9s} salt {s}: " + " ".join(f"{v:7.2f}" for v in row))
    print(f"  {name:9s} geometric mean " + " ".join(f"{v:7.2f}" for v in np.exp(np.log(a).mean(0))) + "; range "
          + " ".join(f"{lo:.2f}-{hi:.2f}" for lo, hi in zip(a.min(0), a.max(0))))
