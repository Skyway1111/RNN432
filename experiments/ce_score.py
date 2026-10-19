"""Does understanding pay on the rare-event yardstick, or would any adaptive importance sampler do as well?

az/estimators/ce.py is modes with its gap proposals swapped for full-covariance Gaussians fitted to the same events: the
cross-entropy method's family, reading no structure. Fixed before running: each design below is scored on the held-out
suite over seed salts 0 to 2, paired; the design with the best geometric-mean advantage over c = 2 to 8 is then scored over
salts 0 to 4 on the held-out suite and once on the bench suite, on the ladder 3e6 to 1e10, the protocol of modes_score.
Each (suite, design, salt) is cached in data/cache/ce_score/; a run computes missing ones for up to 15 minutes, so rerun
until all tables are complete.

Finding: research/angles/mode-mixture.md
"""
import json
import time

import numpy as np

from az.estimators import ce
from az.model import DATA
from az.rare import LEVELS, run

B = [3e6, 1e7, 3e7, 1e8, 3e8, 1e9, 3e9, 1e10]
CACHE = DATA / "cache" / "ce_score"
CACHE.mkdir(parents=True, exist_ok=True)
DESIGNS = {  # name: (components per level at c = 2, 4, 6, 8; first-stage share)
    "K8-adapt": ((8, 4, 4, 4), 0.3),
    "K8": ((8, 4, 4, 4), 0.0),
    "K4-adapt": ((4, 2, 2, 2), 0.3),
    "K16-adapt": ((16, 8, 8, 8), 0.3),
    "K32-adapt": ((32, 16, 16, 16), 0.3),
}
start = time.time()


def score(suite, design, salt):
    path = CACHE / f"{suite}_{design}_{salt}.json"
    if not path.exists():
        if time.time() - start > 900:
            return None
        ks, ce.ADAPT = DESIGNS[design]
        ce.K, ce.SALT = dict(zip((2.0, 4.0, 6.0, 8.0), ks)), salt
        ce.mixture.cache_clear()
        path.write_text(json.dumps(run("ce", suite, B)))
    return json.loads(path.read_text())["advantage"]


def table(suite, design, salts):
    got = [score(suite, design, s) for s in salts]
    if None in got:
        return None
    a = np.array(got)
    return np.exp(np.log(a).mean(0)), a.min(0), a.max(0)


def fmt(t):
    return " ".join(f"{v:7.2f}" for v in t[0]) + "; range " + " ".join(f"{lo:.2f}-{hi:.2f}" for lo, hi in zip(t[1], t[2]))


print("advantage at c = " + ", ".join(f"{c:g}" for c in LEVELS) + ": geometric mean over salts; range")
print("design sweep, held-out suite, salts 0-2")
sweep = {d: table("heldout", d, (0, 1, 2)) for d in DESIGNS}
for d, t in sweep.items():
    print(f"  {d:10s} " + (fmt(t) if t is not None else "incomplete; rerun"))
if all(t is not None for t in sweep.values()):
    best = max(sweep, key=lambda d: np.log(sweep[d][0][1:]).mean())
    print(f"chosen: {best}")
    for suite in ("heldout", "bench"):
        t = table(suite, best, range(5))
        print(f"  {suite:8s} salts 0-4 " + (fmt(t) if t is not None else "incomplete; rerun"))
        if t is not None:
            (DATA.parent / "results" / "bench" / f"ce_rare_{suite}.json").write_text((CACHE / f"{suite}_{best}_0.json").read_text())
    print("modes over the chosen design, paired by salt 0-4 (modes from data/cache/modes_score/, experiments.modes_score)")
    for suite in ("heldout", "bench"):
        mpath = [DATA / "cache" / "modes_score" / f"{suite}_{s}.json" for s in range(5)]
        if not all(p.exists() for p in mpath) or table(suite, best, range(5)) is None:
            continue
        r = np.array([json.loads(p.read_text())["advantage"] for p in mpath]) / np.array([score(suite, best, s) for s in range(5)])
        slope = [np.polyfit(LEVELS[1:], np.log(x[1:]), 1)[0] for x in r]
        print(f"  {suite:8s} " + fmt((np.exp(np.log(r).mean(0)), r.min(0), r.max(0))) + "; slope of log ratio per unit c, c = 2-8, per salt "
              + " ".join(f"{v:+.3f}" for v in slope))
