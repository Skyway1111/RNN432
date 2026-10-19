"""How far does the modes estimator beat sampling, scored over seed realizations rather than one?

One realization's advantage is a draw: on the held-out suite, seeds alone move the c = 8 advantage between 17 and 114
(research/angles/mode-mixture.md). Fixed before the bench suite is run: the score of az/estimators/modes.py as
committed is the harness's advantage per level on the budget ladder 3e6 to 1e10, as the geometric mean over seed salts
0 to 4, with the range, on the held-out suite and then the bench suite. Salt 0's per-budget rows are written to
results/bench/modes_rare_SUITE.json. Each (suite, salt) is cached in data/cache/modes_score/; a run computes missing
ones for up to 15 minutes, so rerun until both tables are complete. `python -m experiments.modes_score heldout` runs
only the held-out suite.

Finding: research/angles/mode-mixture.md
"""
import json
import sys
import time

import numpy as np

from az.estimators import modes
from az.model import DATA
from az.rare import LEVELS, run

B = [3e6, 1e7, 3e7, 1e8, 3e8, 1e9, 3e9, 1e10]
CACHE = DATA / "cache" / "modes_score"
CACHE.mkdir(parents=True, exist_ok=True)
SALTS = (0, 1, 2, 3, 4)
start = time.time()
for suite in sys.argv[1:] or ("heldout", "bench"):
    for salt in SALTS:
        path = CACHE / f"{suite}_{salt}.json"
        if path.exists() or time.time() - start > 900:
            continue
        modes.SALT = salt
        r = run("modes", suite, B)
        path.write_text(json.dumps(r))
        if salt == 0:
            (DATA.parent / "results" / "bench" / f"modes_rare_{suite}.json").write_text(json.dumps(r, indent=1))
print("advantage at c = " + ", ".join(f"{c:g}" for c in LEVELS) + ", per salt and geometric mean over salts " + ", ".join(map(str, SALTS)))
for suite in ("heldout", "bench"):
    got = [json.loads((CACHE / f"{suite}_{s}.json").read_text())["advantage"] for s in SALTS if (CACHE / f"{suite}_{s}.json").exists()]
    if len(got) < len(SALTS):
        print(f"  {suite}: {len(got)} of {len(SALTS)} salts done; rerun")
        continue
    a = np.array(got)
    for s, row in zip(SALTS, a):
        print(f"  {suite:8s} salt {s}: " + " ".join(f"{v:7.2f}" for v in row))
    print(f"  {suite:8s} geometric mean " + " ".join(f"{v:7.2f}" for v in np.exp(np.log(a).mean(0))) + "; range "
          + " ".join(f"{lo:.2f}-{hi:.2f}" for lo, hi in zip(a.min(0), a.max(0))))
