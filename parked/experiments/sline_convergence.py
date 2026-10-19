"""Does the S-line lattice converge faster than sampling, and does the exact S-line integral earn its cost at high budgets?

The suite's truth (2^24 inputs per model) has a sampling error of about 8.5e-5, as large as sline's error near 1e10 ops.
So: twelve suite models (copies 0 and 1 of each base) with a truth from 2^30 inputs each (sd about 1e-5; cached in
data/cache/truth30.npy, rebuilt by this script when missing), and az.estimators.sline at 4 x 4 to 64 x 64 strata.
Reported: RMSE against ops, sampling's RMSE at those ops, and the log-log slope (sampling's is -0.5); then the same for
the same strata and lattice with one correct/incorrect input per node in place of the exact S-line.

Finding: research/angles/family-lattice.md
"""
import numpy as np
import torch as th
from scipy.stats import norm

from az.bench import suite
from az.estimators import sline as S
from az.estimators.gauss_margin import FAMS, FREE
from az.model import DATA, SAMPLE_OPS, accuracy, logits
from experiments.common import DEV

labels, ws = suite(0.002, 32)
pick = [b * 32 + j for b in range(6) for j in range(2)]
path = DATA / "cache" / "truth30.npy"
if not path.exists():
    np.save(path, np.stack([np.array(pick, float), [accuracy([ws[i]], 2**30, seed=1234 + i)[0] for i in pick]]))
P = np.load(path)[1]

rows = []
for k in (4, 8, 16, 32, 48, 64):
    X, q, T, M = S.nodes(k, k)
    est = []
    for i in pick:
        W = [th.tensor(a, device=DEV, dtype=th.float64) for a in ws[i]]
        p, ops = zip(*(S.sline(*W, X[c : c + 2**17], q[c : c + 2**17], T[c : c + 2**17], M[c : c + 2**17]) for c in range(0, len(X), 2**17)))
        est.append((th.cat(p).mean().item(), th.cat(ops).sum().item()))
    S.nodes.cache_clear()
    est = np.array(est)
    err, ops = est[:, 0] - P, est[:, 1].mean()
    rmse, rs = np.sqrt(np.mean(err ** 2)), np.sqrt(np.mean(P * (1 - P)) * SAMPLE_OPS / ops)
    rows.append((ops, rmse))
    print(f"{k} x {k}: ops {ops:.2e}, rmse {rmse:.2e}, bias {err.mean():+.1e}, sampling {rs:.2e}, mse/samp {(rmse / rs) ** 2:.3f}", flush=True)
r = np.array(rows)
print(f"log-log slope of rmse against ops {np.polyfit(np.log(r[:, 0]), np.log(r[:, 1]), 1)[0]:.2f} (sampling -0.50); "
      f"the truth's sd about {np.sqrt(0.12 / 2**30):.1e}")

print("\nthe same strata and lattice with one correct/incorrect input per node instead of the exact S-line (S from the "
      "lattice's tenth coordinate), k x k strata per family")
phi = 1.0
for _ in range(100):
    phi = (1 + phi) ** (1 / 11)
R10 = (1 / phi) ** np.arange(1, 11) % 1
grid = np.linspace(-5, 5, 200001)
cdf_t = np.cumsum(360 * norm.pdf(grid) * norm.cdf(grid) ** 7 * (1 - norm.cdf(grid)) ** 2)
cdf_t /= cdf_t[-1]
rows = []
for k in (7, 19, 40, 80, 120):
    m = k * k
    u = (0.5 + np.arange(720 * m)[:, None] * R10) % 1
    T = np.interp((np.arange(720 * m) % m // k + u[:, 0]) / k, cdf_t, grid)
    M = norm.ppf(norm.cdf(T) + (1 - norm.cdf(T)) * np.sqrt((np.arange(720 * m) % k + u[:, 1]) / k))
    Sv = norm.ppf(norm.cdf(T) + (norm.cdf(M) - norm.cdf(T)) * u[:, 9])
    z = norm.ppf(norm.cdf(T)[:, None] * u[:, 2:9])
    x = np.zeros((720 * m, 10))
    for f, (p_, q_, r_) in enumerate(FAMS):
        sl = slice(f * m, (f + 1) * m)
        x[sl][:, FREE[f]] = z[sl]
        x[sl, p_], x[sl, q_], x[sl, r_] = M[sl], Sv[sl], T[sl]
    x = th.tensor(x, device=DEV)
    lab = x.argsort(1)[:, -2]
    est = [sum((logits([ws[i]], x[j : j + 2**20][None])[0].argmax(1) == lab[j : j + 2**20]).sum().item() for j in range(0, len(x), 2**20)) / len(x) for i in pick]
    err, ops = np.array(est) - P, len(x) * SAMPLE_OPS
    rmse, rs = np.sqrt(np.mean(err ** 2)), np.sqrt(np.mean(P * (1 - P)) * SAMPLE_OPS / ops)
    rows.append((ops, rmse))
    print(f"{k} x {k}: ops {ops:.2e}, rmse {rmse:.2e}, sampling {rs:.2e}, mse/samp {(rmse / rs) ** 2:.3f}", flush=True)
r = np.array(rows)
print(f"log-log slope {np.polyfit(np.log(r[:, 0]), np.log(r[:, 1]), 1)[0]:.2f}")
