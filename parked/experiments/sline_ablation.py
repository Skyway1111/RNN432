"""Which part of the S-line lattice beats sampling: the lattice, the family enumeration, or the exact S-line?

At matched numbers of forward passes, on the whole suite:
  1. plain lattice: Roberts' 10-dimensional Kronecker sequence mapped to N(0, I) inputs, correct/incorrect averaged;
  2. families enumerated: per family the same strata and lattice as az.estimators.sline, but the correct/incorrect of one
     input per node, with S drawn from its law by the lattice's tenth coordinate, instead of the exact S-integral;
  3. the estimator itself (exact S-lines), at strata whose counted ops match.
Each line of 3 costs about six passes, so 1 and 2 get six inputs' worth per line of 3 (more nodes per family in 2).
Then variant 2 alone at low budgets, 1 x 1 to 5 x 5 strata per family (one input per node, from 720 inputs up).

Finding: research/angles/family-lattice.md
"""
import numpy as np
import torch as th
from scipy.stats import norm

from az.bench import suite, truth
from az.estimators import sline as S
from az.estimators.gauss_margin import FAMS, FREE
from az.model import SAMPLE_OPS, logits
from experiments.common import DEV

labels, ws = suite(0.002, 32)
P = truth(0.002, 32)
phi = 1.0
for _ in range(100):
    phi = (1 + phi) ** (1 / 11)
R10 = (1 / phi) ** np.arange(1, 11) % 1
t = np.linspace(-5, 5, 200001)
cdf_t = np.cumsum(360 * norm.pdf(t) * norm.cdf(t) ** 7 * (1 - norm.cdf(t)) ** 2)
cdf_t /= cdf_t[-1]


def ratio(est, ops):
    err = np.array(est) - P
    return np.mean(err ** 2) / (np.mean(P * (1 - P)) * SAMPLE_OPS / ops), np.sqrt(np.mean(err ** 2))


def family_lattice(k):
    """Inputs (720 k^2, 10): per family, k x k strata of (T, M | T), positions, lower inputs and S from the lattice."""
    m = k * k
    u = (0.5 + np.arange(720 * m)[:, None] * R10) % 1
    T2 = np.interp((np.arange(720 * m) % m // k + u[:, 0]) / k, cdf_t, t)
    M2 = norm.ppf(norm.cdf(T2) + (1 - norm.cdf(T2)) * np.sqrt((np.arange(720 * m) % k + u[:, 1]) / k))
    S2 = norm.ppf(norm.cdf(T2) + (norm.cdf(M2) - norm.cdf(T2)) * u[:, 9])
    z = norm.ppf(norm.cdf(T2)[:, None] * u[:, 2:9])
    x2 = np.zeros((720 * m, 10))
    for f, (p_, q_, r_) in enumerate(FAMS):
        sl = slice(f * m, (f + 1) * m)
        x2[sl][:, FREE[f]] = z[sl]
        x2[sl, p_], x2[sl, q_], x2[sl, r_] = M2[sl], S2[sl], T2[sl]
    return th.tensor(x2, device=DEV)


def indicator_mean(x):
    out = []
    for w in ws:
        c = sum((logits([w], x[i : i + 2**20][None])[0].argmax(1) == x[i : i + 2**20].argsort(1)[:, -2]).sum().item() for i in range(0, len(x), 2**20))
        out.append(c / len(x))
    return out


for kt in (3, 8):
    X, q, T, M = S.nodes(kt, kt)
    est, ops = [], []
    for w in ws:
        W = [th.tensor(a, device=DEV, dtype=th.float64) for a in w]
        p, o = zip(*(S.sline(*W, X[c : c + 2**17], q[c : c + 2**17], T[c : c + 2**17], M[c : c + 2**17]) for c in range(0, len(X), 2**17)))
        est.append(th.cat(p).mean().item())
        ops.append(th.cat(o).sum().item())
    S.nodes.cache_clear()
    ops = np.mean(ops)
    n = int(ops // SAMPLE_OPS)  # inputs the other two may use
    r3, e3 = ratio(est, ops)
    x1 = th.tensor(norm.ppf((0.5 + np.arange(n)[:, None] * R10) % 1), device=DEV)
    r1, e1 = ratio(indicator_mean(x1), n * SAMPLE_OPS)
    k = max(1, int(np.sqrt(n / 720)))  # strata per side for the family-enumerated indicator
    m = k * k
    x2 = family_lattice(k)
    r2, e2 = ratio(indicator_mean(x2), 720 * m * SAMPLE_OPS)
    print(f"{kt} x {kt} ({ops:.1e} ops): plain lattice mse/samp {r1:.2f} (rmse {e1:.4f}); families enumerated {r2:.2f} "
          f"(rmse {e2:.4f}, {k} x {k} strata); exact S-lines {r3:.2f} (rmse {e3:.4f})", flush=True)

print("\nfamilies enumerated with one correct/incorrect input per node, low budgets")
for k in (1, 2, 3, 5):
    r, e = ratio(indicator_mean(family_lattice(k)), 720 * k * k * SAMPLE_OPS)
    print(f"{k} x {k}: ops {720 * k * k * SAMPLE_OPS:.1e}, rmse {e:.4f}, mse/samp {r:.2f}", flush=True)
