"""Does the family-stratified lattice win on models it was not designed on, and were its design choices the best ones there?

A held-out suite: the six bases with 32 noisy copies each from noise seeds 5000 + i (the bench suite uses 1000 + i), truth
from 2^24 inputs per model (cached in data/cache/val_suite_truth.npy, rebuilt when missing). On it:
  1. az.estimators.lattice at budgets from 3e5 to 1e9;
  2. the design alternatives at about 1e8 ops: equal-mass (T, M) strata against stratifying S as well, and against
     strata graded toward low T and small M - T (weights = stratum masses);
  3. below one node per family: the 90 Latin families against the family as an eleventh lattice coordinate.

Finding: research/angles/family-lattice.md
"""
import numpy as np
import torch as th
from scipy.stats import norm

from az.estimators import lattice as L
from az.estimators.gauss_margin import FAMS, FREE, Q
from az.model import DATA, NAMES, SAMPLE_OPS, accuracy, load, logits, noisy
from experiments.common import DEV

VAL = [w for i, name in enumerate(NAMES) for w in noisy(load(name), 0.002, 32, seed=5000 + i)]
path = DATA / "cache" / "val_suite_truth.npy"
if not path.exists():
    np.save(path, np.concatenate([accuracy(VAL[i : i + 16], 2**24, seed=77 + i) for i in range(0, len(VAL), 16)]))
PV = np.load(path)


def score(label, fn):
    e = np.array([fn(w) for w in VAL])
    err, ops = e[:, 0] - PV, e[:, 1].mean()
    print(f"  {label:44s} ops {ops:.1e}, rmse {np.sqrt(np.mean(err ** 2)):.4f}, bias {err.mean():+.4f}, "
          f"mse/samp {np.mean(err ** 2) / (np.mean(PV * (1 - PV)) * SAMPLE_OPS / ops):.2f}", flush=True)


def weighted(X, q, W):
    X, q, W = (th.tensor(a, device=DEV) for a in (X, q, W))
    return lambda w: (sum(((logits([w], X[c : c + 2**20][None])[0].argmax(1) == q[c : c + 2**20]).double() * W[c : c + 2**20]).sum().item()
                          for c in range(0, len(X), 2**20)), len(X) * SAMPLE_OPS)


def strata_design(te, me, se):
    """Per family: strata of T (edges in its CDF), M | T and S | T, M; one lattice node each, weighted by the stratum mass."""
    te, me, se = map(np.asarray, (te, me, se))
    km, ks = len(me) - 1, len(se) - 1
    n = (len(te) - 1) * km * ks
    u = (0.5 + np.arange(720 * n).reshape(-1, n, 1) * L.roberts(10)) % 1.0
    it, im, i_s = np.arange(n) // (km * ks), np.arange(n) // ks % km, np.arange(n) % ks
    t, cdf = L._cdf_t()
    T = np.interp(te[it] + (te[it + 1] - te[it]) * u[..., 0], cdf, t)
    M = norm.ppf(norm.cdf(T) + (1 - norm.cdf(T)) * np.sqrt(me[im] + (me[im + 1] - me[im]) * u[..., 1]))
    S = norm.ppf(norm.cdf(T) + (norm.cdf(M) - norm.cdf(T)) * (se[i_s] + (se[i_s + 1] - se[i_s]) * u[..., 9]))
    z = norm.ppf(norm.cdf(T)[..., None] * u[..., 2:9])
    W = np.tile((te[it + 1] - te[it]) * (me[im + 1] - me[im]) * (se[i_s + 1] - se[i_s]), 720) / 720
    return weighted(L.place(T, M, S, z).reshape(-1, 10), np.repeat(Q, n), W)


print("1. the estimator on the held-out suite")
for b in (3e5, 1e6, 2.2e6, 1e7, 1e8, 1e9):
    score(f"lattice, budget {b:.0e}", lambda w: L.estimate(w, b))

print("\n2. stratification alternatives at about 1e8 ops")
lin = lambda k: np.linspace(0, 1, k + 1)  # noqa: E731
score("equal-mass T 6 x M 6 (as shipped)", strata_design(lin(6), lin(6), lin(1)))
score("T 3 x M 3 x S 4", strata_design(lin(3), lin(3), lin(4)))
score("T 4 x M 4 x S graded to the ends 3", strata_design(lin(4), lin(4), [0, 0.15, 0.85, 1]))
score("graded T 6 x M 5", strata_design([0, 0.03, 0.1, 0.25, 0.5, 0.75, 1], [0, 0.08, 0.2, 0.4, 0.65, 1], lin(1)))

print("\n3. below one node per family")
for N in (90, 360):
    u = (0.5 + np.arange(N)[:, None] * L.roberts(11)) % 1
    fam = np.minimum((u[:, 10] * 720).astype(int), 719)
    t, cdf = L._cdf_t()
    T = np.interp(u[:, 0], cdf, t)
    M = norm.ppf(norm.cdf(T) + (1 - norm.cdf(T)) * np.sqrt(u[:, 1]))
    S = norm.ppf(norm.cdf(T) + (norm.cdf(M) - norm.cdf(T)) * u[:, 9])
    z = norm.ppf(norm.cdf(T)[:, None] * u[:, 2:9])
    X = np.zeros((N, 10))
    for k, f in enumerate(fam):
        p, q, r = FAMS[f]
        X[k, FREE[f]] = z[k]
        X[k, p], X[k, q], X[k, r] = M[k], S[k], T[k]
    score(f"family as a lattice coordinate, {N} nodes", weighted(X, Q[fam], np.full(N, 1 / N)))
    Xl, ql = L.latin_inputs(N // 90)
    score(f"90 Latin families x {N // 90} (as shipped)", weighted(Xl.cpu().numpy(), ql.cpu().numpy(), np.full(len(Xl), 1 / len(Xl))))
