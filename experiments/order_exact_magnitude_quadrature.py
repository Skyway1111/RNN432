"""Is acc(y), exact accuracy over all 10! orderings of magnitudes y, a smooth function of the tie gaps S−T and M−S?

The order-exact, magnitude-quadrature angle needs acc(y) to live on a few gaps. (1) acc on a grid of (S−T, M−S) with
the other magnitudes at the mean set, and its spread when those 7 are redrawn; (2) at near-ties, which orderings fail;
(3) over random magnitude sets, the held-out share of acc(y)'s variance that the gaps and sign count explain.
Finding: research/findings/orderings.md
"""
import numpy as np
from scipy.stats import norm

from az.model import NAMES, load, noisy
from experiments.common import exact_acc, order_stat_means, orderings

mu = order_stat_means()
models = {n: load(n) for n in NAMES}
GAPS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
top_gaps = []  # smallest top-2 logit gap per evaluation


def acc(w, y):
    a, gap, _ = exact_acc(w, y)
    top_gaps.append(gap)
    return float(a)


def top2(rest, st, ms):
    """Magnitude set with T, y_4.. = rest and the given gaps S−T, M−S above it."""
    return np.r_[rest[0] + st + ms, rest[0] + st, rest]


print("(1) acc(y) on a grid: rows S−T, columns M−S; T and below at the mean set " + str(np.round(mu[2:], 3)))
for name, w in models.items():
    print(f"  {name:8s}" + "".join(f"{b:>7}" for b in GAPS))
    for a in GAPS:
        print(f"  {a:<8}" + "".join(f"{acc(w, top2(mu[2:], a, b)):7.4f}" for b in GAPS))

# Spread over the other 7: given y_3 = T, the rest are 7 i.i.d. N(0,1) truncated below T (common draws for all cells).
rng = np.random.default_rng(1)
lows = np.sort(norm.ppf(rng.uniform(0, norm.cdf(mu[2]), (32, 7))), 1)[:, ::-1]
print("\n    spread when y_4..y_10 are redrawn given T (32 draws, M−S = 0.5): acc at mean set | mean ± sd [min, max]")
for name, w in models.items():
    cells = []
    for a in (0.02, 0.1, 0.5):
        v = np.array([acc(w, top2(np.r_[mu[2], lo], a, 0.5)) for lo in lows])
        cells.append(f"S−T {a}: {acc(w, top2(mu[2:], a, 0.5)):.4f} | {v.mean():.4f} ± {v.std():.4f} [{v.min():.3f}, {v.max():.3f}]")
    print(f"  {name:8s}" + "   ".join(cells))

print("\n(2) failing orderings at near-ties; families are (p, q, r) position triples, 720 of them, 5040 orderings each")
pos = orderings()[1]
p, q, r = pos[:, 0], pos[:, 1], pos[:, 2]
fam = (p * 10 + q) * 10 + r
real = fam.bincount(minlength=1000) > 0
ties = {"S−T 0.05": np.r_[mu[:2], mu[1] - 0.05, mu[3:]], "S−T 0.01": np.r_[mu[:2], mu[1] - 0.01, mu[3:]],
        "M−S 0.05": np.r_[mu[1] + 0.05, mu[1:]]}
for name in ("example", "s0"):
    for tname, y in ties.items():
        a, gap, pred = exact_acc(models[name], y)
        top_gaps.append(gap)
        wrong = pred != q
        err = fam[wrong].bincount(minlength=1000)
        rate = err[real].double() / 5040
        cover = (err.sort(descending=True).values.double().cumsum(0) / wrong.sum()).cpu().numpy()
        top = err.topk(4)
        print(f"  {name} {tname}: acc {float(a):.4f}, wrong {wrong.sum().item()}; predicts 3rd {(pred[wrong] == r[wrong]).double().mean().item():.2f},"
              f" largest {(pred[wrong] == p[wrong]).double().mean().item():.2f}; error rate p before q {wrong[p < q].double().mean().item():.4f},"
              f" after {wrong[p > q].double().mean().item():.4f}; families covering 50/90/99% of errors:"
              f" {(cover < 0.5).sum() + 1}/{(cover < 0.9).sum() + 1}/{(cover < 0.99).sum() + 1}")
        print(f"    errors in families ≥ 95% wrong {(err[real] * (rate >= 0.95)).sum().item() / wrong.sum().item():.2f};"
              f" mixed families (5–95% wrong) {((rate > 0.05) & (rate < 0.95)).double().mean().item():.3f};"
              f" acc if each family took its majority decision {1 - (rate > 0.5).double().mean().item():.4f}")
        print("    top families (p,q,r) error rate: " + ", ".join(f"({k // 100},{k // 10 % 10},{k % 10}) {c / 5040:.2f}"
                                                         for k, c in zip(top.indices.tolist(), top.values.tolist())))

print("\n(3) acc(y) over 2048 random magnitude sets: held-out variance explained (2-fold), features binned as in error_set_dimension")
ys = np.sort(np.random.default_rng(2).standard_normal((2048, 10)), 1)[:, ::-1]
sc = np.abs(ys[:, :3]).mean(1)
edges = [0.02, 0.05, 0.1, 0.2, 0.4, 0.8]
gaps = np.digitize((ys[:, 1] - ys[:, 2]) / sc, edges) * 7 + np.digitize((ys[:, 0] - ys[:, 1]) / sc, edges)
keys = {"gaps": gaps, "gaps+signs": gaps * 4 + np.minimum((ys > 0).sum(1), 3)}
half = np.arange(len(ys)) % 2


def held_out(v, key):
    pred = np.empty_like(v)
    for h in (0, 1):
        tr, te = half != h, half == h
        means = {k: v[tr][key[tr] == k].mean() for k in np.unique(key[tr])}
        pred[te] = [means.get(k, v[tr].mean()) for k in key[te]]
    return 1 - ((v - pred) ** 2).mean() / v.var(), ((v - pred) ** 2).mean() ** 0.5


variants = [(n, models[n]) for n in ("example", "s0")] + [(f"{n} noisy", noisy(models[n], 0.002, 1, seed=3)[0]) for n in ("example", "s0")]
for name, w in variants:
    v = np.array([acc(w, y) for y in ys])
    fits = " | ".join(f"{k} {r2:.3f} (residual sd {sd:.4f})" for k, (r2, sd) in ((k, held_out(v, key)) for k, key in keys.items()))
    print(f"  {name:13s} mean {v.mean():.4f} ± {v.std() / len(v) ** 0.5:.4f}, sd {v.std():.4f} | {fits}")
print(f"\nsmallest top-2 logit gap over every ordering evaluated: {min(top_gaps):.1e} (float64 rounding ~1e-12)")
