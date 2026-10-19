"""Exact accuracy over all 10! orderings of fixed magnitudes: which errors come from order structure, which from magnitudes?

Gaussian accuracy = E over sorted magnitudes y of acc(y), where acc(y) averages over all orderings of y, uniformly
(i.i.d. inputs). acc(y) is computed exactly as a fraction by enumerating 3,628,800 orderings; the smallest gap between
the top two logits certifies that float rounding cannot flip a decision.
Finding: research/findings/orderings.md
"""
import numpy as np

from az.model import NAMES, load
from az.zoo import SEEDS, SEQS, checkpoint
from experiments.common import N_ORD as N, exact_acc, order_stat_means, orderings

POS = orderings()[1]  # POS[j, rank] = position
LABEL = POS[:, 1]


def exact(ws, y):
    """Per model: (accuracy as Fraction, smallest top-2 logit gap, prediction per ordering)."""
    return [exact_acc(w, y) for w in ws]


mu = order_stat_means()
print("E[sorted N(0,1)^10]:", np.round(mu, 4))
profiles = {"mean": mu, "mean + 1": mu + 1, "mean - 1": mu - 1, "S-T gap 0.05": np.r_[mu[:2], mu[1] - 0.05, mu[3:]]}
models = [load(n) for n in NAMES]
for pname, y in profiles.items():
    res = exact(models, y)
    print(f"profile {pname}: " + " | ".join(f"{n} {float(a):.4f}" for n, (a, _, _) in zip(NAMES, res))
          + f"   (smallest top-2 gap {min(g for _, g, _ in res):.2e})")

# Example model at the mean profile: what the order-level errors look like.
acc, gap, pred = exact(models[:1], mu)[0]
wrong = pred != LABEL
p, q, r = POS[:, 0], POS[:, 1], POS[:, 2]
print(f"\nexample, mean profile: exactly {acc} = {float(acc):.6f}; wrong on {wrong.sum().item()} of {N} orderings")
def rate(m):
    return f"{wrong[m].double().mean().item():.4f}"


print(f"  error rate when max comes before 2nd: {rate(p < q)}, after: {rate(p > q)}")
print("  error rate by 2nd-max position 0..9: " + " ".join(rate(q == i) for i in range(10)))
print("  error rate by max position 0..9:     " + " ".join(rate(p == i) for i in range(10)))
print(f"  wrong predictions: 3rd-largest {(pred[wrong] == r[wrong]).double().mean().item():.2f}, largest {(pred[wrong] == p[wrong]).double().mean().item():.2f}")

# How much of Gaussian accuracy is order structure vs magnitudes: acc(y) over random magnitude sets.
rng = np.random.default_rng(0)
ys = np.array([np.sort(rng.standard_normal(10))[::-1] for _ in range(400)])
accs = np.array([[float(a) for a, _, _ in exact(models[:2], y.copy())] for y in ys])
near = ys[:, 1] - ys[:, 2] < 0.1
few = (ys > 0).sum(1) <= 1
print("\nacc(y) over 400 random magnitude sets (its mean estimates Gaussian accuracy):")
for j, name in enumerate(NAMES[:2]):
    a = accs[:, j]
    print(f"  {name}: mean {a.mean():.4f} (SE {a.std() / 20:.4f}) median {np.median(a):.4f}"
          f" share below 0.9 {(a < 0.9).mean():.3f} | mean when S-T < 0.1 ({near.mean():.2f} of sets) {a[near].mean():.3f},"
          f" otherwise {a[~near].mean():.3f}; when <= 1 positive ({few.mean():.3f}) {a[few].mean() if few.any() else float('nan'):.3f}")

# Does phase 3 fix order-level errors? Exact accuracy at the mean profile across training.
print("\nexact accuracy at the mean profile across training (columns: 0, 2^20 ... 2^30 sequences)")
for seed in SEEDS:
    print(f"  seed {seed}: " + " ".join(f"{float(a):.3f}" for a, _, _ in exact([checkpoint(seed, n) for n in SEQS], mu)))
