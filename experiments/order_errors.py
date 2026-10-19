"""Do all the models' order-level errors share one mechanism, and does it survive weight noise?

The example model's 1,403 wrong orderings at the mean magnitude set all have the 2nd max first and the max last
(research/findings/orderings.md). For the six base models and eight suite copies:
  1. At the mean magnitude set: the exact error over all 10! orderings, how many (p, q) cells hold 50 / 90 / 99% of
     it, the share with the max at position 9 and at 7-9, what the wrong predictions pick, and that error as a share of
     the model's total error (its truth from the bench cache).
  2. Over 2^22 Gaussian inputs: the share of all errors with the max at 7-9 and at 9, against the 30% and 10% of mass
     those positions hold, and the share of errors near a tie (S - T or M - S below 0.1).

Finding: research/findings/orderings.md
"""
import torch as th

from az.bench import suite, truth
from az.model import NAMES, load, logits
from experiments.common import exact_acc, inputs, order_stat_means, orderings, ranks

y = order_stat_means()
_, pos = orderings()
p_pos, q_pos, r_pos = pos[:, 0], pos[:, 1], pos[:, 2]
labels, ws = suite(0.002, 32)
acc_true = dict(zip(labels, truth(0.002, 32)))
acc_true.update(zip(NAMES, (0.9533, 0.9102, 0.8803, 0.9492, 0.9319, 0.8904)))  # noise_sensitivity.txt
copies = [0, 1, 32, 33, 64, 128, 160, 161]
x = inputs(2**22, seed=11)
srt, vals = ranks(x)
near = ((vals[:, 1] - vals[:, 2]) < 0.1) | ((vals[:, 0] - vals[:, 1]) < 0.1)

print("1. order-level errors at the mean magnitude set")
print(f"   {'model':10s} {'error':>7s} {'of total':>8s}  {'(p,q) cells 50/90/99%':>22s}  {'p = 9':>5s} {'p >= 7':>6s}  {'picks 3rd / max':>15s}")
for name, w in [(n, load(n)) for n in NAMES] + [(labels[i], ws[i]) for i in copies]:
    acc, _, pred = exact_acc(w, y)
    wrong = pred != q_pos
    n = int(wrong.sum())
    _, cnt = th.unique(p_pos[wrong] * 10 + q_pos[wrong], return_counts=True)
    cs = cnt.sort(descending=True).values.cumsum(0).double() / n
    need = "/".join(str(int((cs < f).sum().item()) + 1) for f in (0.5, 0.9, 0.99))
    late9 = (p_pos[wrong] == 9).double().mean().item()
    late7 = (p_pos[wrong] >= 7).double().mean().item()
    picks = f"{(pred[wrong] == r_pos[wrong]).double().mean():.2f} / {(pred[wrong] == p_pos[wrong]).double().mean():.2f}"
    print(f"   {name:10s} {1 - float(acc):7.4f} {(1 - float(acc)) / (1 - acc_true[name]):8.3f}  {need:>22s}  {late9:5.2f} {late7:6.2f}  {picks:>15s}")

print("\n2. all errors over 2^22 Gaussian inputs: share with the max at 7-9 (30% of mass), at 9 (10%), near a tie (34%)")
for name, w in [(n, load(n)) for n in NAMES] + [(labels[i], ws[i]) for i in copies]:
    with th.no_grad():
        pred = th.cat([logits([w], x[c : c + 2**20][None])[0].argmax(1) for c in range(0, len(x), 2**20)])
    err = pred != srt[:, 1]
    p = srt[:, 0]
    print(f"   {name:10s} error {err.double().mean():.4f}; share of errors with p >= 7 {(p[err] >= 7).double().mean():.2f}, p = 9 {(p[err] == 9).double().mean():.2f}, "
          f"near a tie {near[err].double().mean():.2f}; error rate at p = 9 {err[p == 9].double().mean():.4f} vs p <= 6 {err[p <= 6].double().mean():.4f}")
