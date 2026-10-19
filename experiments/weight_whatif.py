"""What does changing one weight do to accuracy? Ground truth for the explorer's what-if: every weight at six step sizes.

All models in a batch see the same 2^20 inputs, so differences between them carry no sampling noise.
Saves the table to results/experiments/weight_whatif.npy (432 x 6, accuracy after the step). Example model.

Finding: research/findings/whatif.md
"""
from pathlib import Path

import numpy as np
import torch as th

from az.model import W, correct, load

DELTAS = [-0.5, -0.1, -0.02, 0.02, 0.1, 0.5]
N = 2**20
w = load("example")
names = [f"Wi[{i}]" for i in range(16)] + [f"Wh[{i},{j}]" for i in range(16) for j in range(16)] + [f"Wo[{i},{j}]" for i in range(10) for j in range(16)]
base = np.concatenate([w.Wi.ravel(), w.Wh.ravel(), w.Wo.ravel()])


def unpack(v):
    return W(v[:16].reshape(16), v[16:272].reshape(16, 16), v[272:].reshape(10, 16))


x = th.randn(N, 10, generator=th.Generator(device="cuda").manual_seed(31), device="cuda", dtype=th.float64)
jobs = [(i, d) for i in range(432) for d in DELTAS]
acc = np.zeros(len(jobs))
with th.no_grad():
    for start in range(0, len(jobs), 16):
        ws = []
        for i, d in jobs[start : start + 16]:
            v = base.copy()
            v[i] += d
            ws.append(unpack(v))
        hits = th.zeros(len(ws), dtype=th.float64, device="cuda")
        for c in range(0, N, 2**18):
            hits += correct(ws, x[c : c + 2**18][None].expand(len(ws), -1, -1)).double().sum(1)
        acc[start : start + len(ws)] = (hits / N).cpu().numpy()
    base_acc = (correct([w], x[None]).double().sum() / N).item()
table = acc.reshape(432, len(DELTAS))
np.save(Path("results/experiments/weight_whatif.npy"), table)
print(f"base accuracy {base_acc:.4f}; deltas {DELTAS}")
drop = base_acc - table
print("\nmean drop by matrix and step size (rows Wi, Wh, Wo):")
for lo, hi, m in [(0, 16, "Wi"), (16, 272, "Wh"), (272, 432, "Wo")]:
    print(f"  {m}: " + "  ".join(f"{d:+.2f}: {drop[lo:hi, k].mean():+.4f}" for k, d in enumerate(DELTAS)))
sens = np.abs(drop[:, 2:4]).mean(1) / 0.02
order = np.argsort(-sens)
print("\n30 most sensitive weights by |drop| per unit step at +-0.02; then the drop at each step size")
for i in order[:30]:
    print(f"  {names[i]:10s} = {base[i]:+7.3f}  |d acc / d w| {sens[i]:6.3f}   " + " ".join(f"{drop[i, k]:+.4f}" for k in range(len(DELTAS))))
print("\nweights whose change at +-0.1 improves accuracy by > 0.002:")
for i in np.where(drop[:, [1, 4]].min(1) < -0.002)[0]:
    print(f"  {names[i]:10s} = {base[i]:+7.3f}   -0.1: {-drop[i, 1]:+.4f}   +0.1: {-drop[i, 4]:+.4f}")
print("\nshare of total |drop| at +-0.1 carried by the top 10 / 30 / 100 weights: "
      + " / ".join(f"{np.sort(np.abs(drop[:, [1, 4]]).mean(1))[::-1][:k].sum() / np.abs(drop[:, [1, 4]]).mean(1).sum():.2f}" for k in (10, 30, 100)))
