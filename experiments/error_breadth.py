"""Are errors confined to near-ties, and does weight noise change that?

Finding: research/findings/errors.md
"""
from az.model import load, noisy, trace
from experiments.common import inputs, ranks

x = inputs(2**20, seed=10)
srt, y = ranks(x)
p, q, r = srt[:, 0], srt[:, 1], srt[:, 2]
near = (y[:, 1] - y[:, 2] < 0.1) | (y[:, 0] - y[:, 1] < 0.1)
print(f"near-tie mass (S-T or M-S < 0.1): {near.double().mean().item():.3f}")
for name in ["example", "s0"]:
    base = load(name)
    for k, w in enumerate([base] + noisy(base, 0.002, 3, seed=5)):
        pred = trace(w, x)[0].argmax(-1)
        err = pred != q
        e = err.double()
        print(f"{name}{' noisy' + str(k) if k else ''}: error {e.mean().item():.4f}"
              f"  predicts 3rd {(pred[err] == r[err]).double().mean().item():.2f} max {(pred[err] == p[err]).double().mean().item():.2f}"
              f"  error rate near {e[near].mean().item():.3f} far {e[~near].mean().item():.3f}"
              f"  share of errors near {near[err].double().mean().item():.2f}")
