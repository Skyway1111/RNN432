"""Which weights account for phase 3's loss drop? Integrate gradients along the path 2^24 -> 2^25 -> ... -> 2^30.

Splicing breaks every model (phase3_splice), so this attributes the change additively instead: each weight's share is
the integral of dLoss/dw times its change, which sums to the loss change when the path is fair. Each checkpoint is
first rescaled per neuron to unit RMS activity (function-preserving), and each segment is a straight line. A segment is
fair if its loss never rises far above its endpoints; the attribution is reported only if every segment is.
Finding: research/findings/training.md
"""
import numpy as np
import torch as th

from az.model import W, trace
from az.zoo import SEEDS, checkpoint
from experiments.common import inputs, ranks

x = inputs(2**17, seed=31)
label = ranks(x)[0][:, 1]
STEPS = 11
SEQS = [2**i for i in range(24, 31)]


def unit_rms(w):
    h = th.relu(trace(w, x)[1]).reshape(-1, 16)
    c = 1 / h.pow(2).mean(0).sqrt().clamp(min=1e-9).cpu().numpy()
    return [th.tensor(m, device=x.device) for m in W(w.Wi * c, w.Wh * c[:, None] / c[None, :], w.Wo / c[None, :])]


def loss_acc(Wi, Wh, Wo):
    h = th.zeros(len(x), 16, dtype=x.dtype, device=x.device)
    for t in range(10):
        h = th.relu(h @ Wh.T + x[:, t, None] * Wi)
    logits = h @ Wo.T
    return th.nn.functional.cross_entropy(logits, label), (logits.argmax(-1) == label).double().mean()


for seed in SEEDS:
    pts = [unit_rms(checkpoint(seed, n)) for n in SEQS]
    contrib = [th.zeros_like(p) for p in pts[0]]
    worst, exact = 0.0, 0.0
    for a, b in zip(pts, pts[1:]):
        d = [q - p for p, q in zip(a, b)]
        losses = []
        for i, al in enumerate(np.linspace(0, 1, STEPS)):
            with th.enable_grad():
                ws = [(p + al * di).requires_grad_() for p, di in zip(a, d)]
                loss, _ = loss_acc(*ws)
                grads = th.autograd.grad(loss, ws)
            wt = 0.5 if i in (0, STEPS - 1) else 1.0
            for c, g, di in zip(contrib, grads, d):
                c += wt * g * di / (STEPS - 1)
            losses.append(loss.item())
        worst = max(worst, max(losses) - max(losses[0], losses[-1]))
        exact += losses[-1] - losses[0]
    Ci, Ch, Co = contrib
    total = sum(c.sum().item() for c in contrib)
    rel = [((q - p).norm() / p.norm()).item() for p, q in zip(pts[0], pts[-1])]
    print(f"seed {seed}: relative change 2^24->2^30, unit-RMS coords: Wi {rel[0]:.2f} Wh {rel[1]:.2f} Wo {rel[2]:.2f}")
    print(f"   worst rise of loss above a segment's endpoints: {worst:.3f}; loss change exact {exact:+.4f}, attributed {total:+.4f}")
    if worst > 0.05 or abs(total - exact) > 0.1 * abs(exact):
        print("   path not fair; attribution withheld")
        continue
    print(f"   share of loss drop: Wi {Ci.sum().item() / total:.2f} Wh {Ch.sum().item() / total:.2f} Wo {Co.sum().item() / total:.2f}")
    neuron = (Ci + Ch.sum(1) + Ch.sum(0) - Ch.diagonal() + Co.sum(0)) / total
    print(f"   share by neuron (in + out): {np.round(neuron.cpu().numpy(), 2)}")
