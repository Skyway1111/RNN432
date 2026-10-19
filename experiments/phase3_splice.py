"""Which weights carry phase 3's accuracy gain? Splice blocks between each seed's 2^24 and 2^30 checkpoints.

Forward: start at 2^24, take one block from 2^30. Reverse: start at 2^30, put one block back to 2^24.
Each splice runs on raw weights and on weights rescaled per neuron to unit RMS activity (function-preserving),
so a mismatch in neuron scale cannot masquerade as a mechanism.
Finding: research/findings/training.md
"""
import numpy as np
import torch as th

from az.model import W, accuracy, trace
from az.zoo import SEEDS, checkpoint
from experiments.common import inputs

x = inputs(2**16, seed=30)


def unit_rms(w):
    h = th.relu(trace(w, x)[1]).reshape(-1, 16)
    c = 1 / h.pow(2).mean(0).sqrt().clamp(min=1e-9).cpu().numpy()
    return W(w.Wi * c, w.Wh * c[:, None] / c[None, :], w.Wo / c[None, :])


def splice(a, b, block, idx=None):
    """a with block taken from b; idx limits it to one neuron (its incoming row, or outgoing column)."""
    Wi, Wh, Wo = (m.copy() for m in a)
    if block == "Wi":
        Wi[:] = b.Wi
    elif block == "Wh":
        Wh[:] = b.Wh
    elif block == "Wo":
        Wo[:] = b.Wo
    elif block == "in":
        Wi[idx], Wh[idx] = b.Wi[idx], b.Wh[idx]
    elif block == "out":
        Wh[:, idx], Wo[:, idx] = b.Wh[:, idx], b.Wo[:, idx]
    return W(Wi, Wh, Wo)


def acc(ws):
    return accuracy(ws, 2**18, seed=4)


for seed in SEEDS:
    for norm in (False, True):
        a, b = checkpoint(seed, 2**24), checkpoint(seed, 2**30)
        if norm:
            a, b = unit_rms(a), unit_rms(b)
        lo, hi = acc([a, b])
        gain = hi - lo
        blocks = ["Wi", "Wh", "Wo"]
        fwd = acc([splice(a, b, k) for k in blocks])
        rev = acc([splice(b, a, k) for k in blocks])
        print(f"seed {seed} {'unit-RMS' if norm else 'raw     '}: 2^24 {lo:.3f} -> 2^30 {hi:.3f} (gain {gain:+.3f}) | "
              + " | ".join(f"{k}: fwd {(f - lo) / gain:+.2f} rev {(hi - r) / gain:+.2f}" for k, f, r in zip(blocks, fwd, rev)))
        if norm:
            fin = (acc([splice(a, b, "in", i) for i in range(16)]) - lo) / gain
            fout = (acc([splice(a, b, "out", i) for i in range(16)]) - lo) / gain
            print(f"   per-neuron forward share, incoming: {np.round(fin, 2)}")
            print(f"   per-neuron forward share, outgoing: {np.round(fout, 2)}")
