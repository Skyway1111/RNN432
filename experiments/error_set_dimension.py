"""How much of the error variance do low-dimensional features (tie gaps, top positions, sign count) explain?

Finding: research/findings/errors.md
"""
import torch as th

from az.model import load, noisy, trace
from experiments.common import group_mean, inputs, ranks

N = 2**22
x = inputs(N, seed=14)
srt, y = ranks(x)
p, q, r = srt[:, 0], srt[:, 1], srt[:, 2]
sc = y[:, :3].abs().mean(1) + 1e-9
edges = th.tensor([0.02, 0.05, 0.1, 0.2, 0.4, 0.8], device=x.device, dtype=th.float64)
bST = th.bucketize(((y[:, 1] - y[:, 2]) / sc).contiguous(), edges)
bMS = th.bucketize(((y[:, 0] - y[:, 1]) / sc).contiguous(), edges)
npos = (y > 0).sum(1).clamp(max=3)
gaps = bST * 7 + bMS
keys = {"gaps": gaps, "gaps+signs": gaps * 4 + npos, "(p,q)": p * 10 + q, "gaps+(p,q)": gaps * 100 + p * 10 + q,
        "gaps+(p,q,r)": (gaps * 100 + p * 10 + q) * 10 + r, "gaps+(p,q,r)+signs": ((gaps * 100 + p * 10 + q) * 10 + r) * 4 + npos}
for name in ["example", "s0"]:
    for k, w in enumerate([load(name)] + noisy(load(name), 0.002, 1, seed=3)):
        err = (trace(w, x)[0].argmax(-1) != q).double()
        out = []
        for kn, key in keys.items():
            u, inv = th.unique(key, return_inverse=True)
            out.append(f"{kn} {1 - ((err - group_mean(err, inv, len(u))) ** 2).mean().item() / err.var().item():.3f}")
        print(f"{name}{' noisy' if k else ''} error {err.mean().item():.3f} | variance explained: " + " | ".join(out))
