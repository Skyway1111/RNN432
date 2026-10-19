"""How far is the reference-route surrogate's accuracy from the model's, across the noisy suite?

A surrogate applies, to each Gaussian input, the exact linear map of a reference input with the same (p, q) or (p, q, r):
the mean magnitude set with the lower values in one fixed order (or the mean map over 8 orders). Its accuracy is a sum of
polytope masses, computable without sampling; this measures whether it is the right quantity, by evaluating surrogate and
model on the same 2^18 inputs for all 192 suite models. Variants: per (p, q) with 1 or 8 orders; per (p, q, r) with 1 order;
and reference magnitudes at the mean set or with the gaps shrunk to 0.1.

Finding: research/angles/route-polytope.md
"""
import numpy as np
import torch as th

from az.bench import suite, truth
from az.model import logits
from experiments.common import DEV, contributions, inputs, order_stat_means, ranks

np.set_printoptions(precision=4, suppress=True, linewidth=160)
x = inputs(2**18, seed=41)
srt, ys = ranks(x)
p, q, r = srt[:, 0], srt[:, 1], srt[:, 2]
key2 = p * 10 + q
key3 = (p * 10 + q) * 10 + r
ystar = order_stat_means()
near = ystar.copy()
near[1] = near[2] + 0.1
near[0] = near[1] + 0.1
rng = np.random.default_rng(0)
orders = [np.arange(8)] + [rng.permutation(8) for _ in range(7)]
orders3 = [np.arange(7)]


def reference_inputs(y, triples, n_rest, order_list):
    out, keys = [], []
    for tr in triples:
        rest = [t for t in range(10) if t not in tr]
        for o in order_list:
            v = np.zeros(10)
            for k, t in enumerate(tr):
                v[t] = y[k]
            v[rest] = y[len(tr):][o]
            out.append(v)
            keys.append(tr)
    return th.tensor(np.array(out), dtype=th.float64, device=DEV), keys


pairs = [(a, b) for a in range(10) for b in range(10) if a != b]
triples = [(a, b, c) for a in range(10) for b in range(10) for c in range(10) if len({a, b, c}) == 3]
variants = {
    "per (p,q), 1 order, mean set": (reference_inputs(ystar, pairs, 8, orders[:1]), 2),
    "per (p,q), 8 orders, mean set": (reference_inputs(ystar, pairs, 8, orders), 2),
    "per (p,q,r), 1 order, mean set": (reference_inputs(ystar, triples, 7, orders3), 3),
    "per (p,q), 1 order, gaps 0.1": (reference_inputs(near, pairs, 8, orders[:1]), 2),
    "per (p,q), 8 orders, gaps 0.1": (reference_inputs(near, pairs, 8, orders), 2),
}
labels, ws = suite(0.002, 32)
p_true = truth(0.002, 32)
bias = {k: np.zeros(len(ws)) for k in variants}
acc_model = np.zeros(len(ws))
with th.no_grad():
    for mi, w in enumerate(ws):
        Wo = th.tensor(w.Wo, device=DEV)
        Lm = logits([w], x[None])[0]
        okm = Lm.argmax(1) == q
        acc_model[mi] = okm.double().mean().item()
        for name, ((X, keys), arity) in variants.items():
            A = contributions(w, X) / X[..., None]  # (n_ref, 10, 16)
            n_orders = len(X) // (len(pairs) if arity == 2 else len(triples))
            A = A.reshape(-1, n_orders, 10, 16).mean(1)  # (n_keys, 10, 16)
            if arity == 2:
                index = th.full((100,), -1, dtype=th.long, device=DEV)
                for i, (a, b) in enumerate(pairs):
                    index[a * 10 + b] = i
                g = index[key2]
            else:
                index = th.full((1000,), -1, dtype=th.long, device=DEV)
                for i, (a, b, c) in enumerate(triples):
                    index[(a * 10 + b) * 10 + c] = i
                g = index[key3]
            h = th.einsum("nt,nti->ni", x, A[g])
            oks = (h @ Wo.T).argmax(1) == q
            bias[name][mi] = (oks.double().mean() - okm.double().mean()).item()
print(f"models {len(ws)}; model accuracy on these inputs vs truth cache: rms diff {np.sqrt(np.mean((acc_model - p_true) ** 2)):.4f}")
print(f"{'surrogate':34s} {'mean bias':>9s} {'rms bias':>8s} {'max |bias|':>10s}   by base model (mean bias over 32 copies)")
for name, b in bias.items():
    per_base = [b[i * 32 : (i + 1) * 32].mean() for i in range(6)]
    print(f"{name:34s} {b.mean():+9.4f} {np.sqrt(np.mean(b ** 2)):8.4f} {np.abs(b).max():10.4f}   {np.array(per_base)}")
print("\nfor reference: sampling RMSE at 1e6 / 1e7 / 1e8 ops = 0.019 / 0.006 / 0.002")
