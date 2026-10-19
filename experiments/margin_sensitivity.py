"""Can the reference routes predict which weights matter, without probing: margins of 270 reference inputs and their gradients?

Reference inputs: one per (p, q) at the mean magnitude set, and two near-tie variants (S - T = 0.05, M - S = 0.05). For each
weight, three deductive predictors of its effect: the mean relative margin sensitivity |dm/dw| / |m|, the linearized share of
reference inputs that flip at a step of 0.1, and the exact share that flip when the weight is actually stepped. Compared by
rank correlation with the ground-truth accuracy drops in weight_whatif.npy. A sampled, smoothed accuracy gradient on Gaussian
inputs is included as the first-order ceiling. Example model.

Finding: research/findings/whatif.md
"""
import numpy as np
import torch as th
from scipy.stats import spearmanr

from az.model import load
from experiments.common import DEV, order_stat_means

np.set_printoptions(precision=2, suppress=True, linewidth=200)
w = load("example")
Wi, Wh, Wo = (th.tensor(a, dtype=th.float64, device=DEV, requires_grad=True) for a in w)
params = (Wi, Wh, Wo)
names = [f"Wi[{i}]" for i in range(16)] + [f"Wh[{i},{j}]" for i in range(16) for j in range(16)] + [f"Wo[{i},{j}]" for i in range(10) for j in range(16)]


def forward(x, Wi=Wi, Wh=Wh, Wo=Wo):
    h = th.zeros(len(x), 16, dtype=th.float64, device=DEV)
    for t in range(10):
        h = th.relu(h @ Wh.T + x[:, t, None] * Wi)
    return h @ Wo.T


def margins(L, q):
    other = L.clone()
    other[th.arange(len(L)), q] = -1e9
    return L[th.arange(len(L)), q] - other.max(1).values


ystar = order_stat_means()
sets = [ystar.copy(), ystar.copy(), ystar.copy()]
sets[1][2] = sets[1][1] - 0.05  # S - T = 0.05
sets[2][1] = sets[2][0] - 0.05  # M - S = 0.05
pairs = [(a, b) for a in range(10) for b in range(10) if a != b]
X, Q = [], []
for y in sets:
    for a, b in pairs:
        v = np.zeros(10)
        v[a], v[b] = y[0], y[1]
        v[[t for t in range(10) if t not in (a, b)]] = y[2:]
        X.append(v)
        Q.append(b)
X, Q = th.tensor(np.array(X), device=DEV), th.tensor(Q, device=DEV)
with th.enable_grad():  # common.py disables autograd globally
    m = margins(forward(X), Q)
    G = th.stack([th.cat([g.reshape(-1) for g in th.autograd.grad(m[i], params, retain_graph=True)]) for i in range(len(X))])  # (270, 432)
md = m.detach()
print(f"reference inputs {len(X)}: margin mean {md.mean():.1f}, min {md.min():.1f}, wrong {(md < 0).sum().item()}")
rel = (G.abs() / md.abs()[:, None]).mean(0)
lin_flip = ((md[:, None] + 0.1 * G < 0).double().mean(0) + (md[:, None] - 0.1 * G < 0).double().mean(0)) / 2

base = np.concatenate([w.Wi.ravel(), w.Wh.ravel(), w.Wo.ravel()])
exact_flip = np.zeros(432)
with th.no_grad():
    for i in range(432):
        f = 0.0
        for d in (-0.1, 0.1):
            v = base.copy()
            v[i] += d
            L = forward(X, th.tensor(v[:16], device=DEV), th.tensor(v[16:272].reshape(16, 16), device=DEV), th.tensor(v[272:].reshape(10, 16), device=DEV))
            f += (margins(L, Q) < 0).double().mean().item() / 2
        exact_flip[i] = f - (md < 0).double().mean().item()

xg = th.randn(2**16, 10, generator=th.Generator(device=DEV).manual_seed(5), device=DEV, dtype=th.float64)
qg = xg.argsort(1)[:, -2]
tau = 3.0
with th.enable_grad():
    smooth = th.sigmoid(margins(forward(xg), qg) / tau).mean()
    sampled = th.cat([g.reshape(-1) for g in th.autograd.grad(smooth, params)]).abs()

truth = np.load("results/experiments/weight_whatif.npy")
base_acc = 0.9532
drop_01 = np.abs(base_acc - truth[:, [1, 4]]).mean(1)
drop_002 = np.abs(base_acc - truth[:, [2, 3]]).mean(1)
preds = {"relative margin sensitivity": rel.cpu().numpy(), "linearized flips at 0.1": lin_flip.cpu().numpy(), "exact flips at 0.1": exact_flip, "sampled smoothed gradient": sampled.cpu().numpy()}
print("\nSpearman rank correlation with the ground-truth |accuracy change| at steps of 0.1 / 0.02, and overlap of the top 30 weights at 0.1")
top_truth = set(np.argsort(-drop_01)[:30])
for k, v in preds.items():
    print(f"  {k:28s}: {spearmanr(v, drop_01).correlation:.3f} / {spearmanr(v, drop_002).correlation:.3f};  top-30 overlap {len(top_truth & set(np.argsort(-v)[:30]))}")
for lo, hi, mat in [(0, 16, "Wi"), (16, 272, "Wh"), (272, 432, "Wo")]:
    print(f"  within {mat}: " + "; ".join(f"{k} {spearmanr(v[lo:hi], drop_01[lo:hi]).correlation:.2f}" for k, v in preds.items()))
print("\ntop 15 weights by relative margin sensitivity, with the ground-truth drop at +-0.1")
for i in np.argsort(-rel.cpu().numpy())[:15]:
    print(f"  {names[i]:10s} = {base[i]:+7.3f}  rel {rel[i]:.3f}  truth drop {drop_01[i]:.4f}")
