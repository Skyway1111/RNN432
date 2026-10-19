"""If an estimator could reuse work done on a copy's base model, how cheaply could it follow the copy's own accuracy?

Advice = anything computed deductively from the base's weights; the per-copy work is what counts. gauss_margin at level
(7, 8) (the 16-run design and the spread term) on the bases example, s0, s4 and their 32 copies; truth for copies from the
bench cache, for bases from 2^24 samples. Three ways to get a copy's estimate:
  1. rerun the estimator on the copy (the ceiling, 1.5e10 ops): how well the change (copy - base) follows the true change;
  2. a second-order Taylor expansion of the estimate in the weights at the base, g.d + d'Hd/2 with d = copy - base
     (advice: the gradient and Hessian-vector products; per copy about 2 x 432^2 ops): how well it follows the rerun;
  3. rerun only where the base marks a (cell, family) pair as near the boundary given the noise: the typical shift
     0.002 ||d mu / dW|| against the design's sd, and the share of pairs with |mu| < k sqrt(var + shift^2).
Taylor runs in float32 (it matches float64 to about 1e-8 in accuracy).

Finding: research/angles/advice-noise-kernel.md
"""
import numpy as np
import torch as th

from az.bench import suite, truth
from az.estimators.gauss_margin import FAMS, FREE, Q, cell_spread, grid, lower_inputs
from az.model import NAMES, accuracy, load
from experiments.common import DEV

labels, ws = suite(0.002, 32)
p_true = truth(0.002, 32)
LEVEL = (7, 8)
mass, nodes, cov = grid(LEVEL)
SHAPE = (LEVEL[0], LEVEL[1], LEVEL[1])
C, R, CH = len(nodes), 16, 60
DT = th.float32
nd = th.tensor(nodes, device=DEV)
roles = th.cat([th.stack([nd[:, 0] + nd[:, 2] + nd[:, 1], nd[:, 0] + nd[:, 2], nd[:, 0]], 1)[:, None].expand(-1, R, -1),
                lower_inputs(nd[:, 0])], 2).to(DT)  # (C, R, 10): M, S, T, then the lower inputs
order = th.cat([th.tensor(FAMS, device=DEV), th.tensor(FREE, device=DEV)], 1)
qt = th.tensor(Q, device=DEV)
CHUNKS = [th.arange(c, min(720, c + CH), device=DEV) for c in range(0, 720, CH)]
XS = []
for fs in CHUNKS:
    X = th.zeros(C, len(fs), R, 10, dtype=DT, device=DEV)
    XS.append(X.scatter_(3, order[fs][None, :, None].expand(C, -1, R, -1), roles[:, None].expand(-1, len(fs), -1, -1).contiguous()))
MASS = th.tensor(mass, device=DEV, dtype=DT)


def margins(params, k):
    Wi, Wh, Wo = params
    x = XS[k].reshape(-1, 10)
    h = th.zeros(len(x), 16, dtype=DT, device=DEV)
    for s in range(10):
        h = th.relu(h @ Wh.T + x[:, s, None] * Wi)
    L = (h @ Wo.T).reshape(C, len(CHUNKS[k]), R, 10)
    qq = qt[CHUNKS[k]][None, :, None, None].expand(C, -1, R, 1)
    return L.gather(3, qq)[..., 0] - L.scatter(3, qq, -1e9).max(3).values


def error_mass(params, k):
    m = margins(params, k)
    mu, sd = m.mean(2), m.std(2, correction=0).clamp(min=1e-9)
    return (MASS[:, None] * th.special.ndtr(-mu / (sd ** 2 + cell_spread(mu, nodes, cov, SHAPE)).sqrt())).sum() / 720


def tensors(w, grad=False):
    return [th.tensor(a, device=DEV, dtype=DT, requires_grad=grad) for a in w]


def estimate(w):
    with th.no_grad():
        return 1 - sum(error_mass(tensors(w), k).item() for k in range(len(XS)))


def taylor(P, d):
    """g.d and d'Hd / 2 of the estimate at parameters P along d."""
    lin = quad = 0.0
    with th.enable_grad():  # experiments.common turns autograd off
        for k in range(len(XS)):
            g = th.autograd.grad(-error_mass(P, k), P, create_graph=True)
            gd = sum((a * b).sum() for a, b in zip(g, d))
            hd = th.autograd.grad(gd, P)
            lin += gd.item()
            quad += 0.5 * sum((a * b).sum() for a, b in zip(hd, d)).item()
    return lin, quad


def shift_share(w):
    """Median noise shift 0.002 ||d mu / dW|| and median design sd over pairs, and shares near the boundary for k = 3, 5."""
    Wi, Wh, Wo = tensors(w)
    out = []
    for k in range(len(XS)):
        x = XS[k].reshape(-1, 10)
        hs, zs = [th.zeros(len(x), 16, dtype=DT, device=DEV)], []
        for s in range(10):
            zs.append(hs[-1] @ Wh.T + x[:, s, None] * Wi)
            hs.append(th.relu(zs[-1]))
        L = hs[-1] @ Wo.T
        q = qt[CHUNKS[k]][None, :, None].expand(C, -1, R).reshape(-1)
        c = L.scatter(1, q[:, None], -1e9).argmax(1)
        m = (L.gather(1, q[:, None]) - L.gather(1, c[:, None]))[:, 0].reshape(C, -1, R)
        dWo = th.zeros(len(x), 10, 16, dtype=DT, device=DEV)
        dWo.scatter_(1, q[:, None, None].expand(-1, 1, 16), hs[-1][:, None])
        dWo.scatter_add_(1, c[:, None, None].expand(-1, 1, 16), -hs[-1][:, None])
        delta = (Wo[q] - Wo[c]) * (zs[-1] > 0)
        dWh, dWi = th.zeros(len(x), 16, 16, dtype=DT, device=DEV), th.zeros(len(x), 16, dtype=DT, device=DEV)
        for s in range(9, -1, -1):  # backprop the margin to every weight, per input
            dWh += delta[:, :, None] * hs[s][:, None, :]
            dWi += delta * x[:, s, None]
            if s:
                delta = (delta @ Wh) * (zs[s - 1] > 0)
        G = th.cat([dWi, dWh.reshape(len(x), -1), dWo.reshape(len(x), -1)], 1).reshape(C, -1, R, 432).mean(2)
        mu, sd = m.mean(2), m.std(2, correction=0)
        out.append(th.stack([0.002 * G.norm(dim=-1), sd, mu.abs()], -1).reshape(-1, 3))
    s, sd, mu = th.cat(out).T
    var = sd ** 2 + s ** 2
    return s.median().item(), sd.median().item(), [(mu < k * var.sqrt()).double().mean().item() for k in (3, 5)]


rms = lambda a: float(np.sqrt(np.mean(np.square(a))))  # noqa: E731
for b in (0, 1, 5):
    base = load(NAMES[b])
    tb, eb = accuracy([base], 2**24, seed=99)[0], estimate(base)
    copies = range(b * 32, b * 32 + 32)
    d_true = p_true[list(copies)] - tb
    d_est = np.array([estimate(ws[j]) for j in copies]) - eb
    P = tensors(base, grad=True)
    tay = np.array([sum(taylor(P, [th.tensor(c - a, device=DEV, dtype=DT) for c, a in zip(ws[j], base)])) for j in copies])
    s, sd, share = shift_share(base)
    print(f"{NAMES[b]}: true change mean {d_true.mean():+.4f} sd {d_true.std():.4f}")
    print(f"  rerun on the copy: change follows the true change with rms {rms(d_est - d_true):.4f} (centred {np.std(d_est - d_true):.4f}), corr {np.corrcoef(d_est, d_true)[0, 1]:.3f}")
    print(f"  second-order Taylor: against the rerun rms {rms(tay - d_est):.4f}, mean {np.mean(tay - d_est):+.4f}; against the true change rms {rms(tay - d_true):.4f}")
    print(f"  noise shift 0.002 ||d mu/dW|| median {s:.2f} against design sd median {sd:.2f}; pairs to rerun at k = 3, 5: {share[0]:.2f}, {share[1]:.2f}", flush=True)
