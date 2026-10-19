"""Does a first-order margin model predict which inputs flip under weight noise: P(flip | x) = Phi(-|m| / (sigma |dm/dW_hh|))?

Per Gaussian input, the margin m = l_q' - (best other logit) where q' is the model's own prediction, and its gradient with
respect to W_hh (256 numbers) by reverse mode. Isotropic noise of sigma on W_hh shifts m by about N(0, sigma^2 |dm/dW_hh|^2)
to first order, so the predicted flip probability is Phi(-|m| / (sigma |g|)). Compared with the empirical flip rate over
64 noisy copies, in bins of the prediction, at sigma = 0.002 and 0.005. Example model.

Finding: research/findings/noise.md
"""
import numpy as np
import torch as th
from scipy.stats import norm
from torch.func import grad, vmap

from az.model import load, noisy
from experiments.common import DEV

np.set_printoptions(precision=3, suppress=True, linewidth=200)
w = load("example")
Wi, Wh, Wo = (th.tensor(a, dtype=th.float64, device=DEV) for a in w)
N, COPIES = 2**14, 64
x = th.randn(N, 10, generator=th.Generator(device=DEV).manual_seed(8), device=DEV, dtype=th.float64)


def logits(Wh_, xi):
    h = th.zeros(16, dtype=th.float64, device=DEV)
    for t in range(10):
        h = th.relu(Wh_ @ h + xi[t] * Wi)
    return Wo @ h


def margin(Wh_, xi, onehot):
    L = logits(Wh_, xi)
    return (L * onehot).sum() - (L - 1e9 * onehot).max()


with th.no_grad():
    L0 = vmap(logits, in_dims=(None, 0))(Wh, x)
pred = L0.argmax(1)
onehot = th.nn.functional.one_hot(pred, 10).double()
m = vmap(margin, in_dims=(None, 0, 0))(Wh, x, onehot).detach()
g = vmap(grad(margin), in_dims=(None, 0, 0))(Wh, x, onehot).reshape(N, -1).norm(dim=1)
print(f"margin |m|: median {m.median():.1f}, 10th pct {m.quantile(0.1):.2f};  |dm/dW_hh|: median {g.median():.0f}, 90th pct {g.quantile(0.9):.0f}")

for sigma in (0.002, 0.005):
    p_pred = th.tensor(norm.cdf(-(m / (sigma * g)).cpu().numpy()), device=DEV)
    flips = th.zeros(N, dtype=th.float64, device=DEV)
    with th.no_grad():
        for c in range(0, COPIES, 16):
            ws = noisy(w, sigma, 16, seed=100 + c)
            Whs = th.stack([th.tensor(wc.Wh, dtype=th.float64, device=DEV) for wc in ws])
            Lc = vmap(vmap(logits, in_dims=(None, 0)), in_dims=(0, None))(Whs, x)  # (16, N, 10)
            flips += (Lc.argmax(2) != pred[None]).double().sum(0)
    p_emp = flips / COPIES
    print(f"\nsigma {sigma}: mean predicted flip rate {p_pred.mean():.4f}, empirical {p_emp.mean():.4f};  "
          f"correlation across inputs {th.corrcoef(th.stack([p_pred, p_emp]))[0, 1]:.3f}")
    edges = [0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    print("  bins of predicted P(flip): [lo, hi)  share of inputs  mean predicted  mean empirical")
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (p_pred >= lo) & (p_pred < hi)
        if sel.any():
            print(f"    [{lo:.3f}, {hi:.3f})  {sel.double().mean():.4f}  {p_pred[sel].mean():.4f}  {p_emp[sel].mean():.4f}")
    sel = p_pred >= 0.5
    print(f"    [0.500, 1]      {sel.double().mean():.4f}  {p_pred[sel].mean():.4f}  {p_emp[sel].mean():.4f}")
    print(f"  share of empirical flips on inputs predicted below 0.01: {p_emp[p_pred < 0.01].sum() / p_emp.sum():.3f}")
