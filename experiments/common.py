"""Shared setup for experiments: Gaussian inputs on the GPU, their rank structure, and exact accuracy over orderings."""
import itertools
from fractions import Fraction
from functools import cache
from math import factorial

import numpy as np
import torch as th
from scipy import integrate
from scipy.stats import norm

from az.model import D_HID, N_IN, trace

th.set_grad_enabled(False)
DEV = "cuda" if th.cuda.is_available() else "cpu"
N_ORD = factorial(N_IN)


def order_stat_means() -> np.ndarray:
    """E[y_k] for the sorted values of 10 i.i.d. N(0,1), by deterministic quadrature."""
    def dens(y, k):
        return y * N_ORD / (factorial(k - 1) * factorial(10 - k)) * norm.cdf(y) ** (10 - k) * norm.sf(y) ** (k - 1) * norm.pdf(y)

    return np.array([integrate.quad(dens, -12, 12, args=(k,), epsabs=1e-13)[0] for k in range(1, 11)])


@cache
def orderings() -> tuple[th.Tensor, th.Tensor]:
    """All 10! orderings in lexicographic order: seq[j, t] = rank (0 = largest) fed at step t; pos[j, rank] = its step."""
    it = itertools.chain.from_iterable(itertools.permutations(range(N_IN)))
    seq = th.from_numpy(np.fromiter(it, np.int8, N_IN * N_ORD).reshape(-1, N_IN)).to(DEV, th.long)
    return seq, seq.argsort(1)


def exact_acc(w, y) -> tuple[Fraction, float, th.Tensor]:
    """Accuracy of w over all 10! orderings of magnitudes y (largest first), exactly; the smallest top-2 logit gap,
    which certifies float64 rounding cannot flip a decision; and the prediction per ordering of orderings().
    Orderings share prefixes, so the pass walks the prefix tree: 6.2M recurrent steps instead of 36M."""
    seq, pos = orderings()
    vals = th.as_tensor(np.ascontiguousarray(y), dtype=th.float64, device=DEV)
    Wi, Wh, Wo = (th.tensor(a, dtype=th.float64, device=DEV) for a in w)
    h = th.zeros(1, D_HID, dtype=th.float64, device=DEV)
    for t, m in enumerate(range(N_IN, 0, -1)):
        x = vals[seq[:: factorial(m - 1), t]].view(-1, m, 1)
        h = th.relu((h @ Wh.T)[:, None] + x * Wi).reshape(-1, D_HID)
    L = h @ Wo.T
    top = L.topk(2, dim=1).values
    pred = L.argmax(1)
    return Fraction((pred == pos[:, 1]).sum().item(), N_ORD), (top[:, 0] - top[:, 1]).min().item(), pred


def contributions(w, x: th.Tensor) -> th.Tensor:
    """(N, 10, 16): the part of h_10 each input x_t contributes, P_t W_hi x_t with P_t the gated product of W_hh.
    Exact by piecewise linearity: the ten parts sum to h_10."""
    Wi, Wh, _ = (th.tensor(a, dtype=x.dtype, device=x.device) for a in w)
    h = th.zeros(len(x), D_HID, dtype=x.dtype, device=x.device)
    C = th.zeros(len(x), N_IN, D_HID, dtype=x.dtype, device=x.device)
    for t in range(N_IN):
        z = h @ Wh.T + x[:, t, None] * Wi
        on = (z > 0).to(x.dtype)
        C = C @ Wh.T
        C[:, t] = x[:, t, None] * Wi
        C = C * on[:, None]
        h = z * on
    return C


def inputs(n: int, seed: int) -> th.Tensor:
    g = th.Generator(device=DEV).manual_seed(seed)
    return th.randn(n, 10, generator=g, device=DEV, dtype=th.float64)


def ranks(x: th.Tensor):
    """Positions ordered by value, largest first (N, 10), and the sorted values."""
    srt = x.argsort(-1, descending=True)
    return srt, x.gather(1, srt)


ROLES = ("max lagged", "max", "new-record", "max-last", "max\\x[-2]-last", "max\\x[-3]-last", "2nd max", "2nd max-last", "last+", "last-")


def role_r2(w, x: th.Tensor, first: int = 3, decode: bool = False) -> th.Tensor:
    """R² (16 neurons, len(ROLES)) of each neuron's activity fit by one scaled candidate quantity, pooled over steps.
    With decode=True instead: R² (len(ROLES),) of each quantity read linearly from all 16 neurons (distributed codes).

    After reading x_0..x_k: max = max(0, x_0..x_k), lagged = the same without x_k, max\\x[-j] drops x_{k-j+1},
    2nd max = max(0, second largest), last = x_k. Steps k >= first.
    """
    h = th.relu(trace(w, x)[1])
    zero = th.zeros(len(x), 1, dtype=x.dtype, device=x.device)
    feats = []
    for k in range(first, 10):
        seen = th.cat([zero, x[:, : k + 1]], 1)
        top2 = seen.topk(2, dim=1).values
        last = x[:, k]
        mx = top2[:, 0]
        loo = [th.cat([seen[:, : k + 1 - j], seen[:, k + 2 - j :]], 1).max(1).values for j in range(3)]
        feats.append(th.stack([loo[0], mx, mx - loo[0], mx - last, loo[1] - last, loo[2] - last,
                               top2[:, 1].clamp(min=0), th.relu(top2[:, 1] - last), th.relu(last), th.relu(-last)], 1))
    F = th.stack(feats, 1).reshape(-1, len(ROLES))
    H = h[:, first:].reshape(-1, 16)
    if decode:
        fit = H @ th.linalg.lstsq(H, F).solution
        return 1 - ((F - fit) ** 2).sum(0) / ((F - F.mean(0)) ** 2).sum(0)
    coef = (H.T @ F) / (F * F).sum(0)
    resid = (H * H).sum(0)[:, None] - coef * coef * (F * F).sum(0)
    den = ((H - H.mean(0)) ** 2).sum(0)[:, None]
    return th.where(den > 1e-9 * len(H), 1 - resid / den.clamp(min=1e-300), th.zeros_like(resid))


def group_mean(v: th.Tensor, key: th.Tensor, n: int) -> th.Tensor:
    """Mean of v (N, ...) within each key in range(n), broadcast back to (N, ...)."""
    cnt = th.zeros(n, device=v.device, dtype=v.dtype).index_add_(0, key, th.ones(len(v), device=v.device, dtype=v.dtype))
    s = th.zeros((n,) + v.shape[1:], device=v.device, dtype=v.dtype).index_add_(0, key, v)
    return (s / cnt.clamp(min=1).view(-1, *[1] * (v.dim() - 1)))[key]
