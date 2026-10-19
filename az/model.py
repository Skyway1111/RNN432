"""The 2nd-argmax RNNs M_{16,10}: weights, forward pass, and the noisy-model suite.

h_0 = 0, h_{t+1} = ReLU(Wh h_t + Wi x_t), logits = Wo h_10, x ~ N(0, I_10).
Weights come from data/models.npz, converted from ARC's public bucket
gs://arc-ml-public/alg/ (example = one_layer_16_hidden_94_acc_2nd_argmax.pth,
sK = zoo/2nd_argmax_16_10_1073741824_K.pth).
"""
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch as th

DATA = Path(__file__).resolve().parent.parent / "data"
NAMES = ("example", "s0", "s1", "s2", "s3", "s4")
N_IN, D_HID = 10, 16
SAMPLE_OPS = N_IN * D_HID * (D_HID + 1) + N_IN * D_HID  # multiply-adds per forward pass


class W(NamedTuple):
    Wi: np.ndarray  # (16,)
    Wh: np.ndarray  # (16, 16)
    Wo: np.ndarray  # (10, 16)


def load(name: str) -> W:
    z = np.load(DATA / "models.npz")
    return W(z[f"{name}.Wi"], z[f"{name}.Wh"], z[f"{name}.Wo"])


def noisy(w: W, sigma: float, k: int, seed: int) -> list[W]:
    """k copies of w with i.i.d. N(0, sigma^2) added to every weight (ARC's MSE protocol)."""
    rng = np.random.default_rng(seed)
    return [W(*(a + sigma * rng.standard_normal(a.shape) for a in w)) for _ in range(k)]


def logits(ws: list[W], x: th.Tensor) -> th.Tensor:
    """Logits (B, N, 10) of B models on inputs x (B, N, 10), batched on x's device."""
    Wi, Wh, Wo = (th.tensor(np.stack(a), dtype=x.dtype, device=x.device) for a in zip(*ws))
    h = th.zeros(*x.shape[:2], D_HID, dtype=x.dtype, device=x.device)
    for t in range(N_IN):
        h = th.relu(th.einsum("bnj,bij->bni", h, Wh) + x[..., t, None] * Wi[:, None])
    return th.einsum("bnj,bij->bni", h, Wo)


def trace(w: W, x: th.Tensor, jacobian: bool = False):
    """One model on x (N, 10): logits (N, 10), pre-activations (N, 10, 16), and d logits / dx (N, 10, 10) if asked."""
    Wi, Wh, Wo = (th.tensor(a, dtype=x.dtype, device=x.device) for a in w)
    h = th.zeros(len(x), D_HID, dtype=x.dtype, device=x.device)
    J = th.zeros(len(x), D_HID, N_IN, dtype=x.dtype, device=x.device) if jacobian else None
    pre = []
    for t in range(N_IN):
        z = h @ Wh.T + x[:, t, None] * Wi
        pre.append(z)
        on = (z > 0).to(x.dtype)
        if jacobian:
            J = th.einsum("ij,njk->nik", Wh, J)
            J[:, :, t] += Wi
            J = J * on[:, :, None]
        h = z * on
    return h @ Wo.T, th.stack(pre, 1), (th.einsum("oi,nik->nok", Wo, J) if jacobian else None)


def correct(ws: list[W], x: th.Tensor) -> th.Tensor:
    """(B, N) bool: model predicts the position of the second-largest input."""
    return logits(ws, x).argmax(-1) == x.argsort(-1)[..., -2]


def accuracy(ws: list[W], n: int, seed: int, chunk: int = 2**20) -> np.ndarray:
    """Sampled accuracy of each model on n Gaussian inputs (float64, GPU if available)."""
    dev = "cuda" if th.cuda.is_available() else "cpu"
    g = th.Generator(device=dev).manual_seed(seed)
    hits = th.zeros(len(ws), dtype=th.float64, device=dev)
    with th.no_grad():
        for start in range(0, n, chunk):
            m = min(chunk, n - start)
            x = th.randn(len(ws), m, N_IN, generator=g, device=dev, dtype=th.float64)
            hits += correct(ws, x).sum(1)
    return (hits / n).cpu().numpy()
