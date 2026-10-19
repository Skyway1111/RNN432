"""Training checkpoints of the five zoo models, from ARC's public bucket into data/checkpoints.npz.

python -m az.zoo   fetches all 5 seeds x 12 checkpoints (after 0, 2^20, ..., 2^30 training sequences).
The example model has no checkpoints. Files load with weights_only=True: tensors, no code.
"""
import io
import urllib.request

import numpy as np
import torch as th

from az.model import DATA, W

SEEDS = range(5)
SEQS = (0,) + tuple(2**i for i in range(20, 31))
URL = "https://storage.googleapis.com/arc-ml-public/alg/zoo/2nd_argmax_16_10_{n}_{seed}.pth"
PATH = DATA / "checkpoints.npz"


def checkpoint(seed: int, n: int) -> W:
    z = np.load(PATH)
    return W(*(z[f"s{seed}.{n}.{k}"] for k in ("Wi", "Wh", "Wo")))


def fetch() -> None:
    out = {}
    for seed in SEEDS:
        for n in SEQS:
            with urllib.request.urlopen(URL.format(n=n, seed=seed), timeout=30) as r:
                sd = th.load(io.BytesIO(r.read()), weights_only=True, map_location="cpu")
            out[f"s{seed}.{n}.Wi"] = sd["rnn.weight_ih_l0"][:, 0].double().numpy()
            out[f"s{seed}.{n}.Wh"] = sd["rnn.weight_hh_l0"].double().numpy()
            out[f"s{seed}.{n}.Wo"] = sd["linear.weight"].double().numpy()
    np.savez(PATH, **out)


if __name__ == "__main__":
    fetch()
