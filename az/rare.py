"""The rare-event yardstick: how often a model is confidently wrong, at rising levels of confidence.

python -m az.rare ESTIMATOR [--budgets 1e6,1e7,...] [--suite bench|heldout]

Event at level c: the best other logit beats the label's by at least c |x|, max_{j != q} l_j - l_q >= c |x|. Both sides
scale with x, so the event depends on the input's direction only; c = 0 is the error rate.
Truth: float32 counts on 2^31 inputs per bench model and 2^30 per held-out model (noise seeds 5000 + i), cached in
data/cache/ and rebuilt when missing (about an hour and half an hour on an RTX 5080).
Score per level: relative MSE, the mean over models of ((p_hat - p) / p)^2, against sampling's analytic (1 - p) / (k p)
at k = ops // TAIL_OPS; and the ops each needs for 10% relative RMSE, interpolated over the budgets.
An estimator module az/estimators/NAME.py defines tail(w, budget) -> (estimates for each level in LEVELS, ops).
"""
import argparse
import importlib
import json
import time

import numpy as np
import torch as th

from az.bench import suite
from az.model import DATA, D_HID, N_IN, NAMES, SAMPLE_OPS, load, noisy

LEVELS = (0.0, 2.0, 4.0, 6.0, 8.0)
TAIL_OPS = SAMPLE_OPS + N_IN  # a forward pass and |x|
TRUTH_LOG2 = {"bench": 31, "heldout": 30}
TARGET = 0.1  # relative RMSE at which ops are compared
CHUNK = 2**21


def exceed(ws, n: int, seed: int) -> np.ndarray:
    """Counts (len(ws), len(LEVELS)) of confident errors among n Gaussian inputs per model, float32 on the GPU."""
    dev = "cuda" if th.cuda.is_available() else "cpu"
    Wi, Wh, Wo = (th.tensor(np.stack(a), dtype=th.float32, device=dev) for a in zip(*ws))
    lv = th.tensor(LEVELS, dtype=th.float32, device=dev)
    g = th.Generator(device=dev).manual_seed(seed)
    counts = th.zeros(len(ws), len(LEVELS), dtype=th.int64, device=dev)
    with th.no_grad():
        for start in range(0, n, CHUNK):
            m = min(CHUNK, n - start)
            x = th.randn(len(ws), m, N_IN, generator=g, device=dev)
            h = th.zeros(len(ws), m, D_HID, device=dev)
            for t in range(N_IN):
                h = th.relu(th.baddbmm(x[..., t, None] * Wi[:, None], h, Wh.transpose(1, 2)))
            L = th.bmm(h, Wo.transpose(1, 2))
            q = x.topk(2, dim=-1).indices[..., 1:]
            gap = L.scatter(-1, q, -th.inf).max(-1).values - L.gather(-1, q)[..., 0]
            counts += (gap[..., None] >= lv * x.norm(dim=-1, keepdim=True)).sum(1)
    return counts.cpu().numpy()


def models(which: str):
    if which == "bench":
        return suite(0.002, 32)
    ws = [w for i, name in enumerate(NAMES) for w in noisy(load(name), 0.002, 32, seed=5000 + i)]
    return [f"{name}/{j}" for name in NAMES for j in range(32)], ws


def truth(which: str, batches: int | None = None) -> np.ndarray | None:
    """(models, levels) event probabilities of the suite, built in cached 16-model batches. With batches set, computes at
    most that many missing batches and returns None while any remain (each bench batch takes about five minutes)."""
    stem = DATA / "cache" / f"rare_{which}_n{TRUTH_LOG2[which]}"
    if stem.with_suffix(".npy").exists():
        return np.load(stem.with_suffix(".npy"))
    _, ws = models(which)
    n = 2 ** TRUTH_LOG2[which]
    parts = [stem.parent / f"{stem.name}_m{i}.npy" for i in range(0, len(ws), 16)]
    for i, part in zip(range(0, len(ws), 16), parts):
        if not part.exists():
            if batches is not None and batches <= 0:
                return None
            np.save(part, exceed(ws[i:i + 16], n, seed=7000 + i + (0 if which == "bench" else 500)) / n)
            batches = None if batches is None else batches - 1
    p = np.concatenate([np.load(part) for part in parts])
    np.save(stem.with_suffix(".npy"), p)
    return p


def ops_at_target(ops: np.ndarray, rel_rmse: np.ndarray) -> float:
    """Ops at which relative RMSE reaches TARGET, by log-log interpolation; inf if it never does, nan if the smallest
    budget already does (widen the ladder downward)."""
    below = np.nonzero(rel_rmse <= TARGET)[0]
    if not len(below):
        return np.inf
    j = below[0]
    if j == 0:
        return np.nan
    x0, x1, y0, y1 = np.log(ops[j - 1]), np.log(ops[j]), np.log(rel_rmse[j - 1]), np.log(rel_rmse[j])
    return float(np.exp(x0 + (np.log(TARGET) - y0) * (x1 - x0) / (y1 - y0)))


def run(est_name: str, which: str, budgets: list[float]) -> dict:
    est = importlib.import_module(f"az.estimators.{est_name}")
    _, ws = models(which)
    p = truth(which)
    rows = []
    for b in budgets:
        t0 = time.perf_counter()
        out = [est.tail(w, b) for w in ws]
        e, ops = np.array([o[0] for o in out]), float(np.mean([o[1] for o in out]))
        k = max(int(ops // TAIL_OPS), 1)
        rel = np.mean(((e - p) / p) ** 2, 0)
        samp = np.mean((1 - p) / (k * p), 0)
        rows.append(dict(budget=b, ops=ops, rel_rmse=np.sqrt(rel).tolist(), rel_bias=np.mean((e - p) / p, 0).tolist(),
                         ratio=(rel / samp).tolist(), secs=time.perf_counter() - t0))
    ops = np.array([r["ops"] for r in rows])
    rr = np.array([r["rel_rmse"] for r in rows])
    mine = [ops_at_target(ops, rr[:, j]) for j in range(len(LEVELS))]
    samp = (np.mean((1 - p) / p, 0) / TARGET ** 2 * TAIL_OPS).tolist()
    return dict(estimator=est_name, suite=which, levels=LEVELS, truth_median=np.median(p, 0).tolist(), rows=rows,
                ops_at_target=mine, sampling_ops_at_target=samp, advantage=[s / m for s, m in zip(samp, mine)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("estimator")
    ap.add_argument("--suite", default="bench", choices=TRUTH_LOG2)
    ap.add_argument("--budgets", default="1e5,1e6,1e7,1e8,1e9,1e10,1e11")
    a = ap.parse_args()
    r = run(a.estimator, a.suite, [float(b) for b in a.budgets.split(",")])
    print(f"{r['estimator']}  suite={r['suite']}  levels c={LEVELS}  median p=" + " ".join(f"{v:.1e}" for v in r["truth_median"]))
    print(f"{'budget':>8} {'ops':>9}  relative RMSE per level / relative MSE over sampling's")
    for w in r["rows"]:
        print(f"{w['budget']:8.0e} {w['ops']:9.2e}  " + "  ".join(f"{e:6.3f}/{q:5.2f}" for e, q in zip(w["rel_rmse"], w["ratio"])))
    print("ops to 10% relative RMSE:      " + "  ".join(f"{v:8.1e}" for v in r["ops_at_target"]))
    print("sampling's, analytic:          " + "  ".join(f"{v:8.1e}" for v in r["sampling_ops_at_target"]))
    print("advantage (sampling / it):     " + "  ".join(f"{v:8.2f}" for v in r["advantage"]))
    out = DATA.parent / "results" / "bench" / f"{a.estimator}_rare_{a.suite}.json"
    out.write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
