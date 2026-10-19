"""ARC's challenge metric: an estimator's MSE against sampling's at equal compute.

python -m az.bench ESTIMATOR [--sigma S] [--draws D] [--budgets 1e4,1e5,...]

Suite: each base model in NAMES, plus `draws` noisy copies per model when sigma > 0.
Truth: 2^24-sample accuracy per suite model, cached in data/cache/ (regenerable, gitignored).
Output: results/bench/NAME_sigmaS.json (committed).
Sampling with k = ops // SAMPLE_OPS samples has MSE p(1-p)/k exactly, so the baseline is analytic.
An estimator module az/estimators/NAME.py defines estimate(w, budget) -> (estimate, ops).
"""
import argparse
import importlib
import json
import time

import numpy as np

from az.model import DATA, NAMES, SAMPLE_OPS, accuracy, load, noisy

TRUTH_N = 2**24


def suite(sigma: float, draws: int) -> tuple[list[str], list]:
    labels, ws = [], []
    for i, name in enumerate(NAMES):
        base = load(name)
        copies = noisy(base, sigma, draws, seed=1000 + i) if sigma > 0 else [base]
        labels += [f"{name}/{j}" for j in range(len(copies))]
        ws += copies
    return labels, ws


def truth(sigma: float, draws: int) -> np.ndarray:
    path = DATA / "cache" / f"truth_sigma{sigma}_draws{draws}_n{TRUTH_N}.npy"
    if path.exists():
        return np.load(path)
    _, ws = suite(sigma, draws)
    p = np.concatenate([accuracy(ws[i : i + 16], TRUTH_N, seed=7 + i) for i in range(0, len(ws), 16)])
    np.save(path, p)
    return p


def run(est_name: str, sigma: float, draws: int, budgets: list[float]) -> dict:
    est = importlib.import_module(f"az.estimators.{est_name}")
    labels, ws = suite(sigma, draws)
    p = truth(sigma, draws)
    rows = []
    for b in budgets:
        t0 = time.perf_counter()
        out = np.array([est.estimate(w, b) for w in ws])
        secs = time.perf_counter() - t0
        e, ops = out[:, 0], out[:, 1].mean()
        mse = float(np.mean((e - p) ** 2))
        k = max(int(ops // SAMPLE_OPS), 1)
        mse_s = float(np.mean(p * (1 - p)) / k)
        rows.append(dict(budget=b, ops=float(ops), rmse=mse**0.5, bias=float(np.mean(e - p)),
                         rmse_sampling=mse_s**0.5, mse_ratio=mse / mse_s, secs=secs))
    return dict(estimator=est_name, sigma=sigma, draws=draws, n_models=len(ws),
                truth_mean=float(p.mean()), rows=rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("estimator")
    ap.add_argument("--sigma", type=float, default=0.002)
    ap.add_argument("--draws", type=int, default=32)
    ap.add_argument("--budgets", default="1e5,1e6,1e7,1e8")
    a = ap.parse_args()
    r = run(a.estimator, a.sigma, a.draws, [float(b) for b in a.budgets.split(",")])
    print(f"{r['estimator']}  sigma={r['sigma']}  models={r['n_models']}  mean acc={r['truth_mean']:.4f}")
    print(f"{'budget':>8} {'ops':>9} {'rmse':>8} {'bias':>8} {'rmse_samp':>9} {'mse/samp':>8} {'secs':>6}")
    for w in r["rows"]:
        print(f"{w['budget']:8.0e} {w['ops']:9.2e} {w['rmse']:8.5f} {w['bias']:+8.5f} "
              f"{w['rmse_sampling']:9.5f} {w['mse_ratio']:8.3f} {w['secs']:6.1f}")
    out = DATA.parent / "results" / "bench" / f"{a.estimator}_sigma{a.sigma}.json"
    out.write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
