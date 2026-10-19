"""What are modes' and ce's expected ops to 10% relative RMSE per level, and where does the expected error sit?

One realization's score is a heavy-tailed draw (research/angles/mode-mixture.md), so this scores the expectation. For an
importance sampler the relative variance per point at level c is exactly m - 1, m = E_{u ~ p|A}[p(u) / q(u)] / P(A),
the mismatch, read here from each copy's own events among 2^29 nominal inputs. Both estimators spend ADAPT of their points
on the library mixture q1 and the rest on q2, the mixture their first stage learns (modes: orders; ce: component
weights), so a copy's expected relative variance per point is ADAPT (m1 - 1) + (1 - ADAPT) E[m2 - 1], q2 averaged over
seed salts 0-2 at budget 3e8; expected ops to 10% are its mean over the suite times ops per point, over 0.01.
Part 1 checks the prediction against realized errors over 64 salts; part 3 against the realized mean over 32 salts.
Run `python -m experiments.expected_score` repeatedly: each run builds missing caches (data/cache/expected_score/) for up
to 15 minutes, then prints what is complete.

Finding: research/angles/mode-mixture.md
"""
import math
import time

import numpy as np
import torch as th

from az import osp
from az.estimators import ce, modes
from az.model import DATA
from az.rare import TAIL_OPS, models, truth

DEV = "cuda"
CACHE = DATA / "cache" / "expected_score"
CACHE.mkdir(parents=True, exist_ok=True)
LV = (2.0, 4.0, 6.0, 8.0)
LOG2, KEEP, BUDGET, SALTS = 29, 20000, 3e8, 3
START = time.time()


def busy():
    return time.time() - START > 900


def events(suite, i, w):
    """{level: (event directions, at most KEEP of them, P(A) from the count)} from 2^LOG2 nominal inputs; cached."""
    path = CACHE / f"ev_{suite}_{i}.pt"
    if path.exists():
        return th.load(path)
    if busy():
        return None
    W = [th.tensor(a, dtype=th.float32, device=DEV) for a in w]
    g = th.Generator(device=DEV).manual_seed(31000 + i + (500 if suite == "heldout" else 0))
    kept, count = {c: [] for c in LV}, {c: 0 for c in LV}
    for _ in range(2 ** (LOG2 - 22)):
        z = th.randn(2**22, 10, generator=g, device=DEV)
        u = (z / z.norm(dim=1, keepdim=True)).double()
        f = modes.margin(W, u)
        for c in LV:
            e = u[f >= c]
            count[c] += len(e)
            kept[c].append(e[: max(KEEP - sum(map(len, kept[c])), 0)])
    out = {c: (th.cat(kept[c]).cpu(), count[c] / 2**LOG2) for c in LV}
    th.save(out, path)
    return out


def ce_weight(comps, alphas, u):
    lq = th.cat([th.full((len(u), 1), math.log(ce.BETA / osp.AREA), dtype=th.float64, device=DEV), alphas.log() + comps.log_ang(u)], 1)
    return th.exp(-math.log(osp.AREA) - th.logsumexp(lq, 1))


def mismatch(weight, ev):
    return np.array([weight(ev[c][0].to(DEV)).mean().item() / ev[c][1] for c in LV])


def expected(suite, i, w):
    """(estimator: expected relative variance per point times ops per point, per level) for one copy; cached."""
    path = CACHE / f"exp_{suite}_{i}.npy"
    if path.exists():
        return np.load(path, allow_pickle=True).item()
    ev = events(suite, i, w)
    if ev is None or busy():
        return None
    b = modes.base_of(w)
    out = {}
    for est, weight in (("modes", lambda q: lambda u: osp.mixture_weight(*q, modes.BETA, u)), ("ce", lambda q: lambda u: ce_weight(*q, u))):
        mod = modes if est == "modes" else ce
        q1 = mod.mixture(b)[:2]
        m1 = mismatch(weight(q1), ev)
        runs = []
        for s in range(SALTS):
            mod.SALT = s
            runs.append(mismatch(weight(mod.tail(w, BUDGET, learned=True)[2]), ev))
        mod.SALT = 0
        draw = modes.GAP_OPS if est == "modes" else ce.FINAL_OPS
        per = TAIL_OPS + mod.DRAW_OPS + draw + (mod.COMP_OPS + mod.LEARN_OPS * mod.ADAPT) * len(q1[0])  # ops per point, as tail counts them
        out[est] = (mod.ADAPT * (m1 - 1) + (1 - mod.ADAPT) * (np.mean(runs, 0) - 1)) * per
    np.save(path, out, allow_pickle=True)
    return out


def calibration():
    """Realized relative MSE of modes without its first stage, 64 salts at 1e8, against (m - 1) x ops per point / 1e8."""
    path = CACHE / "calibration.npy"
    if path.exists():
        return np.load(path, allow_pickle=True).item()
    if busy():
        return None
    labels, ws = models("heldout")
    p = truth("heldout")
    adapt, modes.ADAPT = modes.ADAPT, 0.0
    out = {}
    for lab in ("s0/1", "s3/2", "s2/0", "s4/1"):
        i = labels.index(lab)
        e = []
        for s in range(64):
            modes.SALT = s
            e.append(modes.tail(ws[i], 1e8)[0])
        per = TAIL_OPS + modes.DRAW_OPS + modes.GAP_OPS + modes.COMP_OPS * 20  # ops per point without the first stage
        real = (((np.array(e) - p[i]) / p[i]) ** 2).mean(0)[1:]
        ev = events("heldout", i, ws[i])
        m = mismatch(lambda u: osp.mixture_weight(*modes.mixture(modes.base_of(ws[i]))[:2], modes.BETA, u), ev)
        out[lab] = (real, (m - 1) * per / 1e8)
    modes.ADAPT, modes.SALT = adapt, 0
    np.save(path, out, allow_pickle=True)
    return out


def realized(suite, est):
    """Relative MSE per level averaged over the suite and 32 salts at BUDGET; the largest single squared error."""
    path = CACHE / f"realized_{suite}_{est}.npy"
    if path.exists():
        return np.load(path)
    if busy():
        return None
    mod = modes if est == "modes" else ce
    _, ws = models(suite)
    p = truth(suite)
    err = []
    for s in range(32):
        mod.SALT = s
        err.append(((np.array([mod.tail(w, BUDGET)[0] for w in ws]) - p) / p) ** 2)
    mod.SALT = 0
    err = np.array(err)
    out = np.stack([err.mean((0, 1)), err.max((0, 1))])
    np.save(path, out)
    return out


th.set_grad_enabled(False)
print("Part 1: realized relative MSE of modes without its first stage (64 salts, 1e8) / predicted from the mismatch, c = 2, 4, 6, 8")
cal = calibration()
if cal is None:
    print("  incomplete; rerun")
else:
    for lab, (real, pred) in cal.items():
        print(f"  {lab:6s} " + "  ".join(f"{r:.4f}/{q:.4f}" for r, q in zip(real, pred)))

for suite in ("heldout", "bench"):
    labels, ws = models(suite)
    rows = [expected(suite, i, w) for i, w in enumerate(ws)]
    print(f"\nPart 2, {suite}: expected ops to 10% relative RMSE, first stages included, c = 2, 4, 6, 8")
    if any(r is None for r in rows):
        print(f"  {sum(r is not None for r in rows)} of {len(rows)} copies done; rerun")
        continue
    R = {e: np.array([r[e] for r in rows]) for e in ("modes", "ce")}
    o = {e: R[e].mean(0) / 0.01 for e in R}
    for e in R:
        print(f"  {e:5s} " + " ".join(f"{v:9.2e}" for v in o[e]) + "   largest copy's share " + " ".join(f"{v:.2f}" for v in R[e].max(0) / R[e].sum(0)))
    print("  edge, ce over modes " + " ".join(f"{v:6.2f}" for v in o["ce"] / o["modes"]))
    for b, name in enumerate(("example", "s0", "s1", "s2", "s3", "s4")):
        sl = slice(32 * b, 32 * b + 32)
        print(f"  {name:7s} share of the expected error: modes " + " ".join(f"{v:.2f}" for v in R["modes"][sl].sum(0) / R["modes"].sum(0))
              + "   ce " + " ".join(f"{v:.2f}" for v in R["ce"][sl].sum(0) / R["ce"].sum(0)))

print(f"\nPart 3, heldout: realized relative MSE averaged over 32 salts at {BUDGET:.0e}, c = 0, 2, 4, 6, 8; largest single squared error")
got = {e: realized("heldout", e) for e in ("modes", "ce")}
if any(v is None for v in got.values()):
    print("  incomplete; rerun")
else:
    for e, v in got.items():
        print(f"  {e:5s} " + " ".join(f"{x:.4f}" for x in v[0]) + "   largest " + " ".join(f"{x:.1f}" for x in v[1]))
    print("  ratio ce / modes " + " ".join(f"{x:.2f}" for x in got["ce"][0] / got["modes"][0]))
