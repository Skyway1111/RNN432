"""Which stations carry the cascade near a tie?

The decision near a tie is a cascade (research/findings/circuit.md): round 1 flips the gates that are thresholds on
their own step's input, and later rounds retune gates downstream. For 128 configurations (family, T, M) at the tie
S = T and 4,096 draws of the lower inputs each, the clean example and three suite copies, per (step, neuron) gate:
  A: how often round 1 flips it against the reference route (a station firing from its own input);
  B: how often the converged route differs from round 1 at it (retuned by other stations);
  C: the mean number of retuned gates on draws where it fired at round 1, against the mean over all draws.
Reported: the maps by step; the share of retunings at the steps that read M, S and T; the ten most active stations
after step 0 (where the first input's sign flips every gate at once, a station on its own) and the ten most retuned
gates, with their shares of all flips; and how concentrated the round-1 flip sets are. The clean example's A and B
maps go to results/experiments/cascade_stations.npy for the explorer (python -m web.explorer.build).

Finding: research/findings/circuit.md
"""
from pathlib import Path

import numpy as np
import torch as th
from scipy.stats import norm

from az.bench import suite
from az.estimators.gauss_margin import FAMS
from az.estimators.lattice import _cdf_t
from az.model import load
from experiments.common import DEV
from experiments.station_rounds import place, run

labels, ws = suite(0.002, 32)
rng = np.random.default_rng(9)
th.manual_seed(9)
t_grid, cdf_t = _cdf_t()
C, D, ROUNDS = 128, 4096, 12
OUT = Path(__file__).resolve().parent.parent / "results" / "experiments" / "cascade_stations.npy"

fam_np = rng.integers(720, size=C)
T_np = np.interp(rng.uniform(size=C), cdf_t, t_grid)
M_np = norm.ppf(norm.cdf(T_np) + (1 - norm.cdf(T_np)) * np.sqrt(rng.uniform(size=C)))
fam, T, M = (th.tensor(np.repeat(a, D), device=DEV) for a in (fam_np, T_np, M_np))
z = th.special.ndtri(th.special.ndtr(T)[:, None] * th.rand(C * D, 7, device=DEV, dtype=th.float64))
X, q = place(fam, T, M, T + 1e-9, z)
zbar = th.special.ndtri(th.special.ndtr(T)[:, None] * th.tensor((np.arange(7) + 0.5) / 7, device=DEV))
Xbar, _ = place(fam, T, M, T + 1e-9, zbar)
is_top = th.zeros(C * D, 10, dtype=th.bool, device=DEV).scatter_(1, th.tensor(FAMS, device=DEV)[fam], True)

saved = {}
for name, w in [("example", load("example")), ("example/0", ws[0]), ("s0/0", ws[32]), ("s4/0", ws[160])]:
    W = [th.tensor(a, device=DEV, dtype=th.float64) for a in w]
    with th.no_grad():
        P_true, _, _ = run(W, X)
        P_ref, H_ref, _ = run(W, Xbar)
        P1, H1, _ = run(W, X, from_states=H_ref)
        Pk, Hk = P1, H1
        for _ in range(ROUNDS):
            Pn, Hn, _ = run(W, X, from_states=Hk)
            if (Pn == Pk).all():
                break
            Pk, Hk = Pn, Hn
    conv = (Pk == P_true).all((1, 2)).double().mean().item()
    F = P1 != P_ref  # (N, 10, 16) stations that fired at round 1
    R = Pk != P1  # retuned later
    A, B = F.double().mean(0), R.double().mean(0)
    nR = R.sum((1, 2)).double()
    Cg = th.zeros(10, 16, dtype=th.float64, device=DEV)
    for t in range(10):
        for n in range(16):
            sel = F[:, t, n]
            Cg[t, n] = nR[sel].mean() if sel.any() else th.nan
    print(f"{name}: converged to the true route on {conv:.4f} of draws; gates off the reference per draw: at round 1 {F.sum((1, 2)).double().mean():.2f}, "
          f"retuned later {nR.mean():.2f}; draws with no round-1 flip {(F.sum((1, 2)) == 0).double().mean():.3f}, with none retuned {(nR == 0).double().mean():.3f}")
    print(f"  stations firing at round 1, by step (sum over neurons): {A.sum(1).cpu().numpy().round(2).tolist()}")
    print(f"  gates retuned later, by step:                           {B.sum(1).cpu().numpy().round(2).tolist()}")
    low = is_top.logical_not()[:, :, None].expand(-1, -1, 16)
    print(f"  share of retunings at the steps that read M, S or T: {1 - R[low].double().sum() / R.double().sum():.3f}")
    sign0 = F[:, 0].any(1)
    print(f"  the first input's sign: flips every gate at step 0 on {sign0.double().mean():.3f} of draws, with {nR[sign0].mean():.1f} gates retuned later against {nR[~sign0].mean():.1f} without it")
    for label, mp in (("most active stations after step 0 (round-1 flip rate; mean retuned gates on their draws, baseline %.2f)" % nR.mean(), A), ("most retuned gates (retune rate)", B)):
        flat = mp.clone()
        if mp is A:
            flat[0] = 0
        flat = flat.flatten()
        top = flat.argsort(descending=True)[:10]
        share = flat[top].sum() / flat.sum()
        items = ", ".join(f"step {int(i) // 16} n{int(i) % 16} {flat[i]:.3f}" + (f" ({Cg.flatten()[i]:.1f})" if mp is A else "") for i in top)
        print(f"  {label}: top ten hold {share:.2f} of the total: {items}")
    key = (F.reshape(len(F), 160).long() * th.tensor([1 << (i % 62) for i in range(160)], device=DEV)).sum(1)
    _, cnt = th.unique(key, return_counts=True)
    cs = cnt.sort(descending=True).values.cumsum(0).double() / len(F)
    print(f"  distinct round-1 flip sets {len(cnt)}; sets for 50 / 90% of draws {int((cs < 0.5).sum()) + 1} / {int((cs < 0.9).sum()) + 1}")
    if name == "example":
        saved = dict(flip1=A.cpu().numpy(), retune=B.cpu().numpy())
np.save(OUT, np.stack([saved["flip1"], saved["retune"]]))
print(f"saved the clean example's station maps (2, 10, 16) to {OUT.name}")
