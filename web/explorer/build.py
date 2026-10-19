"""Build web/explorer/data.js, everything the explorer shows, from the weights and the committed results.

python -m web.explorer.build

Recomputes the cheap facts (impulse responses, coefficients and votes by rank, margin sensitivities) and reads the
probed what-if table and the row damage from results/experiments/. The neurons' words live in web/neurons.js.
"""
import json
import re
from pathlib import Path

import numpy as np
import torch as th

from az.model import load, trace
from experiments.common import DEV, contributions, inputs, order_stat_means, ranks
from experiments.station_rounds import run

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[1] / "results" / "experiments"
DELTAS = [-0.5, -0.1, -0.02, 0.02, 0.1, 0.5]
CORE = (2, 4, 7, 6, 1, 8)

w = load("example")
Wi, Wh, Wo = (np.asarray(a, dtype=float) for a in w)
WoT = th.tensor(Wo, device=DEV)

# tuning: impulse responses for the max, and the 2nd max's part of two-impulse responses
single = th.eye(10, dtype=th.float64, device=DEV)
bump_M = th.relu(trace(w, single)[1][:, -1]).cpu().numpy()  # (10 positions, 16)
pairs = [(a, b) for a in range(10) for b in range(10) if a != b]
two = th.zeros(len(pairs), 10, dtype=th.float64, device=DEV)
for i, (a, b) in enumerate(pairs):
    two[i, a], two[i, b] = 2.0, 1.0
C2 = contributions(w, two)  # (90, 10, 16)
bump_S = np.zeros((10, 16))
for i, (a, b) in enumerate(pairs):
    bump_S[b] += C2[i, b].cpu().numpy() / 9

# coefficients and votes by rank on Gaussian inputs
x = inputs(2**16, seed=23)
srt, ys = ranks(x)
p, q, r = srt[:, 0], srt[:, 1], srt[:, 2]
C = contributions(w, x) / x[..., None]
by_rank = C.gather(1, srt[..., None].expand(-1, -1, 16))
coef_mean, coef_sd = by_rank.mean(0).T.cpu().numpy(), by_rank.std(0).T.cpu().numpy()  # (16, 10 ranks)
h10 = th.relu(trace(w, x)[1][:, -1])
mean_act = h10.mean(0).cpu().numpy()
J = C @ WoT.T
steps = th.arange(10, device=DEV)[None].expand(len(x), -1)
self_vote = J.gather(2, steps[..., None])[..., 0]
rank_of_step = srt.argsort(1)
votes = []
for k in range(10):
    m = rank_of_step == k
    votes.append(dict(rank=k + 1, self=self_vote[m].mean().item(), sd=self_vote[m].std().item(),
                      q=J.gather(2, q[:, None, None].expand(-1, 10, 1))[..., 0][m].mean().item(),
                      p=J.gather(2, p[:, None, None].expand(-1, 10, 1))[..., 0][m].mean().item(),
                      r=J.gather(2, r[:, None, None].expand(-1, 10, 1))[..., 0][m].mean().item()))
acc = (trace(w, x)[0].argmax(1) == q).double().mean().item()

# what-if: probed table and the deductive relative margin sensitivity
table = np.load(RESULTS / "weight_whatif.npy")
ystar = order_stat_means()
sets = [ystar.copy(), ystar.copy(), ystar.copy()]
sets[1][2] = sets[1][1] - 0.05
sets[2][1] = sets[2][0] - 0.05
X, Q = [], []
for y in sets:
    for a, b in pairs:
        v = np.zeros(10)
        v[a], v[b] = y[0], y[1]
        v[[t for t in range(10) if t not in (a, b)]] = y[2:]
        X.append(v)
        Q.append(b)
Xt, Qt = th.tensor(np.array(X), device=DEV), th.tensor(Q, device=DEV)
params = [th.tensor(a, dtype=th.float64, device=DEV, requires_grad=True) for a in (Wi, Wh, Wo)]
with th.enable_grad():
    h = th.zeros(len(Xt), 16, dtype=th.float64, device=DEV)
    for t in range(10):
        h = th.relu(h @ params[1].T + Xt[:, t, None] * params[0])
    L = h @ params[2].T
    other = L.clone()
    other[th.arange(len(L)), Qt] = -1e9
    m = L[th.arange(len(L)), Qt] - other.max(1).values
    G = th.stack([th.cat([g.reshape(-1) for g in th.autograd.grad(m[i], params, retain_graph=True)]) for i in range(len(Xt))])
rel = (G.abs() / m.detach().abs()[:, None]).mean(0).cpu().numpy()
rank = np.empty(432, dtype=int)
rank[np.argsort(-rel)] = np.arange(1, 433)

row_damage = [float(v) for v in re.findall(r"\d+:([\d.]+)", Path(RESULTS / "noise_fragility_map.txt").read_text().splitlines()[1])]

# the cascade: station maps at near ties (experiments/cascade_stations.py), and a near-tie input whose answer the
# cascade flips: the model is wrong on it, the reference route's gates applied to it are right
stations = np.load(RESULTS / "cascade_stations.npy")  # (2, 10, 16): round-1 flip rate, retune rate
Wt = [th.tensor(a, device=DEV, dtype=th.float64) for a in w]
xs = inputs(2**14, seed=31)
o, ys_ = ranks(xs)
near = (ys_[:, 1] - ys_[:, 2]) < 0.1
X, o = xs[near], o[near]
xr = X.clone()
quant = th.special.ndtri(th.special.ndtr(X.gather(1, o[:, 2:3])) * th.tensor((np.arange(7) + 0.5) / 7, device=DEV))
xr.scatter_(1, o[:, 3:].sort(1).values, quant)
P_ref = run(Wt, xr)[0]
wrong = run(Wt, X)[2].argmax(1) != o[:, 1]
right0 = run(Wt, X, gates=P_ref)[2].argmax(1) == o[:, 1]
flipped = None
for k in (wrong & right0).nonzero().squeeze(1).tolist():
    v = X[k].round(decimals=3)
    vr = v.clone()
    vr.scatter_(0, o[k, 3:].sort().values, quant[k])
    if run(Wt, v[None])[2].argmax(1).item() != o[k, 1].item() and run(Wt, v[None], gates=run(Wt, vr[None])[0])[2].argmax(1).item() == o[k, 1].item():
        flipped = v.cpu().numpy().tolist()
        break


neurons = []
for n in range(16):
    neurons.append(dict(index=n, group="core" if n in CORE else "timer", bumpM=bump_M[:, n].round(2).tolist(), bumpS=bump_S[:, n].round(2).tolist(),
                        coefMean=coef_mean[n].round(2).tolist(), coefSd=coef_sd[n].round(2).tolist(),
                        readout=Wo[:, n].round(2).tolist(), meanAct=round(float(mean_act[n]), 2), rowDamage=row_damage[n]))

data = dict(
    model="example (M_16,10, ARC AlgZoo)", baseAccuracy=round(acc, 4), deltas=DELTAS,
    weights=dict(Wi=Wi.round(5).tolist(), Wh=Wh.round(5).tolist(), Wo=Wo.round(5).tolist()),
    neurons=neurons, votesByRank=votes, meanSet=ystar.round(4).tolist(),
    whatif=dict(accuracy=table.round(4).tolist(), relSensitivity=rel.round(3).tolist(), rank=rank.tolist()),
    stations=dict(flip1=stations[0].round(3).tolist(), retune=stations[1].round(3).tolist()),
    examples={
        "typical": [0.3, -1.2, 1.54, 0.1, -0.4, 1.0, -0.9, 0.66, -0.1, 0.4],
        "near tie (S − T = 0.03)": [0.12, 1.0, -0.38, 0.97, 1.54, -0.66, -1.0, 0.38, -0.12, -1.54],
        "near tie the cascade flips": flipped,
        "2nd max first, max last": [1.0, 0.12, -0.38, 0.66, 0.38, -0.66, -1.0, -0.12, -1.54, 1.54],
        "two impulses": [0, 0, 1.0, 0, 0, 0, 2.0, 0, 0, 0],
    },
)
(HERE / "data.js").write_text("window.DATA = " + json.dumps(data) + ";\n", encoding="utf-8")
print(f"wrote {HERE / 'data.js'}: {len(neurons)} neurons, accuracy {acc:.4f}, 432 what-if rows, cascade example {flipped}")
