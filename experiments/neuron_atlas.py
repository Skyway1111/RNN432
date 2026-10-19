"""What does each of the example model's 16 neurons do, and which of its links matter most?

Per neuron, on Gaussian inputs: how often it is on and how large it runs; its incoming and outgoing links ranked by the
push they carry (mean |weight × source activity| per step) and by the accuracy lost when the link alone is cut to zero;
what it votes for (its readout column); and the accuracy left when it is silenced (every incoming weight zeroed, so it
never fires), when only its vote is cut (its readout column zeroed), and when only what it feeds is cut (its column of
W_hh zeroed). Then the timers' tuning: the position of a lone input that each answers most. The link-cut table goes to
results/experiments/neuron_atlas.npy for the site.

Finding: research/findings/neurons.md
"""
from pathlib import Path

import numpy as np
import torch as th

from az.model import W, load, logits, trace
from experiments.common import DEV, contributions, inputs, ranks

np.set_printoptions(precision=2, suppress=True, linewidth=200)
CORE = [2, 4, 7, 6, 1, 8]
w = load("example")
Wi, Wh, Wo = w
x = inputs(2**20, seed=61)
q = ranks(x)[0][:, 1]
_, pre, _ = trace(w, x)
h = th.relu(pre)                                     # (N, 10, 16): state after each step
prev = th.cat([th.zeros_like(h[:, :1]), h[:, :-1]], 1)  # the state each step reads


def acc(ws: list[W], n: int = 2**18, chunk: int = 64) -> np.ndarray:
    out = []
    for i in range(0, len(ws), chunk):
        part = ws[i:i + chunk]
        xs = x[:n].expand(len(part), -1, -1)
        out.append((logits(part, xs).argmax(-1) == q[:n]).double().mean(1).cpu().numpy())
    return np.concatenate(out)


base = acc([w])[0]
print(f"accuracy on the 2^18 scoring inputs: {base:.4f}")

# pushes: mean |W[i, j] h_j| over inputs and steps (j's state read at each step), and |W_in[i] x_t|
Whd, Wid = th.tensor(Wh, device=DEV), th.tensor(Wi, device=DEV)
push = (prev.abs().mean((0, 1))[None, :] * Whd.abs()).cpu().numpy()          # exact: |W h| = |W| h for h >= 0
push_in = (Wid.abs() * x.abs().mean()).cpu().numpy()

# link cuts: every nonzero W_hh entry set to 0, one at a time
cut = []
for i in range(16):
    for j in range(16):
        Wc = Wh.copy()
        Wc[i, j] = 0
        cut.append(W(Wi, Wc, Wo))
link_acc = acc(cut).reshape(16, 16)
in_cut = acc([W(np.where(np.arange(16) == n, 0, Wi), Wh, Wo) for n in range(16)])

# whole-neuron ablations
def zero_row(n):
    Wh2, Wi2 = Wh.copy(), Wi.copy()
    Wh2[n], Wi2[n] = 0, 0
    return W(Wi2, Wh2, Wo)


def zero_vote(n):
    Wo2 = Wo.copy()
    Wo2[:, n] = 0
    return W(Wi, Wh, Wo2)


def zero_feed(n):
    Wh2 = Wh.copy()
    Wh2[:, n] = 0
    Wh2[n, n] = Wh[n, n]  # it still holds itself; only what it sends to the others is cut
    return W(Wi, Wh2, Wo)


silenced = acc([zero_row(n) for n in range(16)])
no_vote = acc([zero_vote(n) for n in range(16)])
no_feed = acc([zero_feed(n) for n in range(16)])

on = (pre > 0).double().mean((0, 1)).cpu().numpy()
rms = h.pow(2).mean((0, 1)).sqrt().cpu().numpy()
end = h[:, -1].mean(0).cpu().numpy()
imp = th.relu(trace(w, th.eye(10, dtype=th.float64, device=DEV))[1][:, -1]).cpu().numpy()  # (position, neuron)


def name(j):
    return "input" if j == "in" else f"n{j}"


print("\nper neuron: on-rate, rms activity, mean at the end; accuracy silenced / vote cut / feed cut")
for n in range(16):
    tag = "core" if n in CORE else "timer"
    print(f"n{n:2d} {tag:5s} on {on[n]:.2f}  rms {rms[n]:5.1f}  end {end[n]:5.1f}  "
          f"silenced {silenced[n]:.4f}  vote cut {no_vote[n]:.4f}  feed cut {no_feed[n]:.4f}")

print("\nper neuron: strongest incoming links by push, then by accuracy lost when cut (self-links included)")
for n in range(16):
    src = [("in", push_in[n], base - in_cut[n], Wi[n])] + [(j, push[n, j], base - link_acc[n, j], Wh[n, j]) for j in range(16)]
    by_push = sorted(src, key=lambda s: -s[1])[:4]
    by_cut = sorted(src, key=lambda s: -s[2])[:4]
    print(f"n{n:2d} in  push: " + ", ".join(f"{name(j)} {wv:+.2f} ({p:.1f}, cut -{100 * c:.1f})" for j, p, c, wv in by_push))
    print(f"        cut:  " + ", ".join(f"{name(j)} {wv:+.2f} (-{100 * c:.1f})" for j, p, c, wv in by_cut))
for n in range(16):
    dst = [(i, push[i, n], base - link_acc[i, n], Wh[i, n]) for i in range(16) if i != n]
    by_push = sorted(dst, key=lambda s: -s[1])[:4]
    by_cut = sorted(dst, key=lambda s: -s[2])[:4]
    print(f"n{n:2d} out push: " + ", ".join(f"n{i} {wv:+.2f} ({p:.1f}, cut -{100 * c:.1f})" for i, p, c, wv in by_push))
    print(f"        cut:  " + ", ".join(f"n{i} {wv:+.2f} (-{100 * c:.1f})" for i, p, c, wv in by_cut))

flat = [(base - link_acc[i, j], i, j) for i in range(16) for j in range(16)]
flat.sort(reverse=True)
print("\nthe 20 links whose cut costs most: target <- source, weight, points lost, push")
for c, i, j in flat[:20]:
    print(f"  n{i:2d} <- n{j:2d}  {Wh[i, j]:+.3f}  -{100 * c:5.1f}  push {push[i, j]:.1f}")
p_rank = np.argsort(-push.ravel())
c_rank = np.argsort(-(base - link_acc).ravel())
top = lambda r, k: set(r[:k].tolist())  # noqa: E731
print(f"overlap of the top 20 links by push and by cut: {len(top(p_rank, 20) & top(c_rank, 20))} of 20")
from scipy.stats import spearmanr
print(f"Spearman of push and cut over the 256 links: {spearmanr(push.ravel(), (base - link_acc).ravel()).statistic:.2f}")

print("\nreadout: each neuron's vote per unit activity for positions 0-9 (W_oh column)")
for n in range(16):
    print(f"n{n:2d} {Wo[:, n]}")
print("\nimpulse response: what each neuron holds at the end after a lone 1 at position p (columns p = 0..9)")
for n in range(16):
    print(f"n{n:2d} {imp[:, n]}  peak at {int(np.argmax(imp[:, n]))}")

# The runner-up's part of each neuron at the end, per unit of its value, by its position q (exact contributions), and
# what that part votes for q itself: Wo[q, n] * coefS[n, q]. A timer that votes against q and is pushed down by a
# runner-up at q lifts its own veto; one that votes for q and is pushed up adds a vote.
C = contributions(w, x) / x[..., None]                                     # (N, 10 inputs, 16)
cS = C.gather(1, q[:, None, None].expand(-1, 1, 16))[:, 0]                 # (N, 16)
coefS = th.stack([cS[q == t].mean(0) for t in range(10)], 1).cpu().numpy()  # (16, 10 positions)
lift = Wo.T * coefS                                                         # (16, 10): neuron n's vote for q via the runner-up at q
print("\nthe runner-up's part of each neuron at the end, per unit, by its position q = 0..9; then that part's vote for q")
for n in range(16):
    print(f"n{n:2d} part {coefS[n]}\n     vote {lift[n]}")
print("\nper timer: positions where the runner-up pushes it down most, where it votes most against; its mean vote for q")
for n in [i for i in range(16) if i not in CORE]:
    dip, veto = np.argsort(coefS[n])[:2], np.argsort(Wo[:, n])[:2]
    print(f"n{n:2d} dips at q = {sorted(dip.tolist())}  vetoes positions {sorted(veto.tolist())}  "
          f"mean vote for q {lift[n].mean():+5.1f}  largest at q = {int(np.argmax(lift[n]))} ({lift[n].max():+.1f})")
# The clock: the core block of W_hh (n2 and the comparators) turns its state by the angle of its complex eigenvalue pair
# each step. Each neuron reads that rotating mode at the phase of its weights on the block; for the comparators, members
# of the block, the phase is where each sits on the mode. Against it, the circular centre of each impulse bump over 0..7.
blk = [2, 7, 6, 1, 8]
ev, V = np.linalg.eig(Wh[np.ix_(blk, blk)])
k = int(np.argmax(np.angle(ev)))
print(f"\nthe core block's eigenvalues: {', '.join(f'{abs(e):.3f} at {np.degrees(np.angle(e)):+.1f} deg' for e in ev)}")
print("reading phase on the rotating mode, and the circular centre of the impulse bump over positions 0..7")
for n in [7, 6, 1, 8] + [i for i in range(16) if i not in CORE]:
    ph = np.degrees(np.angle(Wh[n, blk] @ V[:, k]))
    centre = (np.degrees(np.angle((imp[:8, n] * np.exp(2j * np.pi * np.arange(8) / 8)).sum())) % 360) / 45
    print(f"n{n:2d} phase {ph:+7.1f} deg  bump centre {centre:4.2f}")
tot = lift.sum(0)
print(f"\nall 16 neurons' vote for the runner-up's own position, per unit, by q: {tot}")
print(f"core six: {lift[CORE].sum(0)}")
print(f"timers:   {lift[[i for i in range(16) if i not in CORE]].sum(0)}")

# (16 targets, 17 sources): accuracy with the link from the input (column 0) or from n0..n15 (columns 1..16) cut
np.save(Path(__file__).resolve().parents[1] / "results" / "experiments" / "neuron_atlas.npy", np.c_[in_cut, link_acc])
