"""Why do the readout rows take their shapes? Deduced from W_hi and W_hh alone, then checked against W_oh.

Part 1, the clock. On the 90 two-impulse inputs (x_p = 2, x_q = 1, the rest 0) the state after both impulses passes
through a few gate patterns D, and on each the recurrence is linear, D W_hh. The angle of each pattern's largest eigenvalue
other than the positive reals is the state's rotation per step; it is compared with the step per position of the readout rows' angles in
W_oh's top singular plane (readout_wheel). Then the core block of W_hh alone, and the angle with each neuron dropped.
Part 2, the decoder. The cross-entropy-optimal linear readout of h_10 on Gaussian inputs, by Newton's method on the
states the recurrence computes (W_oh unused), against the actual W_oh with rows centered (softmax ignores a common shift
of the rows), and the accuracy with it swapped in. All six models.
Part 3, what a readout step does. At that optimum a step eps on W_oh[j, n] costs eps^2 / 2 * H, H = E[p_j (1 - p_j) h_n^2]
to second order. Its ranking of the 160 readout weights against the accuracy drops measured by weight_whatif.
Example model unless stated.

Finding: research/findings/circuit.md
"""
from collections import Counter

import numpy as np
import torch as th
from scipy.stats import spearmanr

from az.model import NAMES, load, trace
from experiments.common import DEV, inputs

np.set_printoptions(precision=1, suppress=True, linewidth=160)
PAIRS = [(a, b) for a in range(10) for b in range(10) if a != b]
CORE = [2, 7, 6, 1, 8]  # running max and the four comparators


def fold(deg):
    """Size of a rotation in [0, 180], blind to the plane's orientation."""
    return np.abs((np.asarray(deg) + 180) % 360 - 180)


def routes_after(w) -> Counter:
    """Gate patterns of the steps after both impulses, over the 90 two-impulse inputs."""
    x = th.zeros(90, 10, dtype=th.float64)
    for i, (a, b) in enumerate(PAIRS):
        x[i, a], x[i, b] = 2.0, 1.0
    on = (trace(w, x)[1] > 0).numpy()
    return Counter(tuple(on[i, t]) for i, (a, b) in enumerate(PAIRS) for t in range(max(a, b) + 1, 10))


def lead_turn(A):
    """Modulus and folded angle of A's largest eigenvalue that turns the state: any but the positive reals, which hold it."""
    z = max((e for e in np.linalg.eigvals(A) if fold(np.degrees(np.angle(e))) > 1), key=abs)
    return abs(z), fold(np.degrees(np.angle(z)))


def weighted_median(v, wt):
    o = np.argsort(v)
    c = np.cumsum(wt[o])
    return v[o][np.searchsorted(c, c[-1] / 2)]


def readout_steps(Wo):
    U, s, _ = np.linalg.svd(Wo)
    ang = np.degrees(np.arctan2(U[:, 1] * s[1], U[:, 0] * s[0]))
    return np.diff(np.degrees(np.unwrap(np.radians(ang)))), ang


def states(w, x):
    Wi, Wh, _ = (th.tensor(a, dtype=x.dtype, device=DEV) for a in w)
    h = th.zeros(len(x), 16, dtype=x.dtype, device=DEV)
    for t in range(10):
        h = th.relu(h @ Wh.T + x[:, t, None] * Wi)
    return h


def hess_grad(H, y, W):
    """Hessian (160, 160) and gradient (10, 16) of the mean cross-entropy of the readout W on states H, labels y."""
    N = len(H)
    P = th.softmax(H @ W.T, 1)
    G = ((P - th.nn.functional.one_hot(y, 10).double()).T @ H) / N
    Z = (P[:, :, None] * H[:, None, :]).reshape(N, 160)
    return (th.block_diag(*th.einsum("ni,nk,nl->ikl", P, H, H)) - Z.T @ Z) / N, G


def decoder(H, y, iters=40):
    """The cross-entropy-optimal readout by Newton's method from zero; the problem is convex, so the optimum is unique up to
    a common shift of the rows."""
    W = th.zeros(10, 16, dtype=th.float64, device=DEV)
    for _ in range(iters):
        Hs, G = hess_grad(H, y, W)
        W = W - th.linalg.lstsq(Hs + 1e-9 * th.eye(160, dtype=th.float64, device=DEV), G.reshape(160, 1)).solution.reshape(10, 16)
    return W


print("Part 1: the clock")
for name in NAMES:
    w = load(name)
    pats = routes_after(w).most_common()
    cnt = np.array([c for _, c in pats])
    mod, ang = np.array([lead_turn(np.diag(np.array(p, float)) @ w.Wh) for p, _ in pats]).T
    med = weighted_median(ang, cnt)
    steps, _ = readout_steps(w.Wo)
    print(f"  {name:8s} {len(pats):2d} patterns over {cnt.sum()} steps; leading turning eigenvalue: step-weighted median angle {med:5.1f},"
          f" {cnt[np.abs(ang - med) < 10].sum() / cnt.sum():.2f} of steps within 10 deg; commonest pattern {mod[0]:.2f} at {ang[0]:.1f} ({cnt[0]} steps);"
          f" readout step per position 0->7, folded: median {np.median(fold(steps[:7])):5.1f}, range {fold(steps[:7]).min():.0f}-{fold(steps[:7]).max():.0f}")

w = load("example")
steps, ang = readout_steps(w.Wo)
pats = routes_after(w)
top = np.array(pats.most_common(1)[0][0], float)
print(f"example readout angles by position {ang.round(0)}; steps 0->7 {steps[:7].round(0)}, mean {fold(steps[:7]).mean():.1f};"
      f" the closing step 7->0 {fold(ang[0] - ang[7]):.0f}; 8 steps sum to {fold(steps[:7]).sum() + fold(ang[0] - ang[7]):.0f}")
x = th.zeros(90, 10, dtype=th.float64)
for i, (a, b) in enumerate(PAIRS):
    x[i, a], x[i, b] = 2.0, 1.0
on = trace(w, x)[1].numpy() > 0
settled = on[:, :, CORE].all(2) & ~on[:, :, 4]  # running max and comparators on, increment off
print("  share of two-impulse inputs whose core gates are settled (n2 n7 n6 n1 n8 on, n4 off) k steps after the later impulse, k = 0..3: "
      + " ".join(f"{np.mean([settled[i, max(a, b) + k] for i, (a, b) in enumerate(PAIRS) if max(a, b) + k < 10]):.2f}" for k in range(4)))
print("  core block (n2 n7 n6 n1 n8) of W_hh alone, eigenvalues: "
      + ", ".join(f"{abs(z):.2f} at {np.degrees(np.angle(z)):+.1f}" for z in sorted(np.linalg.eigvals(w.Wh[np.ix_(CORE, CORE)]), key=lambda z: -abs(z))))
print("  commonest pattern with one neuron's gate shut: leading pair nearest 45 deg, modulus at angle")
for n in np.nonzero(top)[0]:
    D = top.copy()
    D[n] = 0
    z = min((e for e in np.linalg.eigvals(np.diag(D) @ w.Wh) if e.imag > 1e-9), key=lambda e: abs(np.degrees(np.angle(e)) - 45))
    print(f"    n{n:<2d} {'core' if n in CORE else 'sat '}: {abs(z):.2f} at {np.degrees(np.angle(z)):.1f}")

print("\nPart 2: the decoder, the cross-entropy-optimal readout of h_10 (2^20 Gaussian inputs to fit, 2^20 others to score)")
for name in NAMES:
    w = load(name)
    x, xt = inputs(2**20, seed=41), inputs(2**20, seed=42)
    y, yt = x.argsort(1)[:, -2], xt.argsort(1)[:, -2]
    H, Ht = states(w, x), states(w, xt)
    Wd = decoder(H, y)
    Wo = th.tensor(w.Wo, device=DEV)
    A, B = Wo - Wo.mean(0), Wd - Wd.mean(0)
    cos = (A * B).sum(1) / (A.norm(dim=1) * B.norm(dim=1))
    acc = lambda W: ((Ht @ W.T).argmax(1) == yt).double().mean().item()  # noqa: E731
    ce = lambda W: th.nn.functional.cross_entropy(Ht @ W.T, yt).item()  # noqa: E731
    print(f"  {name:8s} per-row cosine min {cos.min():.3f} median {cos.median():.3f}; R2 {1 - ((A - B) ** 2).sum() / (A ** 2).sum():.3f};"
          f" norm derived / actual {B.norm() / A.norm():.2f}; accuracy actual {acc(Wo):.4f} derived {acc(Wd):.4f};"
          f" cross-entropy actual {ce(Wo):.4f} derived {ce(Wd):.4f}")
    if name == "example":
        Hs, _ = hess_grad(H, y, Wd)

print("\nPart 3: a readout step's cost, eps^2/2 E[p_j (1 - p_j) h_n^2] at the derived optimum, against the measured accuracy drops")
drop = 0.9533 - np.load("results/experiments/weight_whatif.npy")[272:]  # W_oh rows; steps -0.5 -0.1 -0.02 0.02 0.1 0.5
d = Hs.diagonal().cpu().numpy()
for k, eps in [(0, -0.5), (5, 0.5), (1, -0.1), (4, 0.1)]:
    print(f"  step {eps:+.1f}: Spearman over the 160 readout weights {spearmanr(d, drop[:, k]).statistic:.3f}")
print(f"  drops at +0.5 against -0.5, correlation {np.corrcoef(drop[:, 0], drop[:, 5])[0, 1]:.2f};"
      f" median drop at 0.5 over drop at 0.1 {np.median((drop[:, 0] + drop[:, 5]) / (drop[:, 1] + drop[:, 4])):.1f} (quadratic: 25)")
cols = np.bincount(np.argsort(-d)[:20] % 16, minlength=16)
print("  neurons of the 20 costliest readout weights: " +", ".join(f"n{n} x{c}" for n, c in enumerate(cols) if c))
