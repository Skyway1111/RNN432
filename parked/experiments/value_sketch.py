"""Is the value of a partial input, P(correct | the first t inputs), low-dimensional enough to sketch?

A backward deduction-projection estimator would carry V_t(h, running top two) from t = 10 down to 0. V_9 is exact and
cheap: given the state before the last input, each pre-activation is affine in x_9, so the logits are affine between at
most 16 kinks and the order breakpoints S and M, and the correct set is an interval per segment with mass from Phi.
V_8 integrates V_9 over x_8 by 192-node quadrature split at S and M. Checks and sketches:
  1. V_9 against a 20,000-node grid over x_9 on 4,096 prefixes.
  2. Sketches of V_9 (2^17 prefixes) and V_8 (2^14), fitted per key (positions of the running top two) on half the
     prefixes by least squares and scored on the other half: a constant; linear in the state's leading 1, 4 or 16
     principal components plus M and S; quadratic in 4 components plus M and S; at t = 9, quadratic in the ten current
     logits plus M and S. Fitting on the true prefix law is kinder than any deterministic design.
Two models.

Finding: research/angles/value-sketch.md
"""
import numpy as np
import torch as th

from az.bench import suite
from experiments.common import DEV

th.set_grad_enabled(False)
labels, ws = suite(0.002, 32)
DT = th.float64


def prefix(w, X):
    """State after the inputs X (N, t), the running top-two values and positions."""
    Wi, Wh, _ = (th.tensor(a, dtype=DT, device=DEV) for a in w)
    h = th.zeros(len(X), 16, dtype=DT, device=DEV)
    for t in range(X.shape[1]):
        h = th.relu(h @ Wh.T + X[:, t, None] * Wi)
    top = X.topk(2, 1)
    return h, top.values[:, 0], top.values[:, 1], top.indices[:, 0], top.indices[:, 1]


def v_last(w, h, M, S, p, q, t_last=9):
    """Exact P(correct) over the last input x ~ N(0, 1), given the state h before it and the running top two."""
    Wi, Wh, Wo = (th.tensor(a, dtype=DT, device=DEV) for a in w)
    z0 = h @ Wh.T
    bp = th.cat([-z0 / Wi, M[:, None], S[:, None]], 1).sort(1).values
    lo = th.cat([th.full_like(bp[:, :1], -np.inf), bp], 1)
    hi = th.cat([bp, th.full_like(bp[:, :1], np.inf)], 1)
    mid = th.where(th.isinf(lo), hi - 1, th.where(th.isinf(hi), lo + 1, (lo + hi) / 2))
    act = ((z0[:, None, :] + mid[..., None] * Wi) > 0).to(DT)
    a = th.einsum("oj,nsj->nso", Wo, act * z0[:, None, :])  # logits = a + b x on each segment
    b = th.einsum("oj,nsj->nso", Wo, act * Wi)
    lab = th.where(mid > M[:, None], p[:, None], th.where(mid > S[:, None], th.full_like(p[:, None], t_last), q[:, None]))
    da, db = a - a.gather(2, lab[..., None]), b - b.gather(2, lab[..., None])
    other = th.ones_like(da, dtype=th.bool).scatter_(2, lab[..., None], False)
    r = -da / db
    up = th.where(other & (db > 0), r, th.full_like(r, np.inf)).min(2).values
    dn = th.where(other & (db < 0), r, th.full_like(r, -np.inf)).max(2).values
    ok = ~(other & (db == 0) & (da >= 0)).any(2)
    U, L = th.minimum(hi, up), th.maximum(lo, dn)
    return ((th.special.ndtr(U) - th.special.ndtr(L)).clamp(min=0) * ok).sum(1)


def v_8(w, h, M, S, p, q, nq=64):
    """V_8 by quadrature over x_8 on three pieces split at S and M, nq equal-mass nodes each."""
    Wi, Wh, _ = (th.tensor(a, dtype=DT, device=DEV) for a in w)
    out = th.zeros(len(h), dtype=DT, device=DEV)
    u = (th.arange(nq, device=DEV, dtype=DT) + 0.5) / nq
    one = th.ones(len(h), dtype=DT, device=DEV)
    for lo, hi in [(None, S), (S, M), (M, None)]:
        a = 0 * one if lo is None else th.special.ndtr(lo)
        b = one if hi is None else th.special.ndtr(hi)
        x = th.special.ndtri((a[:, None] + (b - a)[:, None] * u).clamp(1e-300, 1 - 1e-16)).reshape(-1)
        h9 = th.relu((h @ Wh.T)[:, None] + x.reshape(len(h), nq)[..., None] * Wi).reshape(-1, 16)
        Mr, Sr, pr, qr = (v.repeat_interleave(nq) for v in (M, S, p, q))
        new_max, new_2nd = x > Mr, (x > Sr) & (x <= Mr)
        M9, S9 = th.maximum(Mr, x), th.where(new_max, Mr, th.where(new_2nd, x, Sr))
        p9 = th.where(new_max, th.full_like(pr, 8), pr)
        q9 = th.where(new_max, pr, th.where(new_2nd, th.full_like(qr, 8), qr))
        out += (v_last(w, h9, M9, S9, p9, q9).reshape(len(h), nq) * (b - a)[:, None] / nq).sum(1)
    return out


def score(V, F, key):
    """Held-out rms and mean error of a per-key least-squares sketch on features F (N, k)."""
    tr = th.arange(len(V), device=DEV) % 2 == 0
    pred = th.zeros_like(V)
    for k in key.unique():
        a, b = (key == k) & tr, (key == k) & ~tr
        if not b.any():
            continue
        if a.sum() < 2 * F.shape[1]:
            pred[b] = V[a].mean() if a.any() else V[tr].mean()
            continue
        coef = th.linalg.solve(F[a].T @ F[a] + 1e-6 * a.sum() * th.eye(F.shape[1], dtype=DT, device=DEV), F[a].T @ V[a])
        pred[b] = (F[b] @ coef).clamp(0, 1)
    r = (pred - V)[~tr]
    return f"{r.pow(2).mean().sqrt():.3f} ({r.mean():+.4f})"


def quad(B):
    return th.cat([th.ones_like(B[:, :1]), B] + [B[:, i : i + 1] * B[:, i:] for i in range(B.shape[1])], 1)


print("1. exact V_9 against a 20,000-node grid over x_9")
for i in (32, 0):
    w = ws[i]
    g = th.Generator(device=DEV).manual_seed(1)
    X = th.randn(4096, 9, generator=g, device=DEV, dtype=DT)
    h, M, S, p, q = prefix(w, X)
    v = v_last(w, h, M, S, p, q)
    xs = th.linspace(-8, 8, 20001, device=DEV, dtype=DT)
    wx, xm = th.special.ndtr(xs[1:]) - th.special.ndtr(xs[:-1]), (xs[1:] + xs[:-1]) / 2
    Wi, Wh, Wo = (th.tensor(a, dtype=DT, device=DEV) for a in w)
    vb = []
    for c in range(0, 4096, 256):
        sl = slice(c, c + 256)
        pred = (th.relu((h[sl] @ Wh.T)[:, None] + xm[None, :, None] * Wi) @ Wo.T).argmax(2)
        lab = th.where(xm > M[sl, None], p[sl, None], th.where(xm > S[sl, None], th.full_like(p[sl, None], 9), q[sl, None]))
        vb.append(((pred == lab).to(DT) * wx).sum(1))
    print(f"  {labels[i]:10s} max |exact - grid| {(v - th.cat(vb)).abs().max():.1e}, mean V_9 {v.mean():.4f}")

print("\n2. sketches, held-out rms (mean error) in probability units")
g = th.Generator(device=DEV).manual_seed(5)
for i in (32, 0):
    w = ws[i]
    for t, n in ((9, 2**17), (8, 2**14)):
        h, M, S, p, q = prefix(w, th.randn(n, t, generator=g, device=DEV, dtype=DT))
        V = v_last(w, h, M, S, p, q) if t == 9 else th.cat([v_8(w, *(a[c : c + 1024] for a in (h, M, S, p, q))) for c in range(0, n, 1024)])
        key = p * 10 + q
        tr = th.arange(n, device=DEV) % 2 == 0
        ev, vec = th.linalg.eigh(th.cov(h[tr].T))
        z = (h - h[tr].mean(0)) @ vec[:, ev.argsort(descending=True)]
        MS = th.stack([M, S], 1)
        row = {"constant": score(V, th.ones_like(V[:, None]), key)}
        for k in (1, 4, 16):
            row[f"linear PC1-{k}"] = score(V, th.cat([th.ones_like(V[:, None]), z[:, :k], MS], 1), key)
        row["quadratic PC1-4"] = score(V, quad(th.cat([z[:, :4], MS], 1)), key)
        if t == 9:
            row["quadratic logits"] = score(V, quad(th.cat([h @ th.tensor(w.Wo, dtype=DT, device=DEV).T, MS], 1)), key)
        print(f"  {labels[i]:10s} t = {t}: V mean {V.mean():.3f}, sd {V.std():.3f}; " + "; ".join(f"{k} {v}" for k, v in row.items()), flush=True)
