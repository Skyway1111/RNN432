"""Do exact line integrals or a cheap linear surrogate of the margin pay on the rare levels?

Part 1, the S-line: hold everything but the 2nd largest value S and move it over (T, M), where the label stays put and the
event is the union of the two tie wedges. P(event | the rest) by 256-point quantile quadrature of S's law, on 2^16 rests
from the nominal law, held-out copy /0 of each base. Per level: the per-line gain, sampling's relative variance per
sample over the conditional's, and the share of lines with any event. An exact S-line costs four to seven passes
(az/estimators/sline.py), so a line pays only above that.
Part 2, screening by a surrogate: the reference route per (p, q), the most common gate pattern among 2^21 inputs of the
clean base with that order, gives a linear map whose margin f~ costs a tenth of a pass. Its correlation with the margin f
on the copy, the share of inputs where it is exact, and at c = 2 and 8 the screening gain 1 / (abar E[1/a | event]) of
accepting with a = 1 when f~ >= c - 2 and a = 0.02 otherwise (abar: acceptance rate).

Finding: research/findings/tail.md
"""
import torch as th

from az.estimators.modes import margin
from az.model import NAMES, load
from az.rare import LEVELS, models

th.set_grad_enabled(False)
labels, ws = models("heldout")
DEV = "cuda"
Phi, iPhi = th.special.ndtr, th.special.ndtri
NQ = 256

print("Part 1: S-line per-line gain per level / share of lines with any event")
for i in range(0, 192, 32):
    W = [th.tensor(a, dtype=th.float32, device=DEV) for a in ws[i]]
    g = th.Generator(device=DEV).manual_seed(3)
    cond = []
    for _ in range(8):
        x = th.randn(2**13, 10, generator=g, device=DEV, dtype=th.float64)
        v, idx = x.sort(1, descending=True)
        q = idx[:, 1]
        a, b = Phi(v[:, 2]), Phi(v[:, 0])
        S = iPhi(a[:, None] + (b - a)[:, None] * (th.arange(NQ, device=DEV) + 0.5) / NQ)
        X = x[:, None, :].repeat(1, NQ, 1)
        X.scatter_(2, q[:, None, None].expand(-1, NQ, 1), S[..., None])
        X = X.reshape(-1, 10)
        f = margin(W, X / X.norm(dim=1, keepdim=True))
        cond.append(th.stack([(f >= c).double() for c in LEVELS], 1).reshape(len(x), NQ, -1).mean(1))
    c = th.cat(cond)
    m, var = c.mean(0), c.var(0)
    print(f"  {labels[i]:10s} " + " ".join(f"{gg:5.1f}/{nz:.4f}" for gg, nz in zip((((1 - m) / m) / (var / m**2)).tolist(), (c > 0).double().mean(0).tolist())), flush=True)

print("Part 2: reference-route surrogate: corr(f~, f), exact share; screening gain at c = 2, 8")
for b, name in enumerate(NAMES):
    W0 = [th.tensor(a, dtype=th.float64, device=DEV) for a in load(name)]
    x = th.randn(2**21, 10, generator=th.Generator(device=DEV).manual_seed(2), device=DEV, dtype=th.float64)
    h = th.zeros(len(x), 16, device=DEV, dtype=th.float64)
    pats = []
    for t in range(10):
        zt = h @ W0[1].T + x[:, t, None] * W0[0]
        pats.append(zt > 0)
        h = th.relu(zt)
    bits = th.stack(pats, 1).reshape(len(x), -1).to(th.uint8)
    top = x.topk(2, dim=1).indices
    pq = top[:, 0] * 10 + top[:, 1]
    ref = th.zeros(100, 10, 16, dtype=th.bool, device=DEV)
    for k in range(100):
        if k // 10 != k % 10:
            uk, cnt = th.unique(bits[pq == k], dim=0, return_counts=True)
            ref[k] = uk[cnt.argmax()].bool().reshape(10, 16)
    Wi, Wh, Wo = (th.tensor(a, dtype=th.float64, device=DEV) for a in ws[32 * b])
    A = []
    for k in range(100):
        H = th.zeros(10, 16, dtype=th.float64, device=DEV)
        for t in range(10):
            H = H @ Wh.T
            H[t] += Wi
            H = H * ref[k, t]
        A.append((H @ Wo.T).T)
    A = th.stack(A)
    W = [th.tensor(a, dtype=th.float32, device=DEV) for a in ws[32 * b]]
    g = th.Generator(device=DEV).manual_seed(3)
    fs, ft = [], []
    for _ in range(16):
        z = th.randn(2**22, 10, generator=g, device=DEV, dtype=th.float64)
        u = z / z.norm(dim=1, keepdim=True)
        top = u.topk(2, dim=1).indices
        L = th.einsum("nij,nj->ni", A[top[:, 0] * 10 + top[:, 1]], u)
        qq = top[:, 1:]
        ft.append(L.scatter(1, qq, -th.inf).max(1).values - L.gather(1, qq)[:, 0])
        fs.append(margin(W, u))
    f, fsur = th.cat(fs), th.cat(ft)
    row = []
    for c in (2.0, 8.0):
        a = th.where(fsur >= c - 2, 1.0, 0.02)
        ev = f >= c
        row.append(f"c={c:g} gain {1 / (a.mean() * (1 / a[ev]).mean()):.2f}")
    print(f"  {labels[32 * b]:10s} corr {th.corrcoef(th.stack([fsur[:2**22], f[:2**22]]))[0, 1]:.2f}, exact {((fsur - f).abs() < 1e-4).double().mean():.3f}; " + "; ".join(row), flush=True)
