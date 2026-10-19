"""Does sequential Monte Carlo over levels, which needs no proposal density, beat importance sampling on the tail?

Subset simulation: N particles; at each level of the ladder 0, 1, ..., 8 the share that clears the next level estimates
the conditional probability, the survivors are resampled, and m moves that keep the target refresh them. Generic: pCN
moves on all ten inputs, accepted when the input stays in the event. Line-collapsed: a particle is the other nine inputs
with S's slot fixed, its target the S-line's event mass (az/estimators/sline.level_mass), so the thin direction across a
tie is integrated exactly and every ratio is Rao-Blackwellized; pCN moves on the nine, accepted with the ratio of line
masses. 16 runs per setting on three held-out copies; ops count every forward pass and every line's multiply-adds.

Finding: research/angles/level-smc.md
"""
import torch as th

from az.estimators import modes
from az.estimators.sline import level_mass
from az.rare import LEVELS, TAIL_OPS, models, truth

th.set_grad_enabled(False)
DEV = "cuda"
labels, ws = models("heldout")
P = truth("heldout")
LADDER = th.arange(9, dtype=th.float64, device=DEV)


def margin(W, x):
    return modes.margin(W, (x / x.norm(dim=-1, keepdim=True)).reshape(-1, 10)).reshape(x.shape[:-1])


def generic(W, R, N, moves, gen):
    """R independent runs of N particles; estimates (R, levels) and ops per run."""
    x = th.randn(R, N, 10, generator=gen, device=DEV, dtype=th.float64)
    f, ops, logp, est = margin(W, x), N * TAIL_OPS, th.zeros(R, dtype=th.float64, device=DEV), {}
    rho = th.full((R, 1, 1), 0.8, dtype=th.float64, device=DEV)
    for c in LADDER.tolist():
        ok = f >= c
        frac = ok.double().mean(1)
        logp = logp + frac.clamp(min=1e-300).log()
        est[c] = logp.exp() * (frac > 0)
        if c == LADDER[-1]:
            break
        idx = th.multinomial(ok.double() + 1e-300, N, replacement=True, generator=gen)
        x, f = x.gather(1, idx[..., None].expand(-1, -1, 10)), f.gather(1, idx)
        for _ in range(moves):
            xp = rho * x + (1 - rho**2).sqrt() * th.randn(x.shape, generator=gen, device=DEV, dtype=th.float64)
            fp = margin(W, xp)
            acc = fp >= c
            x, f = th.where(acc[..., None], xp, x), th.where(acc, fp, f)
            rho = (rho + 0.1 * (acc.double().mean(1, keepdim=True)[..., None] - 0.3)).clamp(0.05, 0.995)
            ops += N * TAIL_OPS
    return th.stack([est[c] for c in LEVELS], 1), ops


def lines(W, X, j):
    rest = X.scatter(1, j[:, None], -th.inf).sort(1, descending=True).values
    out = [level_mass(*W, X[a:a + 2**15], j[a:a + 2**15], rest[a:a + 2**15, 1], rest[a:a + 2**15, 0], LADDER) for a in range(0, len(X), 2**15)]
    return th.cat([m for m, _ in out]), sum(o.sum().item() for _, o in out), rest


def collapsed(W, N, moves, gen, rho=0.9):
    """One run: estimates per level and ops."""
    X = th.randn(N, 10, generator=gen, device=DEV, dtype=th.float64)
    j = X.topk(2, dim=1).indices[:, 1]
    L, ops, rest = lines(W, X, j)
    norm = th.special.ndtr(rest[:, 0]) - th.special.ndtr(rest[:, 1])
    est = {0.0: (L[:, 0] / norm).mean().item()}
    w = L[:, 1] / norm  # the start is nominal: weights to the first target
    logZ = th.log(w.mean())
    for k in range(1, len(LADDER)):
        if k > 1:
            w = L[:, k] / L[:, k - 1].clamp(min=1e-300) * (L[:, k - 1] > 0)
            logZ = logZ + th.log(w.mean())
        if LADDER[k].item() in LEVELS:
            est[LADDER[k].item()] = logZ.exp().item()
        if k == len(LADDER) - 1:
            break
        idx = th.multinomial(w, N, replacement=True, generator=gen)
        X, j, L = X[idx], j[idx], L[idx]
        acc = 0.0
        for _ in range(moves):
            Xp = rho * X + (1 - rho**2) ** 0.5 * th.randn(X.shape, generator=gen, device=DEV, dtype=th.float64)
            Xp = Xp.scatter(1, j[:, None], X.gather(1, j[:, None]))
            Lp, o, _ = lines(W, Xp, j)
            ops += o
            ok = th.rand(N, generator=gen, device=DEV, dtype=th.float64) < (Lp[:, k] / L[:, k].clamp(min=1e-300)).clamp(max=1.0)
            X, L = th.where(ok[:, None], Xp, X), th.where(ok[:, None], Lp, L)
            acc += ok.double().mean().item() / moves
        rho = min(0.995, max(0.3, rho + 0.15 * (acc - 0.35)))
    return [est[c] for c in LEVELS], ops


def report(lab, name, e, ops, p):
    rel = (e - p) / p
    rmse = rel.pow(2).mean(0).sqrt()
    adv = ((1 - p) / p / 0.01 * TAIL_OPS) / (ops * rmse.pow(2) / 0.01)  # sampling's ops to 10% over these, scaling 1/ops
    print(f"  {lab:9s} {name:26s} passes {ops / TAIL_OPS:7.0f}; rel RMSE " + " ".join(f"{v:.3f}" for v in rmse.tolist())
          + "; bias " + " ".join(f"{v:+.2f}" for v in rel.mean(0).tolist()) + "; advantage " + " ".join(f"{v:.2f}" for v in adv.tolist()), flush=True)


print("c = 0, 2, 4, 6, 8; 16 runs each; advantage: sampling's ops to 10% relative RMSE over these (modes on the bench: 0.35, 1.64, 7.3, 34, 92)")
for lab in ("s2/15", "s1/12", "example/0"):
    i = labels.index(lab)
    p = th.tensor(P[i], device=DEV)
    W32 = [th.tensor(a, dtype=th.float32, device=DEV) for a in ws[i]]
    W64 = [th.tensor(a, dtype=th.float64, device=DEV) for a in ws[i]]
    e, ops = generic(W32, 16, 4000, 10, th.Generator(device=DEV).manual_seed(1))
    report(lab, "generic, N 4000, 10 moves", e, ops, p)
    runs = [collapsed(W64, 2000, 3, th.Generator(device=DEV).manual_seed(100 + r)) for r in range(16)]
    report(lab, "line-collapsed, N 2000, 3", th.tensor([r[0] for r in runs], device=DEV), sum(r[1] for r in runs) / 16, p)
