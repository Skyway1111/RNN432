"""Inside a (family, T, M) cell, which features of S and the seven lower inputs carry the indicator's variance?

The family-stratified lattice leaves 59-70% of the indicator's variance inside its (family, T, M) cells
(parked/research/angles/family-lattice.md); S and the lower inputs decide it through gate flips. A presumption keyed on a
few features of them could replace a spread point set only if those features hold most of that variance. For 128
configurations (family, T, M) from the exact laws and 32,768 draws of (S, lower inputs) each, three suite models:
  1. The within-cell share of the indicator's variance explained by nested feature sets, split-half (cell means fitted
     on the odd draws, scored on the even ones, so a key with many cells is not rewarded for memorizing): the S bin;
     the lower inputs' sign count and sign pattern; their record history (which free slots were a running max or 2nd
     max when read); the largest lower input's slot and value bin; the 2nd largest's; and the gate pattern itself, at
     the lower-input steps and at all steps, the most any feature of the route can explain. The same at the tie
     S = T, for the indicator and for the tie margin m_T = l_q - max other, over the lower inputs alone.
  2. The ceiling of a presumption keyed on those features: P(m_T < 0) per configuration from a Gaussian residual with
     oracle moments inside each feature cell (none; the largest lower input; the two largest), bias and per-config rms.
  3. Which gates vary across the draws at the tie, by step and by the kind of input read at that step, and how much
     of a variable gate's pre-activation is a threshold on that step's own input (R^2 on it alone, and on all seven
     lower inputs linearly).

Finding: research/findings/errors.md
"""
import numpy as np
import torch as th
from scipy.stats import norm

from az.bench import suite
from az.estimators.gauss_margin import FAMS, FREE
from az.estimators.lattice import _cdf_t
from experiments.common import DEV

labels, ws = suite(0.002, 32)
rng = np.random.default_rng(3)
th.manual_seed(3)
t_grid, cdf_t = _cdf_t()
C, D = 128, 32768
POW3 = th.tensor([3**i for i in range(10)], device=DEV)
HASH = th.tensor(np.random.default_rng(0).integers(1, 2**40, 160) * 2 + 1, device=DEV)


def configs(n):
    fam = rng.integers(720, size=n)
    T = np.interp(rng.uniform(size=n), cdf_t, t_grid)
    M = norm.ppf(norm.cdf(T) + (1 - norm.cdf(T)) * np.sqrt(rng.uniform(size=n)))
    return fam, T, M


def draw(fam, T, M, tie: bool):
    """Inputs (C D, 10) with the top three placed and S, lower inputs drawn from their exact laws (S = T if tie)."""
    T_rep = th.tensor(np.repeat(T, D), device=DEV)
    M_rep = th.tensor(np.repeat(M, D), device=DEV)
    u = th.rand(C * D, 8, device=DEV, dtype=th.float64)
    z = th.special.ndtri(th.special.ndtr(T_rep)[:, None] * u[:, :7])
    S = T_rep + 1e-9 if tie else th.special.ndtri(th.special.ndtr(T_rep) + (th.special.ndtr(M_rep) - th.special.ndtr(T_rep)) * u[:, 7])
    X = th.zeros(C * D, 10, dtype=th.float64, device=DEV)
    free = th.tensor(FREE, device=DEV)[th.tensor(np.repeat(fam, D), device=DEV)]
    X.scatter_(1, free, z)
    ar = th.arange(C * D, device=DEV)
    pqr = th.tensor(np.array([FAMS[f] for f in np.repeat(fam, D)]), device=DEV)
    X[ar, pqr[:, 0]], X[ar, pqr[:, 1]], X[ar, pqr[:, 2]] = M_rep, S, T_rep
    return X, z, free, pqr, u


def gates(w, x, keep_pre: int = 0):
    """Gate bits (N, 10, 16), logits (N, 10), and the pre-activations of the first keep_pre inputs."""
    Wi, Wh, Wo = (th.tensor(a, device=DEV) for a in w)
    P, Z, L = [], [], []
    for c in range(0, len(x), 2**20):
        h = th.zeros(len(x[c : c + 2**20]), 16, dtype=th.float64, device=DEV)
        Pc, Zc = [], []
        for t in range(10):
            z = h @ Wh.T + x[c : c + 2**20, t, None] * Wi
            Pc.append(z > 0)
            if c < keep_pre:
                Zc.append(z[: keep_pre - c])
            h = th.relu(z)
        P.append(th.stack(Pc, 1))
        L.append(h @ Wo.T)
        if Zc:
            Z.append(th.stack(Zc, 1))
    return th.cat(P), th.cat(L), (th.cat(Z) if Z else None)


def history_free(X, free):
    """Record history over the free slots: per free slot, 2 if a new running max when read, 1 if a new 2nd max, else 0."""
    Mx = th.full((len(X),), -1e9, dtype=th.float64, device=DEV)
    Sx = Mx.clone()
    code = th.zeros(len(X), dtype=th.int64, device=DEV)
    is_free = th.zeros(len(X), 10, dtype=th.bool, device=DEV).scatter_(1, free, True)
    for t in range(10):
        newmax = X[:, t] > Mx
        new2 = ~newmax & (X[:, t] > Sx)
        code += th.where(is_free[:, t], newmax.long() * 2 + new2.long(), 0) * POW3[t]
        Sx = th.where(newmax, Mx, th.where(new2, X[:, t], Sx))
        Mx = th.where(newmax, X[:, t], Mx)
    return code


def ladder(y, keys, names):
    """Split-half share of the within-config variance of y (C D,) explained by each key (C D,) int64, with cells and
    draws per cell."""
    cfg = th.arange(C, device=DEV).repeat_interleave(D)
    odd = (th.arange(C * D, device=DEV) % 2).bool()
    base = y.reshape(C, D)[:, 1::2].mean(1, keepdim=True).expand(-1, D // 2).reshape(-1)
    total = ((y.reshape(C, D)[:, ::2].reshape(-1) - base) ** 2).mean()
    rows = []
    for key, name in zip(keys, names):
        full = th.unique(cfg * (2**44) + key % (2**44), return_inverse=True)[1]
        cnt = th.zeros(full.max().item() + 1, dtype=th.float64, device=DEV).index_add_(0, full[odd], th.ones(int(odd.sum()), dtype=th.float64, device=DEV))
        s = th.zeros_like(cnt).index_add_(0, full[odd], y[odd])
        mean = th.where(cnt > 0, s / cnt.clamp(min=1), th.nan)
        pred = mean[full[~odd]]
        pred = th.where(pred.isnan(), base, pred)
        resid = ((y[~odd] - pred) ** 2).mean()
        rows.append(f"    {name:48s} cells per config {len(cnt) / C:8.1f} ({D / 2 / (len(cnt) / C):6.1f} draws each)   explained {1 - resid / total:.3f}")
    return rows


def keys_of(z, free, X, P, pqr, u, tie):
    ar = th.arange(C * D, device=DEV)
    Lv, Ls = z.max(1)
    z2 = z.clone()
    z2[ar, Ls] = -th.inf
    Lv2, Ls2 = z2.max(1)
    T_rep = X[ar, pqr[:, 2]]
    signs = ((z > 0).long() * th.tensor([2**k for k in range(7)], device=DEV)).sum(1)
    npos = (z > 0).sum(1)
    rec = history_free(X, free)
    b1 = (th.special.ndtr(Lv) / th.special.ndtr(T_rep)).pow(7).mul(8).long().clamp(max=7)
    b2 = (th.special.ndtr(Lv2) / th.special.ndtr(Lv)).pow(6).mul(4).long().clamp(max=3)
    l1, l2 = Ls * 8 + b1, Ls2 * 4 + b2
    is_free = th.zeros(C * D, 10, dtype=th.bool, device=DEV).scatter_(1, free, True)
    Pl = P.reshape(C * D, 160).long()
    pat_all = (Pl * HASH).sum(1)
    pat_low = ((Pl * HASH).reshape(C * D, 10, 16) * is_free[:, :, None]).sum((1, 2))
    k0 = (u[:, 7] * 16).long() if not tie else th.zeros_like(npos)
    names = ["S bin (16)"] if not tie else ["nothing beyond the configuration"]
    keys = [k0]
    for k, name in ((npos, "+ count of positive lower inputs"), (signs, "+ sign pattern of the lower inputs"), (rec, "+ record history of the free slots"),
                    (l1, "+ largest lower: slot, 8 value bins"), (l2, "+ 2nd largest lower: slot, 4 value bins")):
        keys.append(keys[-1] * 59049 + k % 59049)
        names.append(name)
    keys += [k0 * 56 + l1, (k0 * 56 + l1) * 28 + l2, k0 * (2**40) + pat_low % (2**40), k0 * (2**40) + pat_all % (2**40)]
    names += ["largest lower only (with the S bin)", "two largest lower only (with the S bin)", "gate pattern at the lower-input steps", "gate pattern at all steps"]
    return keys, names


for i in (32, 0, 160):
    w = ws[i]
    fam, T, M = configs(C)
    print(f"{labels[i]}")
    for tie in (False, True):
        X, z, free, pqr, u = draw(fam, T, M, tie)
        with th.no_grad():
            P, L, Z = gates(w, X, keep_pre=16 * D if tie else 0)
        ar = th.arange(C * D, device=DEV)
        q_ = pqr[:, 1]
        m = L[ar, q_] - L.scatter(1, q_[:, None], -1e9).max(1).values
        ok = (m > 0).double()
        keys, names = keys_of(z, free, X, P, pqr, u, tie)
        if not tie:
            within = ok.reshape(C, D).var(1, correction=0).mean()
            pw = 1 - ok.mean()
            print(f"  S and the lower inputs drawn: P(wrong) {pw:.4f}; within-config variance {within:.4f}, share of p(1-p) {within / (pw * (1 - pw)):.3f}")
            print("\n".join(ladder(ok, keys, names)))
        else:
            print(f"  at the tie S = T: P(wrong) {(m < 0).double().mean():.4f}; m_T within-config sd {m.reshape(C, D).std(1).mean():.2f}, mean {m.mean():+.2f}")
            print("   indicator at the tie:")
            print("\n".join(ladder(ok, keys[1:], names[1:])))
            print("   tie margin m_T:")
            print("\n".join(ladder(m, keys[1:], names[1:])))
            cfg = th.arange(C, device=DEV).repeat_interleave(D)
            p_emp = (m < 0).double().reshape(C, D).mean(1)
            print("   P(m_T < 0) per configuration from a Gaussian residual with oracle moments inside each feature cell:")
            for key, name in ((th.zeros_like(cfg), "no feature"), (keys[6], "largest lower: slot, 8 value bins"), (keys[7], "two largest lower: slots, bins")):
                full = th.unique(cfg * (2**44) + key % (2**44), return_inverse=True)[1]
                cnt = th.zeros(full.max().item() + 1, dtype=th.float64, device=DEV).index_add_(0, full, th.ones_like(m))
                mu = th.zeros_like(cnt).index_add_(0, full, m) / cnt
                sd = (th.zeros_like(cnt).index_add_(0, full, (m - mu[full]) ** 2) / cnt).sqrt().clamp(min=1e-9)
                p_cell = th.special.ndtr(-mu / sd)
                p_fit = th.zeros(C, dtype=th.float64, device=DEV).index_add_(0, cfg, p_cell[full]) / D
                print(f"    {name:40s} bias {(p_fit - p_emp).mean():+.4f}, per-config rms {(p_fit - p_emp).pow(2).mean().sqrt():.4f}")
            rate = P.reshape(C, D, 10, 16).double().mean(1)
            var = (rate > 0.02) & (rate < 0.98)
            kind = th.full((C, 10), 3, dtype=th.long, device=DEV)
            kind.scatter_(1, th.tensor(np.array([FAMS[f] for f in fam]), device=DEV), th.tensor([[0, 1, 2]], device=DEV).expand(C, -1))
            by_kind = [int(var[kind == k].sum()) for k in range(4)]
            print(f"   variable gates (flip rate in (0.02, 0.98)) per config: mean {var.sum((1, 2)).double().mean():.1f}; by step "
                  f"{var.sum((0, 2)).cpu().numpy().tolist()}; at steps reading M, S, T, a lower input: {by_kind}")
            n_cfg = len(Z) // D
            Zr, Xr, zr = Z.reshape(n_cfg, D, 10, 16), X[: n_cfg * D].reshape(n_cfg, D, 10), z[: n_cfg * D].reshape(n_cfg, D, 7)
            idx = var[:n_cfg].nonzero()
            r2_self, r2_all, late = [], [], []
            ones = th.ones(D, 1, dtype=th.float64, device=DEV)
            for c, t, n in idx.tolist():
                zz = Zr[c, :, t, n]
                vz = zz.var(correction=0).clamp(min=1e-12)
                for A, out in ((th.cat([Xr[c, :, t, None], ones], 1), r2_self), (th.cat([zr[c], ones], 1), r2_all)):
                    f = A @ th.linalg.lstsq(A, zz[:, None]).solution
                    out.append(1 - ((zz - f[:, 0]) ** 2).mean() / vz)
                late.append(kind[c, t].item() == 3 and t > int(np.array(FAMS[fam[c]]).min()))
            r2_self, r2_all, late = th.stack(r2_self), th.stack(r2_all), th.tensor(late, device=DEV)
            print(f"   a variable gate's pre-activation ({len(idx)} gates in {n_cfg} configurations), R^2 on the step's own input: median {r2_self.median():.3f}, "
                  f"share above 0.9 {(r2_self > 0.9).double().mean():.3f}; on all seven lower inputs linearly: median {r2_all.median():.3f}, share above 0.9 {(r2_all > 0.9).double().mean():.3f}")
            print(f"   the same at lower-input steps after the first of M, S, T ({int(late.sum())} gates): own input median {r2_self[late].median():.3f}, "
                  f"share above 0.9 {(r2_self[late] > 0.9).double().mean():.3f}; all seven median {r2_all[late].median():.3f}")
