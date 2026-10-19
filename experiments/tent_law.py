"""The tent law: along the 2nd max's line the error set is two end wedges, so tracking inward from each end until the
first correctness switch gives the exact line integral at a fraction of the full tracking's cost. Does it, and what
law do the wedges follow over the lower inputs?

  1. On 2^14 lines from the exact laws (family, T, M | T, seven lower inputs | T), three suite models: sline.ends
     against the full sline.sline; the share of lines resolved within one piece from each end, within two, and left
     unresolved after eight; the difference's bias (the interior wrong intervals the tent misses) and RMS; the
     multiply-adds per line of each, in passes.
  2. At the T end: for 512 configurations (family, T, M) and 1024 draws of the lower inputs each, the tie margin m_T
     (l_q minus the best other logit at S = T) and its slope along S. The T-wedge's mass, exact from ends, against the
     first-order wedge max(0, -m_T / slope) where the slope is positive; the share of draws whose margin falls as S
     rises; and the law of m_T over the lower inputs: P(m_T < 0) empirically, under a Gaussian with the draws' moments,
     under the 16-run design's moments, and under the composition the brief names, the largest lower input exact
     (its slot and 16 quantile bins of its value) with a Gaussian residual per bin; with the share of m_T's variance
     the largest lower input explains.

Finding: parked/research/angles/family-lattice.md, parked/research/angles/margin-gaussian.md
"""
import numpy as np
import torch as th
from scipy.stats import norm

from az.bench import suite
from az.estimators.gauss_margin import FAMS, FREE, Q, lower_inputs
from az.estimators.lattice import _cdf_t
from az.estimators.sline import NODE, PIECE_READ, PIECE_STEP, STEP, sline
from az.model import SAMPLE_OPS, logits
from experiments.common import DEV

labels, ws = suite(0.002, 32)
rng = np.random.default_rng(1)
t_grid, cdf_t = _cdf_t()


def ends(Wi, Wh, Wo, X, q, T, M, rounds: int = 8):
    """P(correct) on each line from its two ends, by the tent law: the error set is the two end wedges. From each end,
    track the piece touching it (sline.sline's bookkeeping, one piece at a time), and the next piece inward when the
    whole piece is wrong, until the first correctness switch; P(correct) = 1 - the two wedges' mass. Exact on lines
    with at most two switches. Returns P(correct), ops per line, the lines left unresolved after `rounds` pieces from
    an end (their wedge then ends at the last piece tracked), and the two wedges' masses (lines, 2) as shares of
    the line."""
    n, dev = len(X), X.device
    out = th.zeros(n, dtype=X.dtype, device=dev)
    ops = th.full_like(out, NODE)
    stuck = th.zeros(n, dtype=th.bool, device=dev)
    wedges = th.zeros(n, 2, dtype=X.dtype, device=dev)
    inf = th.tensor(th.inf, dtype=X.dtype, device=dev)
    for qq in range(10):
        sel = (q == qq).nonzero().squeeze(1)
        if not len(sel):
            continue
        x, lo_line, hi_line = X[sel], T[sel], M[sel]
        h = th.zeros(len(sel), 16, dtype=X.dtype, device=dev)
        for t in range(qq):
            h = th.relu(h @ Wh.T + x[:, t, None] * Wi)
        ops[sel] += qq * STEP
        norm_line = th.special.ndtr(hi_line) - th.special.ndtr(lo_line)
        wedge = th.zeros(len(sel), 2, dtype=X.dtype, device=dev)
        whole = th.zeros(len(sel), dtype=th.bool, device=dev)  # the T side found the whole line wrong
        for up in (True, False):  # anchored at T with pieces going up, then at M going down
            C = th.zeros(len(sel), 10, 16, dtype=X.dtype, device=dev)
            D = th.zeros_like(C)
            R = th.full_like(C, th.nan)  # every gate's root at every step, valid for the steps not yet retracked
            lo, hi = lo_line.clone(), hi_line.clone()
            resume = th.full((len(sel),), qq, dtype=th.long, device=dev)
            far_step = th.zeros(len(sel), dtype=th.long, device=dev)  # the step whose root bounds the piece inward
            live = ~whole
            for _ in range(rounds):
                if not live.any():
                    break
                idx = live.nonzero().squeeze(1)
                for t in range(qq, 10):
                    act = idx[resume[idx] <= t]
                    if not len(act):
                        continue
                    pc = h[act] if t == qq else C[act, t - 1]
                    pd = th.zeros_like(pc) if t == qq else D[act, t - 1]
                    A = pc @ Wh.T + (0 if t == qq else x[act, t, None] * Wi)
                    B = pd @ Wh.T + (Wi if t == qq else 0)
                    R[act, t] = th.where(B != 0, -A / B, th.nan)
                    ops[sel[act]] += PIECE_STEP
                    r = R[act, qq : t + 1].reshape(len(act), -1)
                    if up:
                        near = th.where(r > lo[act, None], r, inf).min(1)
                        hi[act] = th.minimum(hi_line[act], near.values)
                    else:
                        near = th.where(r < hi[act, None], r, -inf).max(1)
                        lo[act] = th.maximum(lo_line[act], near.values)
                    far_step[act] = qq + near.indices // 16
                    on = ((A + B * ((lo[act] + hi[act]) / 2)[:, None]) > 0).to(X.dtype)
                    C[act, t], D[act, t] = on * A, on * B
                ops[sel[idx]] += PIECE_READ
                Lc, Ld = C[idx, 9] @ Wo.T, D[idx, 9] @ Wo.T
                a, b = Lc[:, qq, None] - Lc, Ld[:, qq, None] - Ld
                other = th.ones_like(a, dtype=th.bool)
                other[:, qq] = False
                root = -a / b
                lo_c = th.maximum(lo[idx], th.where(other & (b > 0), root, -th.inf).max(1).values)
                hi_c = th.minimum(hi[idx], th.where(other & (b < 0), root, th.inf).min(1).values)
                good = ~(other & (b == 0) & (a <= 0)).any(1) & (lo_c < hi_c)
                piece = th.special.ndtr(hi[idx]) - th.special.ndtr(lo[idx])
                if up:
                    wedge[idx, 0] += th.where(good, th.special.ndtr(lo_c) - th.special.ndtr(lo[idx]), piece)
                    whole[idx] |= ~good & (hi[idx] >= hi_line[idx])
                    lo[idx], hi[idx] = th.where(good, lo[idx], hi[idx]), th.where(good, hi[idx], hi_line[idx])
                else:
                    wedge[idx, 1] += th.where(good, th.special.ndtr(hi[idx]) - th.special.ndtr(hi_c), piece)
                    whole[idx] |= ~good & (lo[idx] <= lo_line[idx])
                    lo[idx], hi[idx] = th.where(good, lo[idx], lo_line[idx]), th.where(good, hi[idx], lo[idx])
                resume[idx] = far_step[idx]
                live[idx] = ~good & ~whole[idx]
            stuck[sel[live]] = True
        wedges[sel] = wedge / norm_line[:, None]
        out[sel] = (1 - wedges[sel].sum(1)).clamp(min=0)
    return out, ops, stuck, wedges


def lines(n):
    """Random lines from the exact laws: inputs with x_q = 0 (a placeholder), labels, T, M, and the lower inputs."""
    fam = rng.integers(720, size=n)
    T = np.interp(rng.uniform(size=n), cdf_t, t_grid)
    M = norm.ppf(norm.cdf(T) + (1 - norm.cdf(T)) * np.sqrt(rng.uniform(size=n)))
    z = norm.ppf(norm.cdf(T)[:, None] * rng.uniform(size=(n, 7)))
    X = np.zeros((n, 10))
    for k in range(n):
        p_, _, r_ = FAMS[fam[k]]
        X[k, FREE[fam[k]]] = z[k]
        X[k, r_], X[k, p_] = T[k], M[k]
    return fam, *(th.tensor(a, device=DEV) for a in (X, Q[fam], T, M))


def weights(w):
    return [th.tensor(a, device=DEV, dtype=th.float64) for a in w]


print("1. the tent against the exact line integral")
fam, X, q, T, M = lines(2**14)
for i in (32, 0, 160):
    W = weights(ws[i])
    exact, ops_full = sline(*W, X, q, T, M)
    row = [f"  {labels[i]:10s} full sline {ops_full.mean() / SAMPLE_OPS:.2f} passes"]
    for rounds in (1, 2, 8):
        p, ops, stuck, _ = ends(*W, X, q, T, M, rounds=rounds)
        d = p - exact
        row.append(f"ends({rounds}): {ops.mean() / SAMPLE_OPS:.2f} passes, unresolved {stuck.double().mean():.4f}, "
                   f"bias {d.mean():+.2e}, rms {d.pow(2).mean().sqrt():.2e}, max {d.abs().max():.1e}")
    print("\n    ".join(row))

print("\n2. the law of the T-wedge over the lower inputs")
C, D = 512, 1024
fam, Xc, qc, Tc, Mc = lines(C)
for i in (32, 0, 160):
    W = weights(ws[i])
    T_rep = Tc.repeat_interleave(D)
    z = th.special.ndtri(th.special.ndtr(T_rep)[:, None] * th.rand(C * D, 7, device=DEV, dtype=th.float64))
    X = Xc.repeat_interleave(D, 0)
    free = th.tensor(FREE, device=DEV)[th.tensor(fam, device=DEV)].repeat_interleave(D, 0)
    X.scatter_(1, free, z)
    q_rep, M_rep = qc.repeat_interleave(D), Mc.repeat_interleave(D)
    _, _, _, wedges = ends(*W, X, q_rep, T_rep, M_rep)
    w_emp = wedges[:, 0].reshape(C, D)

    def margin(s):
        Xs = X.clone()
        Xs[th.arange(C * D, device=DEV), q_rep] = s
        L = logits([ws[i]], Xs[None])[0]
        lq = L.gather(1, q_rep[:, None])[:, 0]
        return lq - L.scatter(1, q_rep[:, None], -1e9).max(1).values

    eps = 1e-4
    m = margin(T_rep + 1e-9)
    b = (margin(T_rep + eps) - m) / eps
    norm_line = th.special.ndtr(M_rep) - th.special.ndtr(T_rep)
    rising = b > 0
    w1 = (th.special.ndtr(T_rep + th.clamp(-m / th.where(rising, b, 1.0), min=0)) - th.special.ndtr(T_rep)) / norm_line
    print(f"  {labels[i]:10s} T-end error {w_emp.mean():.4f}; draws with a falling margin at the tie {(~rising).double().mean():.3f}, "
          f"falling and wrong {(~rising & (m < 0)).double().mean():.3f}; on rising draws, first-order wedge minus exact "
          f"{(w1 - wedges[:, 0])[rising].mean():+.5f}")
    m = m.reshape(C, D)
    low = lower_inputs(Tc)  # (C, 16, 7)
    Xd = Xc.repeat_interleave(16, 0)
    Xd.scatter_(1, th.tensor(FREE, device=DEV)[th.tensor(fam, device=DEV)].repeat_interleave(16, 0), low.reshape(-1, 7))
    Xd[th.arange(C * 16, device=DEV), qc.repeat_interleave(16)] = Tc.repeat_interleave(16) + 1e-9
    Ld = logits([ws[i]], Xd[None])[0]
    md = (Ld.gather(1, qc.repeat_interleave(16)[:, None])[:, 0] - Ld.scatter(1, qc.repeat_interleave(16)[:, None], -1e9).max(1).values).reshape(C, 16)
    # the largest lower input exact: m_T by its slot and 16 quantile bins of its value, Gaussian residual per bin
    Lv, Ls = z.max(1)
    bins = (th.special.ndtr(Lv) / th.special.ndtr(T_rep)).pow(7).mul(16).long().clamp(max=15)
    key = (th.arange(C, device=DEV).repeat_interleave(D) * 7 + Ls) * 16 + bins
    ones = th.ones(C * D, dtype=th.float64, device=DEV)
    cnt = th.zeros(C * 7 * 16, dtype=th.float64, device=DEV).index_add_(0, key, ones)
    mu_bin = th.zeros_like(cnt).index_add_(0, key, m.reshape(-1)) / cnt.clamp(min=1)
    var_bin = th.zeros_like(cnt).index_add_(0, key, (m.reshape(-1) - mu_bin[key]) ** 2) / cnt.clamp(min=1)
    p_comp = th.special.ndtr(-mu_bin[key] / var_bin[key].sqrt().clamp(min=1e-9)).reshape(C, D).mean(1)
    fit = mu_bin[key].reshape(C, D)
    r2 = 1 - ((m - fit) ** 2).sum(1) / ((m - m.mean(1, keepdim=True)) ** 2).sum(1).clamp(min=1e-12)
    p_emp = (m < 0).double().mean(1)
    p_g = th.special.ndtr(-m.mean(1) / m.std(1).clamp(min=1e-9))
    p_d = th.special.ndtr(-md.mean(1) / md.std(1).clamp(min=1e-9))
    skew = ((m - m.mean(1, keepdim=True)) ** 3).mean(1) / m.std(1).clamp(min=1e-9) ** 3
    print(f"      P(wrong at the tie) {p_emp.mean():.4f}; Gaussian m_T with the draws' moments {p_g.mean() - p_emp.mean():+.4f} "
          f"(per-config rms {(p_g - p_emp).pow(2).mean().sqrt():.4f}); with the 16-run design's {p_d.mean() - p_emp.mean():+.4f} "
          f"(rms {(p_d - p_emp).pow(2).mean().sqrt():.4f}); largest lower input exact, Gaussian rest {p_comp.mean() - p_emp.mean():+.4f} "
          f"(rms {(p_comp - p_emp).pow(2).mean().sqrt():.4f}); m_T skewness {skew.mean():+.2f}; "
          f"R2 of m_T on the largest lower input's slot and value {r2.mean():.2f}")
