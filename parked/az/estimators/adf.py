"""Covariance propagation conditioned on the label state (ARC's canonical deduction-projection estimator).

State per key (p, q) = positions of the running max and 2nd max: mass and a Gaussian over
v = [h (16), M, S]. Each step appends x_t ~ N(0, 1), splits on x_t vs M and S (exact truncations),
pushes z = Wh h + Wi x_t through ReLU (exact pairwise moments), and moment-matches keys that meet.
At the end P(argmax Wo h = q) comes from sequential truncation on the 9 margins.
"""
import numpy as np

from az import gauss
from az.ops import BVN, FN

H = 16
NV = H + 2  # [h, M, S]
NU = NV + 1  # [h, M, S, x]
IM, IS, IX = H, H + 1, H + 2


def _relu_step(w, m0, mu, S, sel, ops):
    """Children (K, NU) -> (K, NV): v' = [ReLU(Wh h + Wi x), u[sel]]."""
    Wi, Wh, _ = w
    Z = np.zeros((H, NU))
    Z[:, :H], Z[:, IX] = Wh, Wi
    zm = mu @ Z.T
    ZS = np.einsum("ij,kjl->kil", Z, S)
    zS = np.einsum("kil,jl->kij", ZS, Z)
    rm, rC, Pa = gauss.relu_moments(zm, zS)
    K = len(m0)
    mu2 = np.concatenate([rm, mu[np.arange(K)[:, None], sel]], 1)
    S2 = np.empty((K, NV, NV))
    S2[:, :H, :H] = rC
    cross = ZS[np.arange(K)[:, None, None], np.arange(H)[None, :, None], sel[:, None, :]] * Pa[:, :, None]
    S2[:, :H, H:], S2[:, H:, :H] = cross, cross.transpose(0, 2, 1)
    S2[:, H:, H:] = S[np.arange(K)[:, None, None], sel[:, :, None], sel[:, None, :]]
    ops[0] += K * (H * NU * NU + H * H * NU + H * (H + 1) // 2 * (BVN + 30) + H * 3 * FN)
    return m0, mu2, S2


def _cut(m0, mu, S, coef, ops):
    """Truncate each component to sum_i coef[:, i] u_i > 0."""
    ops[0] += len(m0) * (3 * NU * NU + 2 * FN)
    return gauss.truncate(m0, mu, S, coef)


def propagate(w, ops):
    """Final keys after 10 steps: (mass, mean, cov over [h, M, S], p, q)."""
    Wi, Wh, Wo = w
    # t = 0: v = [ReLU(Wi x0), x0, unused], exact moments of a rank-one ReLU.
    same = np.sign(Wi)[:, None] == np.sign(Wi)[None, :]
    mu = np.zeros((1, NV))
    S = np.zeros((1, NV, NV))
    mu[0, :H] = np.abs(Wi) / gauss.SQ2PI
    S[0, :H, :H] = np.where(same, np.abs(np.outer(Wi, Wi)) / 2, 0.0) - np.outer(mu[0, :H], mu[0, :H])
    S[0, :H, IM] = S[0, IM, :H] = Wi / 2
    S[0, IM, IM] = S[0, IS, IS] = 1.0
    m0, p, q = np.ones(1), np.zeros(1, int), np.full(1, -1)
    for t in range(1, 10):
        K = len(m0)
        mu_u = np.concatenate([mu, np.zeros((K, 1))], 1)
        S_u = np.zeros((K, NU, NU))
        S_u[:, :NV, :NV], S_u[:, IX, IX] = S, 1.0
        dM = np.zeros((K, NU))
        dM[:, IX], dM[:, IM] = 1.0, -1.0  # x - M
        a0, a1, a2 = _cut(m0, mu_u, S_u, dM, ops)  # new max
        kids = [(a0, a1, a2, np.array([IX, IM]), np.full(K, t), p)]
        if t == 1:
            b0, b1, b2 = _cut(m0, mu_u, S_u, -dM, ops)
            kids.append((b0, b1, b2, np.array([IM, IX]), p, np.full(K, t)))
        else:
            dS = np.zeros((K, NU))
            dS[:, IX], dS[:, IS] = 1.0, -1.0  # x - S
            s0, s1, s2 = _cut(m0, mu_u, S_u, dS, ops)
            c0, c1, c2 = _cut(m0, mu_u, S_u, -dS, ops)
            # S < x < M as {x > S} minus {x > M}, in unnormalized moments.
            n0 = s0 - a0
            n1 = s0[:, None] * s1 - a0[:, None] * a1
            n2 = s0[:, None, None] * (s2 + s1[:, :, None] * s1[:, None, :]) - a0[:, None, None] * (a2 + a1[:, :, None] * a1[:, None, :])
            ok = n0 > 1e-14 * m0
            b0 = np.where(ok, n0, 0.0)
            b1 = n1 / np.where(ok, n0, 1.0)[:, None]
            b2 = n2 / np.where(ok, n0, 1.0)[:, None, None] - b1[:, :, None] * b1[:, None, :]
            kids.append((b0, b1, b2, np.array([IM, IX]), p, np.full(K, t)))
            kids.append((c0, c1, c2, np.array([IM, IS]), p, q))
        parts = [_relu_step(w, k0, k1, k2, np.tile(sel, (len(k0), 1)), ops) + (kp, kq) for k0, k1, k2, sel, kp, kq in kids]
        m0, mu, S, pp, qq = (np.concatenate(z) for z in zip(*parts))
        live = m0 > 0
        key = pp * 11 + qq + 1
        uk, inv = np.unique(key[live], return_inverse=True)
        m0, mu, S = gauss.merge(m0[live], mu[live], S[live], inv, len(uk))
        p, q = uk // 11, uk % 11 - 1
        ops[0] += live.sum() * NV * NV * 2
    return m0, mu, S, p, q


def estimate(w, budget=None):
    Wo = w.Wo
    ops = [0.0]
    m0, mu, S, p, q = propagate(w, ops)
    # P(argmax Wo h = q) per key by sequential truncation, most constraining margin first.
    total = 0.0
    for k in range(len(m0)):
        mh, Sh = mu[k, :H], S[k, :H, :H]
        A = Wo[q[k]][None, :] - np.delete(Wo, q[k], 0)
        mass = m0[k]
        sd = np.sqrt(np.maximum(np.einsum("ji,il,jl->j", A, Sh, A), 1e-300))
        for j in np.argsort(A @ mh / sd):
            mass, mh, Sh = gauss.truncate(mass, mh, Sh, A[j])
        total += mass
        ops[0] += 9 * (3 * H * H + 2 * FN)
    return float(total / m0.sum()), ops[0]
