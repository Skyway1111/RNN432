"""Gaussian primitives for deduction-projection estimators, vectorized over a leading batch axis.

Unnormalized moments (m0, m1, m2) = (E[1_A], E[u 1_A], E[u u^T 1_A]) let events be added and subtracted.
"""
import numpy as np
from scipy.special import ndtr

_X, _W = np.polynomial.legendre.leggauss(20)
SQ2PI = np.sqrt(2 * np.pi)


def phi(x):
    return np.exp(-0.5 * x * x) / SQ2PI


def bvnu(h, k, r):
    """P(U > h, V > k) for standard bivariate normal (U, V) with correlation r (Genz's BVNU)."""
    h, k, r = np.broadcast_arrays(np.asarray(h, float), np.asarray(k, float), np.asarray(r, float))
    out = np.empty(h.shape)
    lo = np.abs(r) < 0.925
    if lo.any():
        hh, kk, rr = h[lo], k[lo], r[lo]
        hk, hs, asr = hh * kk, (hh * hh + kk * kk) / 2, np.arcsin(rr)
        sn = np.sin(asr[..., None] * (1 + _X) / 2)
        s = (_W * np.exp((sn * hk[..., None] - hs[..., None]) / (1 - sn * sn))).sum(-1)
        out[lo] = s * asr / (4 * np.pi) + ndtr(-hh) * ndtr(-kk)
    hi = ~lo
    if hi.any():
        hh, rr = h[hi], r[hi]
        kk = np.where(rr < 0, -k[hi], k[hi])
        hk = hh * kk
        a2 = np.maximum((1 - rr) * (1 + rr), 0.0)
        a = np.sqrt(a2)
        bs = (hh - kk) ** 2
        c, d = (4 - hk) / 8, (12 - hk) / 16
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            bvn = np.where(a2 > 0, a * np.exp(-(bs / a2 + hk) / 2) * (1 - c * (bs - a2) * (1 - d * bs / 5) / 3 + c * d * a2 * a2 / 5), 0.0)
            b = np.sqrt(bs)
            corr = np.exp(-hk / 2) * SQ2PI * ndtr(-b / a) * b * (1 - c * bs * (1 - d * bs / 5) / 3)
            bvn = bvn - np.where((hk > -160) & (a2 > 0), corr, 0.0)
            ah = a / 2
            xs = (ah[..., None] * (1 + _X)) ** 2
            rs = np.sqrt(1 - xs)
            t = np.exp(-bs[..., None] / (2 * xs) - hk[..., None] / (1 + rs)) / rs - np.exp(-(bs[..., None] / xs + hk[..., None]) / 2) * (1 + c[..., None] * xs * (1 + d[..., None] * xs))
            bvn = bvn + np.where(a2 > 0, ah * (_W * np.nan_to_num(t)).sum(-1), 0.0)
        bvn = -bvn / (2 * np.pi)
        pos = rr > 0
        bvn = np.where(pos, bvn + ndtr(-np.maximum(hh, kk)), -bvn + np.where(kk > hh, ndtr(kk) - ndtr(hh), 0.0))
        out[hi] = bvn
    return np.clip(out, 0.0, 1.0)


def truncate(m0, mu, S, a):
    """Restrict N(mu, S) with mass m0 to a.u > 0. Returns (mass, mean, cov) of the restricted part."""
    Sa = np.einsum("...ij,...j->...i", S, a)
    v = np.maximum(np.einsum("...i,...i->...", a, Sa), 1e-300)
    sd = np.sqrt(v)
    al = np.einsum("...i,...i->...", a, mu) / sd
    P = ndtr(al)
    lam = np.where(P > 1e-300, phi(al) / np.maximum(P, 1e-300), -al)
    mu2 = mu + Sa * (lam / sd)[..., None]
    S2 = S - np.einsum("...i,...j->...ij", Sa, Sa) * (lam * (lam + al) / v)[..., None, None]
    return m0 * P, mu2, S2


def relu_moments(mu, S):
    """Mean and covariance of ReLU(z) for z ~ N(mu, S), exact pairwise (Rosenbaum), batched."""
    s = np.sqrt(np.maximum(np.einsum("...ii->...i", S), 1e-300))
    a = mu / s
    Pa, pa = ndtr(a), phi(a)
    m = mu * Pa + s * pa
    rho = np.clip(S / (s[..., :, None] * s[..., None, :]), -1.0, 1.0)
    h, k = -a[..., :, None], -a[..., None, :]
    h, k = np.broadcast_arrays(h, k)
    L = bvnu(h, k, rho)
    r1 = np.sqrt(np.maximum(1 - rho * rho, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        q1 = np.where(r1 > 0, ndtr((rho * h - k) / r1), (rho * h > k).astype(float))
        q2 = np.where(r1 > 0, ndtr((rho * k - h) / r1), (rho * k > h).astype(float))
        e2 = np.where(r1 > 0, r1 * phi(np.sqrt(np.maximum(h * h - 2 * rho * h * k + k * k, 0.0)) / np.where(r1 > 0, r1, 1.0)) / SQ2PI, 0.0)
    ph, pk = phi(h), phi(k)
    EU = ph * q1 + rho * pk * q2
    EV = pk * q2 + rho * ph * q1
    EUV = rho * L + rho * h * ph * q1 + rho * k * pk * q2 + e2
    m1, m2 = mu[..., :, None], mu[..., None, :]
    s1, s2 = s[..., :, None], s[..., None, :]
    E = m1 * m2 * L + m1 * s2 * EV + m2 * s1 * EU + s1 * s2 * EUV
    d = np.arange(mu.shape[-1])
    E[..., d, d] = (mu * mu + s * s) * Pa + mu * s * pa
    return m, E - m[..., :, None] * m[..., None, :], Pa


def merge(m0, mu, S, key, n):
    """Moment-match the mixture components sharing each key in range(n). Returns (mass, mean, cov)."""
    M0 = np.zeros(n)
    np.add.at(M0, key, m0)
    M1 = np.zeros((n, mu.shape[-1]))
    np.add.at(M1, key, m0[:, None] * mu)
    M2 = np.zeros((n,) + S.shape[1:])
    np.add.at(M2, key, m0[:, None, None] * (S + mu[:, :, None] * mu[:, None, :]))
    w = np.maximum(M0, 1e-300)
    mean = M1 / w[:, None]
    return M0, mean, M2 / w[:, None, None] - mean[:, :, None] * mean[:, None, :]
