"""How fast does a Gaussian mixture over routes (fixed activation sequences, in input space) grow?

Finding: research/angles/route-mixture.md
"""
import numpy as np
from scipy.special import ndtr

from az import gauss
from az.model import load

CAP = 200_000


def grow(w, eps):
    Wi, Wh, _ = w
    m0, mu, S = np.ones(1), np.zeros((1, 10)), np.eye(10)[None].copy()
    A, p, q = np.zeros((1, 16, 10)), np.zeros(1, int), np.full(1, -1)
    counts = []
    for t in range(10):
        K = len(m0)
        if t >= 1:  # label splits: new max, new 2nd max, neither
            a = np.zeros((K, 10))
            a[:, t] += 1
            a[np.arange(K), p] -= 1
            up, dn = gauss.truncate(m0, mu, S, a), gauss.truncate(m0, mu, S, -a)
            parts = [up + (A, np.full(K, t), p)]
            if t == 1:
                parts.append(dn + (A, p, np.full(K, t)))
            else:
                b = np.zeros((K, 10))
                b[:, t] += 1
                b[np.arange(K), q] -= 1
                parts.append(gauss.truncate(*dn, b) + (A, p, np.full(K, t)))
                parts.append(gauss.truncate(*dn, -b) + (A, p, q))
            m0, mu, S, A, p, q = (np.concatenate(z) for z in zip(*parts))
            keep = m0 > 1e-12
            m0, mu, S, A, p, q = m0[keep], mu[keep], S[keep], A[keep], p[keep], q[keep]
        B = np.einsum("ij,kjl->kil", Wh, A)
        B[:, :, t] += Wi
        act = np.ones((len(m0), 16), bool)
        for i in range(16):  # split only ReLUs whose sign is uncertain beyond eps
            a = B[:, i, :]
            sd = np.sqrt(np.maximum(np.einsum("ki,kij,kj->k", a, S, a), 1e-300))
            P = ndtr(np.einsum("ki,ki->k", a, mu) / sd)
            act[P < eps, i] = False
            idx = np.where((P >= eps) & (P <= 1 - eps))[0]
            if len(idx):
                on, off = gauss.truncate(m0[idx], mu[idx], S[idx], a[idx]), gauss.truncate(m0[idx], mu[idx], S[idx], -a[idx])
                m0[idx], mu[idx], S[idx] = on
                act_off = act[idx].copy()
                act_off[:, i] = False
                m0, mu, S = np.concatenate([m0, off[0]]), np.concatenate([mu, off[1]]), np.concatenate([S, off[2]])
                A, B, act = np.concatenate([A, A[idx]]), np.concatenate([B, B[idx]]), np.concatenate([act, act_off])
                p, q = np.concatenate([p, p[idx]]), np.concatenate([q, q[idx]])
            if len(m0) > CAP:
                return counts + [f">{CAP} at step {t}"]
        A = B * act[:, :, None]
        counts.append(len(m0))
    return counts


w = load("example")
for eps in [0.1, 0.03, 0.01]:
    print(f"eps {eps}: components after each step {grow(w, eps)}")
