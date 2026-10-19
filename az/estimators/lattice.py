"""Family-stratified lattice: enumerate the 720 families, stratify the top three by their exact order-statistic laws, and
cover the rest with a Kronecker lattice; average the model's correctness over the nodes.

Input space factors exactly (parked/research/angles/margin-gaussian.md): the family (p, q, r), uniform over 720; T, the 3rd largest
value; M given T, the larger of two N(0, 1) values above T; S given (T, M), N(0, 1) restricted to (T, M); and seven lower
inputs i.i.d. N(0, 1) truncated above at T. Per family, k_T x k_M strata of equal mass in (T, M | T); one node per stratum,
its position in the stratum, S and the lower inputs from Roberts' Kronecker sequence over the (family, stratum) index,
mapped through the exact inverse CDFs. The lattice is deterministic cubature; it has no random numbers.
Budget picks the largest (k, k) or (k, k + 1) strata it affords. Below one node per family it uses the 90 families
(p, q, L[p][q]) of an idempotent Latin square, which balance every pair of positions, with m nodes each stratified in T.
Ops are one forward pass and the inverse CDFs per node.
"""
from functools import cache

import numpy as np
import torch as th
from scipy.stats import norm

from az.estimators.gauss_margin import FAMS, FREE, Q
from az.model import SAMPLE_OPS, logits
from az.ops import FN

NODE_FN = 12  # inverse CDFs of T, M, S and seven lower inputs, and Phi(T), Phi(M)
DEV = "cuda" if th.cuda.is_available() else "cpu"


def roberts(d: int) -> np.ndarray:
    """Roberts' generalized golden ratio sequence step in d dimensions: phi with phi^(d+1) = phi + 1."""
    phi = 1.0
    for _ in range(100):
        phi = (1 + phi) ** (1 / (d + 1))
    return (1 / phi) ** np.arange(1, d + 1) % 1


@cache
def _cdf_t():
    t = np.linspace(-5, 5, 200001)
    c = np.cumsum(360 * norm.pdf(t) * norm.cdf(t) ** 7 * (1 - norm.cdf(t)) ** 2)
    return t, c / c[-1]


def family_nodes(kt: int, km: int, d: int, shift=0.5):
    """Lattice points u (720, kt km, d) over the (family, stratum) index, and per node T and M from their strata
    (u[..., 0] places T in its stratum, u[..., 1] places M | T); the other d - 2 coordinates are free."""
    n = kt * km
    u = (np.asarray(shift) + np.arange(len(FAMS) * n).reshape(-1, n, 1) * roberts(d)) % 1.0
    t, cdf = _cdf_t()
    T = np.interp((np.arange(n) // km + u[..., 0]) / kt, cdf, t)
    M = norm.ppf(norm.cdf(T) + (1 - norm.cdf(T)) * np.sqrt((np.arange(n) % km + u[..., 1]) / km))
    return u, T, M


def place(T, M, S, z):
    """Inputs (720, n, 10) with each family's M, S, T at p, q, r and the lower inputs z (720, n, 7) in its free slots."""
    X = np.zeros((*T.shape, 10))
    for k, (p, q, r) in enumerate(FAMS):
        X[k][:, FREE[k]] = z[k]
        X[k, :, p], X[k, :, q], X[k, :, r] = M[k], S[k], T[k]
    return X


@cache
def latin_families() -> np.ndarray:
    """Indices into FAMS of (p, q, L[p][q]), p != q, for the first idempotent Latin square of order 10 by backtracking."""
    L = [[a if a == b else None for b in range(10)] for a in range(10)]
    todo = [(a, b) for a in range(10) for b in range(10) if a != b]

    def fill(k):
        if k == len(todo):
            return True
        a, b = todo[k]
        for v in range(10):
            if all(L[a][j] != v for j in range(10)) and all(L[j][b] != v for j in range(10)):
                L[a][b] = v
                if fill(k + 1):
                    return True
                L[a][b] = None
        return False

    fill(0)
    return np.array([FAMS.index((a, b, L[a][b])) for a, b in todo])


@cache
def inputs(kt: int, km: int, shift: tuple = (0.5,) * 10):
    """All 720 families, kt x km strata each: inputs (720 kt km, 10) and label slots. shift is the lattice's offset."""
    u, T, M = family_nodes(kt, km, 10, shift)
    S = norm.ppf(norm.cdf(T) + (norm.cdf(M) - norm.cdf(T)) * u[..., 9])
    z = norm.ppf(norm.cdf(T)[..., None] * u[..., 2:9])
    return th.tensor(place(T, M, S, z).reshape(-1, 10), device=DEV), th.tensor(np.repeat(Q, kt * km), device=DEV)


@cache
def latin_inputs(m: int, shift: tuple = (0.5,) * 10):
    """The 90 Latin families, m nodes each stratified in T: inputs (90 m, 10) and label slots. shift is the lattice's offset."""
    fams = latin_families()
    u = (np.asarray(shift) + np.arange(len(fams) * m)[:, None] * roberts(10)) % 1.0
    t, cdf = _cdf_t()
    T = np.interp((np.arange(len(u)) % m + u[:, 0]) / m, cdf, t)
    M = norm.ppf(norm.cdf(T) + (1 - norm.cdf(T)) * np.sqrt(u[:, 1]))
    S = norm.ppf(norm.cdf(T) + (norm.cdf(M) - norm.cdf(T)) * u[:, 9])
    z = norm.ppf(norm.cdf(T)[:, None] * u[:, 2:9])
    X = np.zeros((len(u), 10))
    for k, f in enumerate(np.repeat(fams, m)):
        p, q, r = FAMS[f]
        X[k, FREE[f]] = z[k]
        X[k, p], X[k, q], X[k, r] = M[k], S[k], T[k]
    return th.tensor(X, device=DEV), th.tensor(Q[np.repeat(fams, m)], device=DEV)


def strata(budget: float, per_node: float):
    """The largest (k, k) or (k, k + 1) strata per family the budget affords at per_node ops each, at least (1, 1)."""
    n = budget / (len(FAMS) * per_node)
    k = max(1, int(np.sqrt(n)))
    return (k, k + 1) if k * (k + 1) <= n else (k, k)


def estimate(w, budget):
    nodes = budget / (SAMPLE_OPS + NODE_FN * FN)
    X, q = inputs(*strata(budget, SAMPLE_OPS + NODE_FN * FN)) if nodes >= len(FAMS) else latin_inputs(max(1, int(nodes // 90)))
    with th.no_grad():
        hits = sum((logits([w], X[c : c + 2**20][None])[0].argmax(1) == q[c : c + 2**20]).sum().item() for c in range(0, len(X), 2**20))
    return hits / len(X), len(X) * (SAMPLE_OPS + NODE_FN * FN)
