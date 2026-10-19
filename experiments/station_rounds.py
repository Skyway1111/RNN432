"""How many rounds of gates triggering gates does an account of the tie decision need?

A satellite's gate at a lower-input step is a threshold on that input, set by the state the earlier inputs built
(research/findings/errors.md). Treat each (step, neuron) gate as a station: round 0 freezes every gate at a reference
route, so the model is linear in the inputs; round 1 computes each gate's sign from the reference state and the actual
input, so every station is a fixed threshold on its own step's input; round k computes the gates from round k - 1's
states, so stations shift the thresholds of later stations; the rounds converge to the true model where the gate
pattern stabilizes. For 256 configurations (family, T, M) from the exact laws and 4,096 draws of the lower inputs each,
at the tie S = T and with S drawn, three suite models: per round, the share of gate bits that match the true route, the
R^2 of the round's margin against the true one inside a configuration, the share of draws whose correctness it gets
right, and its P(wrong) against the truth (bias, per-configuration rms). The reference route puts the seven lower
inputs at equally spaced quantiles of their law given T, in slot order.

Finding: research/findings/circuit.md
"""
import numpy as np
import torch as th
from scipy.stats import norm

from az.bench import suite
from az.estimators.gauss_margin import FAMS, FREE
from az.estimators.lattice import _cdf_t
from experiments.common import DEV

labels, ws = suite(0.002, 32)
rng = np.random.default_rng(5)
th.manual_seed(5)
t_grid, cdf_t = _cdf_t()
C, D, ROUNDS = 256, 4096, 6


def configs(n):
    fam = rng.integers(720, size=n)
    T = np.interp(rng.uniform(size=n), cdf_t, t_grid)
    M = norm.ppf(norm.cdf(T) + (1 - norm.cdf(T)) * np.sqrt(rng.uniform(size=n)))
    return fam, T, M


def place(fam, T, M, S, z):
    """Inputs (N, 10) from per-row family, top three and lower inputs (N, 7) in slot order."""
    X = th.zeros(len(z), 10, dtype=th.float64, device=DEV)
    free = th.tensor(FREE, device=DEV)[fam]
    X.scatter_(1, free, z)
    pqr = th.tensor(FAMS, device=DEV)[fam]
    ar = th.arange(len(z), device=DEV)
    X[ar, pqr[:, 0]], X[ar, pqr[:, 1]], X[ar, pqr[:, 2]] = M, S, T
    return X, pqr[:, 1]


def run(W, X, gates=None, from_states=None):
    """Forward pass. gates (N, 10, 16) forces the gate bits; from_states (N, 10, 16) computes each step's gates from
    those states and the actual input instead of the pass's own state. Returns gate bits, states (N, 10, 16) before
    each step, and logits."""
    Wi, Wh, Wo = W
    h = th.zeros(len(X), 16, dtype=th.float64, device=DEV)
    P, H = [], []
    for t in range(10):
        H.append(h)
        z = h @ Wh.T + X[:, t, None] * Wi
        if gates is not None:
            on = gates[:, t]
        elif from_states is not None:
            on = (from_states[:, t] @ Wh.T + X[:, t, None] * Wi) > 0
        else:
            on = z > 0
        P.append(on)
        h = z * on
    return th.stack(P, 1), th.stack(H, 1), h @ Wo.T


def margin(L, q):
    return L.gather(1, q[:, None])[:, 0] - L.scatter(1, q[:, None], -1e9).max(1).values


def main(i):
    W = [th.tensor(a, device=DEV, dtype=th.float64) for a in ws[i]]
    fam_np, T_np, M_np = configs(C)
    fam = th.tensor(np.repeat(fam_np, D), device=DEV)
    T = th.tensor(np.repeat(T_np, D), device=DEV)
    M = th.tensor(np.repeat(M_np, D), device=DEV)
    print(labels[i])
    for tie in (True, False):
        u = th.rand(C * D, 8, device=DEV, dtype=th.float64)
        z = th.special.ndtri(th.special.ndtr(T)[:, None] * u[:, :7])
        S = T + 1e-9 if tie else th.special.ndtri(th.special.ndtr(T) + (th.special.ndtr(M) - th.special.ndtr(T)) * u[:, 7])
        X, q = place(fam, T, M, S, z)
        zbar = th.special.ndtri(th.special.ndtr(T)[:, None] * th.tensor((np.arange(7) + 0.5) / 7, device=DEV))
        Xbar, _ = place(fam, T, M, S, zbar)
        with th.no_grad():
            P_true, _, L_true = run(W, X)
            m_true = margin(L_true, q)
            ok_true = m_true > 0
            P_ref, H_ref, _ = run(W, Xbar)
            rows = []
            P_k, H_k, L_k = run(W, X, gates=P_ref)  # round 0: the reference route's gates, linear in the inputs
            for k in range(ROUNDS + 1):
                m_k = margin(L_k, q)
                bits = (P_k == P_true).double().mean()
                r2 = 1 - ((m_k - m_true) ** 2).reshape(C, D).mean(1) / m_true.reshape(C, D).var(1, correction=0).clamp(min=1e-12)
                agree = ((m_k > 0) == ok_true).double().mean()
                d = (m_k < 0).double().reshape(C, D).mean(1) - (~ok_true).double().reshape(C, D).mean(1)
                rows.append(f"    round {k}: gate bits right {bits:.4f}; margin R^2 mean {r2.mean():.3f}, median {r2.median():.3f}; "
                            f"correctness agrees {agree:.4f}; P(wrong) bias {d.mean():+.4f}, per-config rms {d.pow(2).mean().sqrt():.4f}")
                if k == ROUNDS:
                    break
                states = H_ref if k == 0 else H_k  # round 1 reads the reference states, later rounds the previous round's
                P_k, H_k, L_k = run(W, X, from_states=states)
        print(f"  {'at the tie S = T' if tie else 'S drawn'}: P(wrong) {(~ok_true).double().mean():.4f}; inputs agreeing with the reference route on every gate {(P_true == P_ref).all((1, 2)).double().mean():.3f}")
        print("\n".join(rows))


if __name__ == "__main__":
    for i in (32, 0, 160):
        main(i)
