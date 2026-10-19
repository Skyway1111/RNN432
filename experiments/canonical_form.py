"""What does the example model look like with big-input neurons rescaled to |W_hi| = 1 and the core listed first?

Finding: research/findings/circuit.md
"""
import numpy as np

from az.model import load

np.set_printoptions(precision=2, suppress=True, linewidth=220)
Wi, Wh, Wo = load("example")
c = np.where(np.abs(Wi) > 0.5, 1 / np.abs(Wi), 1.0)
Wi2, Wh2, Wo2 = Wi * c, Wh * c[:, None] / c[None, :], Wo / c[None, :]
order = [2, 4, 7, 6, 1, 8, 14, 0, 3, 5, 9, 10, 11, 12, 13, 15]
print("neuron order", order)
print("W_hi", Wi2[order])
print("W_hh (rows to, cols from)\n", Wh2[np.ix_(order, order)])
print("W_oh transposed (rows neurons, cols positions)\n", Wo2[:, order].T)
ev = np.linalg.eigvals(Wh)
ev = ev[np.argsort(-np.abs(ev))]
print("eigenvalues of W_hh", ev)
print("moduli", np.abs(ev), " angles (deg)", np.degrees(np.angle(ev)).round(0))
