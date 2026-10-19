"""Where in W_hh does sigma = 0.002 noise do its damage: which neurons' incoming or outgoing weights, and does it add up?

Noise on one row (a neuron's incoming recurrent weights) or one column (its outgoing weights) at a time, 16 draws each,
on all six models. The example's rows are also grouped by the circuit's blocks (core, satellites).

Finding: research/findings/noise.md
"""
import numpy as np

from az.model import NAMES, W, accuracy, load

SIGMA, DRAWS, N = 0.002, 16, 2**20
CORE = [2, 4, 7, 6, 1, 8]  # example model only
SAT = [i for i in range(16) if i not in CORE]


def damage(base, mask, seed):
    """Mean and sd of the accuracy drop over DRAWS copies with noise on the masked entries of W_hh."""
    rng = np.random.default_rng(seed)
    ws = [W(base.Wi, base.Wh + SIGMA * mask * rng.standard_normal(base.Wh.shape), base.Wo) for _ in range(DRAWS)]
    a = accuracy(ws, N, seed=1)
    return a.mean(), a.std()


for i, name in enumerate(NAMES):
    base = load(name)
    acc = accuracy([base], N, seed=1)[0]
    full = damage(base, np.ones((16, 16)), 10 * i)
    print(f"{name}: accuracy {acc:.4f}; all of W_hh: drop {acc - full[0]:.4f} sd {full[1]:.4f}")
    rows, cols = [], []
    for n in range(16):
        m = np.zeros((16, 16)); m[n] = 1
        rows.append(damage(base, m, 100 * i + n))
        m = np.zeros((16, 16)); m[:, n] = 1
        cols.append(damage(base, m, 100 * i + 50 + n))
    print("  drop by row (incoming to neuron n):", " ".join(f"{n}:{acc - r[0]:.4f}" for n, r in enumerate(rows)))
    print("  sd   by row:                       ", " ".join(f"{n}:{r[1]:.4f}" for n, r in enumerate(rows)))
    print("  drop by col (outgoing of neuron n):", " ".join(f"{n}:{acc - c[0]:.4f}" for n, c in enumerate(cols)))
    print(f"  sum of row drops {sum(acc - r[0] for r in rows):.4f}; sum of col drops {sum(acc - c[0] for c in cols):.4f}; full {acc - full[0]:.4f}")
    if name == "example":
        for label, to, frm in [("core<-core", CORE, CORE), ("core<-sat", CORE, SAT), ("sat<-core", SAT, CORE), ("sat<-sat", SAT, SAT)]:
            m = np.zeros((16, 16)); m[np.ix_(to, frm)] = 1
            d = damage(base, m, 1000 + len(label))
            print(f"  {label}: drop {acc - d[0]:.4f} sd {d[1]:.4f} ({len(to) * len(frm)} weights)")
