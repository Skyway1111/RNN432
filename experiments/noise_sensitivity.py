"""How does each model's accuracy respond to weight noise, and which matrix is fragile?

Finding: research/findings/noise.md
"""
import numpy as np

from az.model import NAMES, W, accuracy, load, noisy

for i, name in enumerate(NAMES):
    base = load(name)
    print(f"{name}: accuracy {accuracy([base], 2**22, seed=0)[0]:.4f}")
    for sig in [0.0005, 0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1]:
        a = accuracy(noisy(base, sig, 64, seed=i), 2**16, seed=1)
        print(f"  sigma {sig:<6}: mean {a.mean():.3f} sd {a.std():.3f} min {a.min():.3f} max {a.max():.3f}")
    for k, mat in enumerate(["Wi", "Wh", "Wo"]):
        rng = np.random.default_rng(100 + i)
        ws = [W(*(a + (0.003 * rng.standard_normal(a.shape) if j == k else 0) for j, a in enumerate(base))) for _ in range(64)]
        a = accuracy(ws, 2**16, seed=2)
        print(f"  noise on {mat} only, sigma 0.003: mean {a.mean():.3f} sd {a.std():.3f}")
