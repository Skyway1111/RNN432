"""Which neurons compute which known quantity (running max, new record, leave-one-out comparators) in each final model?

Validates the probe on the example model, whose roles ARC identified, then asks whether the zoo seeds share that core,
as single clean neurons or as a distributed (linearly decodable) code.
Finding: research/findings/circuit.md
"""
import numpy as np

from az.model import NAMES, load
from experiments.common import ROLES, inputs, role_r2

np.set_printoptions(precision=3, suppress=True, linewidth=200)
x = inputs(2**18, seed=20)
print("roles:", ", ".join(f"{i}={r}" for i, r in enumerate(ROLES)))
for name in NAMES:
    w = load(name)
    r2 = role_r2(w, x).cpu().numpy()
    best = r2.argmax(1)
    hits = [f"n{i}:{ROLES[best[i]]} {r2[i, best[i]]:.3f}" for i in np.argsort(-r2.max(1)) if r2[i, best[i]] > 0.9]
    print(f"{name}: single neurons with a role fit R2 > 0.9: {len(hits)}  " + " | ".join(hits))
    print(f"   best single-neuron R2 per role: {r2.max(0)}")
    print(f"   decode-from-all-16 R2 per role: {role_r2(w, x, decode=True).cpu().numpy()}")
