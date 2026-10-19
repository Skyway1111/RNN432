"""Can the six-neuron core {2,4,7,6,1,8} of the example model decide alone? (Indices are the example's; zoo seeds are permuted.)

Finding: research/findings/circuit.md
"""
import numpy as np

from az.model import W, load, trace
from experiments.common import inputs, ranks

x = inputs(2**21, seed=13)
q = ranks(x)[0][:, 1]
core = [2, 4, 7, 6, 1, 8]
sat = [i for i in range(16) if i not in core]
Wi, Wh, Wo = load("example")
cut = Wh.copy()
cut[np.ix_(core, sat)] = 0
no_ro = Wo.copy()
no_ro[:, sat] = 0
full = trace(W(Wi, Wh, Wo), x)[0].argmax(-1) == q
for name, w in [("no satellite -> core weights", W(Wi, cut, Wo)), ("no satellite readout", W(Wi, Wh, no_ro)), ("core only", W(Wi, cut, no_ro))]:
    ok = trace(w, x)[0].argmax(-1) == q
    print(f"{name}: accuracy {ok.double().mean().item():.4f}  agreement with full model {(ok == full).double().mean().item():.4f}")
print(f"full: accuracy {full.double().mean().item():.4f}")
