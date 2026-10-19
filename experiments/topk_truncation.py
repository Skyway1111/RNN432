"""If only the top k ranks are kept (the rest zeroed, or pushed far below), does the model still decide the same way?

Finding: research/findings/linearity.md
"""
from az.model import load, trace
from experiments.common import inputs, ranks

x = inputs(2**21, seed=9)
srt, _ = ranks(x)
rank = srt.argsort(-1)
for name in ["example", "s0", "s2"]:
    w = load(name)
    full = trace(w, x)[0].argmax(-1) == srt[:, 1]
    print(f"{name}: accuracy {full.double().mean().item():.4f}")
    for k in [2, 3, 4, 5, 6, 8]:
        for mode in ["zero", "minus 10"]:
            xx = x.clone()
            m = rank >= k
            xx[m] = 0.0 if mode == "zero" else xx[m] - 10.0
            ok = trace(w, xx)[0].argmax(-1) == srt[:, 1]
            print(f"  top {k}, rest {mode}: accuracy {ok.double().mean().item():.4f}  per-input agreement {(ok == full).double().mean().item():.4f}")
