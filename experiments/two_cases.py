"""Does the circuit treat the two kinds of 2nd max differently: a demoted max (q before p) and a runner-up (q after p)?

If q < p, x_q was the running max until x_p replaced it, so the answer is "the previous record"; if q > p, x_q never was,
so the answer is "the largest non-record after the max". Splits accuracy by case and delay, silences each neuron per
case, and measures each neuron's mean contribution to the deciding margins per case. Example model.

Finding: research/findings/circuit.md
"""
import numpy as np
import torch as th

from az.model import W, load, trace
from experiments.common import DEV, inputs, ranks

np.set_printoptions(precision=3, suppress=True, linewidth=160)
w = load("example")
x = inputs(2**20, seed=22)
srt, ys = ranks(x)
p, q, r = srt[:, 0], srt[:, 1], srt[:, 2]
demoted = q < p
L, pre, _ = trace(w, x)
h = th.relu(pre[:, -1])
ok = L.argmax(1) == q


def acc(mask):
    return ok[mask].double().mean().item()


print(f"accuracy {acc(th.ones_like(ok)):.4f}; demoted max (q < p) {acc(demoted):.4f} on {demoted.double().mean():.2f} of inputs; runner-up (q > p) {acc(~demoted):.4f}")
print("accuracy by delay 1..9: demoted (p - q):", np.array([acc(demoted & (p - q == d)) for d in range(1, 10)]))
print("                        runner-up (q - p):", np.array([acc(~demoted & (q - p == d)) for d in range(1, 10)]))
print("accuracy by (q row, p column):")
tab = np.full((10, 10), np.nan)
for i in range(10):
    for j in range(10):
        if i != j:
            tab[i, j] = acc((q == i) & (p == j))
print(tab)
near = (ys[:, 1] - ys[:, 2] < 0.1) | (ys[:, 0] - ys[:, 1] < 0.1)
print(f"error rate near a tie / elsewhere: demoted {1 - acc(demoted & near):.3f} / {1 - acc(demoted & ~near):.3f};  runner-up {1 - acc(~demoted & near):.3f} / {1 - acc(~demoted & ~near):.3f}")

print("\nper neuron: accuracy per case with its readout zeroed, and with the neuron silenced (its outgoing weights zeroed)")
Wi, Wh, Wo = w
for n in range(16):
    ro = Wo.copy(); ro[:, n] = 0
    sil = Wh.copy(); sil[:, n] = 0
    a1 = trace(W(Wi, Wh, ro), x)[0].argmax(1) == q
    a2 = trace(W(Wi, sil, ro), x)[0].argmax(1) == q
    print(f"  n{n:2d}: readout off: demoted {a1[demoted].double().mean():.3f} runner-up {a1[~demoted].double().mean():.3f} | "
          f"silenced: demoted {a2[demoted].double().mean():.3f} runner-up {a2[~demoted].double().mean():.3f}")

print("\nper neuron: mean (sd) contribution to the margin l_q - l_r and l_q - l_p, per case; mean activation h_10")
Wo_t = th.tensor(Wo, device=DEV)
c_qr = (Wo_t[q] - Wo_t[r]) * h
c_qp = (Wo_t[q] - Wo_t[p]) * h
for n in range(16):
    print(f"  n{n:2d}: l_q-l_r demoted {c_qr[demoted, n].mean():+6.1f} ({c_qr[demoted, n].std():5.1f}) runner-up {c_qr[~demoted, n].mean():+6.1f} ({c_qr[~demoted, n].std():5.1f}) | "
          f"l_q-l_p demoted {c_qp[demoted, n].mean():+6.1f} runner-up {c_qp[~demoted, n].mean():+6.1f} | h {h[:, n].mean():5.1f}")
print(f"  total margin l_q-l_r: demoted {c_qr[demoted].sum(1).mean():+.1f} runner-up {c_qr[~demoted].sum(1).mean():+.1f}; l_q-l_p: demoted {c_qp[demoted].sum(1).mean():+.1f} runner-up {c_qp[~demoted].sum(1).mean():+.1f}")
