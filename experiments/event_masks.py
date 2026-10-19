"""Does the record history (when the running max and 2nd max change) fix which neurons fire?

Finding: research/findings/linearity.md
"""
import torch as th

from az.model import load
from experiments.common import inputs, ranks

N = 2**21
w = load("example")
Wi, Wh, Wo = (th.tensor(a, device="cuda") for a in w)
x = inputs(N, seed=3)
h = th.zeros(N, 16, dtype=th.float64, device="cuda")
P = []
for t in range(10):
    z = h @ Wh.T + x[:, t : t + 1] * Wi
    P.append(z > 0)
    h = th.relu(z)
P = th.stack(P, 1)
q = ranks(x)[0][:, 1]
ok = (h @ Wo.T).argmax(-1) == q
M = th.full((N,), -1e9, dtype=th.float64, device="cuda")
S = M.clone()
ev = []
for t in range(10):
    newmax = x[:, t] > M
    new2 = ~newmax & (x[:, t] > S)
    ev.append(newmax.long() * 2 + new2.long())
    S = th.where(newmax, M, th.where(new2, x[:, t], S))
    M = th.where(newmax, x[:, t], M)
code = (th.stack(ev, 1) * th.tensor([3**i for i in range(10)], device="cuda")).sum(1)
vals, inv, cnts = th.unique(code, return_inverse=True, return_counts=True)
cs = (cnts.sort(descending=True).values.double().cumsum(0) / N).cpu().numpy()
print("distinct record histories:", len(vals))
for k in [1, 10, 100, 1000, 3000]:
    print(f"  top {k} cover {cs[k - 1]:.4f}")
sums = th.zeros(len(vals), 160, dtype=th.float64, device="cuda").index_add_(0, inv, P.reshape(N, -1).double())
maj = sums / cnts[:, None].double() > 0.5
agree = P.reshape(N, -1) == maj[inv]
print("mask bits matching the majority mask of their record history:", agree.double().mean().item())
print("per-neuron agreement:", agree.reshape(N, 10, 16).double().mean((0, 1)).cpu().numpy().round(3))
mm = maj[inv].reshape(N, 10, 16).double()
h = th.zeros(N, 16, dtype=th.float64, device="cuda")
for t in range(10):
    h = (h @ Wh.T + x[:, t : t + 1] * Wi) * mm[:, t]
pred2 = (h @ Wo.T).argmax(-1)
print("true accuracy", ok.double().mean().item(), " with canonical masks", (pred2 == q).double().mean().item())
