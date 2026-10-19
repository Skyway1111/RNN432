"""When during training do the core quantities form, as clean single neurons and as distributed codes, against accuracy?

Five zoo seeds, 12 checkpoints each (after 0, 2^20 … 2^30 training sequences; batch 1024, LR warmup ends at 2^24).
Finding: research/findings/training.md
"""
from az.model import accuracy
from az.zoo import SEEDS, SEQS, checkpoint
from experiments.common import ROLES, inputs, role_r2

TRACK = (0, 2, 3, 4, 6)  # max lagged, new-record, max-last, max\x[-2]-last, 2nd max
x = inputs(2**17, seed=21)
head = " ".join(f"{ROLES[i][:12]:>12}" for i in TRACK)
for seed in SEEDS:
    print(f"seed {seed}: columns = accuracy | best single-neuron R2 per quantity | decode-from-all-16 R2 per quantity")
    print(f"  {'seqs':>5} {'acc':>6} | {head} | {head}")
    for n in SEQS:
        w = checkpoint(seed, n)
        single = role_r2(w, x).max(0).values[list(TRACK)]
        dec = role_r2(w, x, decode=True)[list(TRACK)]
        tag = "0" if n == 0 else f"2^{n.bit_length() - 1}"
        print(f"  {tag:>5} {accuracy([w], 2**18, seed=3)[0]:6.3f} | " + " ".join(f"{v:12.3f}" for v in single.tolist())
              + " | " + " ".join(f"{v:12.3f}" for v in dec.tolist()))
