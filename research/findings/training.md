# How the circuit forms during training

> The skeleton is done by 2^24 sequences, at about 75% accuracy; the last 14–22 points come over the remaining 98% of training.
> All five zoo seeds go through the same three phases. So whatever explains the error set ([errors.md](errors.md)) was learned in phase 3, which is where to look.

## Evidence
Zoo seeds s0–s4, 12 checkpoints each; batch 1024, so 2^20 sequences = 1024 steps; LR warmup ends at 2^24. "Decode" = R² of reading the quantity linearly from all 16 neurons; "single" = best one-neuron fit ([role_probe](../../experiments/role_probe.py) method). Raw output: [core_emergence](../../results/experiments/core_emergence.txt).

| Phase | Fact | s0 | s1 | s2 | s3 | s4 |
|---|---|---|---|---|---|---|
| 1 | Accuracy at 2^22 (chance 0.1) | 0.421 | 0.533 | 0.441 | 0.455 | 0.496 |
| 1→2 | Running max, decode R² at 2^22 → 2^24 | .76→.97 | .80→.98 | .86→.99 | .71→.99 | .83→.98 |
| 2 | New-record, single R² at 2^22 → 2^23 | −.60→.90 | .75→.88 | −.52→.86 | never > .62 | .42→.89 |
| 2 | Accuracy at 2^24 | 0.752 | 0.744 | 0.726 | 0.765 | 0.741 |
| 3 | Accuracy at 2^30 | 0.910 | 0.881 | 0.949 | 0.932 | 0.890 |
| 3 | Largest change in core decode R² from 2^24 to 2^30 | .00 | .02 | .17 (switch) | .01 | .01 |

- **Seed 2 switches strategy at 2^25:** "max − last" decode drops 0.97 → 0.80 while "2nd max" rises 0.68 → 0.91. It finishes as the best seed.
- **Single neurons sharpen through phase 3 in s0:** "max − last" 0.73 → 0.91 and "max\x[-2] − last" 0.59 → 0.86.

## What phase 3 changes
- **No part can be swapped.** Splicing any 2^30 block (W_hi, W_hh, W_oh), or a single neuron's 17 incoming weights, into the 2^24 model drops accuracy far below both endpoints, and the same holds in reverse. Neuron-scale normalization doesn't change this. Raw output: [phase3_splice](../../results/experiments/phase3_splice.txt).
- **No fair straight path either.** Even between adjacent checkpoints the straight line rises 0.3–1.2 in loss above both ends, so gradient-path attribution is withheld. With neuron scales normalized, phase 3 moves W_hi by 0.15–0.7 of its norm, W_hh by 1.2–10.7, and W_oh by 3.2–8.4. Raw output: [phase3_path](../../results/experiments/phase3_path.txt).
- **It fixes order logic and magnitude handling both.** Exact accuracy over all orderings of the mean magnitude set ([orderings.md](orderings.md)), 2^24 → 2^30, against Gaussian accuracy. Raw output: [permutation_exact](../../results/experiments/permutation_exact.txt).

| | s0 | s1 | s2 | s3 | s4 |
|---|---|---|---|---|---|
| Ordering-exact at mean magnitudes | .885→.992 | .921→.959 | .875→.999 | .923→.999 | .872→.979 |
| Gaussian accuracy | .752→.910 | .744→.881 | .726→.949 | .765→.932 | .741→.890 |

So phase 3 is a coordinated rewrite of recurrence and readout around a fixed input layer, not an added part. Localizing it needs checkpoints close enough to join by fair paths (dense retraining, [training-dynamics](../../parked/research/angles/training-dynamics.md)).

## Open
- What gets phase 1 to 4–5× chance before any core quantity is readable.
