# Backward value sketches

> P(correct | the first t inputs) is exact for one input left, but near-binary in the state: no low-dimensional sketch holds it.
> A backward deduction-projection would carry V_t(h, running top two) from step 10 down. V_9 costs about two passes and matches a fine grid to 5e-4, but its best sketch, 91 features fitted on the true prefix law, leaves an rms error of 0.16–0.22.

**Status:** dead

## Idea
Given the state before the last input, each pre-activation is affine in that input, so the logits are affine between at most 16 kinks and the order breakpoints S and M. The correct set is then an interval per segment, and V_9 is exact. Steps further back integrate V over one more input, and a projection onto a few features of (h, M, S) per key would keep the cost down.

## Why it died
Raw output: [value_sketch](../../results/experiments/value_sketch.txt). Held-out rms error of per-key least-squares sketches, in probability units:

| Sketch | s0/0, t = 9 | s0/0, t = 8 | example/0, t = 9 | example/0, t = 8 |
|---|---|---|---|---|
| sd of V | 0.306 | 0.225 | 0.221 | 0.162 |
| Constant per key | 0.285 | 0.211 | 0.215 | 0.160 |
| Linear in all 16 principal components, M, S | 0.236 | 0.181 | 0.170 | 0.133 |
| Quadratic in 4 components, M, S | 0.252 | 0.202 | 0.178 | 0.155 |
| Quadratic in the ten logits, M, S (91 terms) | 0.222 | | 0.156 | |

Most prefixes have their answer settled, right or wrong, so V_t is near 0 or 1 with sharp boundaries in the 18-dimensional (h, M, S). Integrating one or two inputs smooths only one or two directions. A sketch would need an error near 0.001 to beat sampling at 1e8 ops; the measured ones are 150–200 times that, fitted on the true prefix law, which no deterministic design can do better than.

## What this closes
Backward value functions with a low-dimensional projection, at any step. The exact V_9 remains a cheap building block: see [exact-suffix](exact-suffix.md).
