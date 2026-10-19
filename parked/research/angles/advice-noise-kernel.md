# Advice from the base model plus the noise kernel

> Reuse work done on a copy's base model and pay per copy only for the noise's effect: no cheap per-copy form follows the copy.
> Rerunning gauss_margin on the copy follows the true change to 0.0006–0.0022, but costs 1.5e10 ops. A second-order Taylor expansion in the weights misses by 0.008–0.098, and the noise shifts margins about twice the design's sd, so 76–97% of the work would rerun anyway.

**Status:** dead

## Idea
Advice is anything computed deductively from a base model's weights, at any cost, reused for its 32 copies; only the per-copy work counts. Noise of σ = 0.002 smooths each input's margin by a width the weights give ([noise.md](../../../research/findings/noise.md)), so a copy's accuracy might follow from the base by a cheap correction. It also needs the estimator to know a copy's base, which is close to fitting the suite.

## Why it died
gauss_margin at level (7, 8) on example, s0 and s4 and their 32 copies each. Raw output: [advice_tracking](../../results/experiments/advice_tracking.txt).

| Fact | example | s0 | s4 |
|---|---|---|---|
| True change, copy minus base: mean, sd | −0.027, 0.014 | −0.110, 0.035 | −0.079, 0.021 |
| Rerun on the copy, error in following the change, rms | 0.0006 | 0.0022 | 0.0020 |
| Second-order Taylor at the base, error against the rerun, rms | 0.0081 | 0.099 | 0.030 |
| Noise shift 0.002 ‖∂μ/∂W‖ against the design's sd, medians | 1.53 vs 0.83 | 1.77 vs 0.67 | 1.33 vs 0.65 |
| (Cell, family) pairs within 3 and 5 sd of the boundary, which must rerun | 0.76, 0.88 | 0.90, 0.97 | 0.87, 0.95 |

The noise moves accuracy by 3–11 points and touches nearly every pair, so neither a low-order expansion nor a selective rerun is cheap. The rerun's good tracking says the estimator's error is mostly shared within a base, but that does not lower its cost.

## What this closes
The middle-band route that [tie-wedge](tie-wedge.md) left open.
