# ADF: covariance propagation keyed on the label state

> Gaussian projections of the state smear exact zeros and exact linear relations, and the error compounds over 10 steps.
> ARC's canonical deduction-projection estimator, conditioned on where the running max and 2nd max sit. Scored in [benchmarks.md](../../../benchmarks.md).

**Status:** dead

## Idea
Per key (p, q), keep a mass and a Gaussian over [h, M, S]. Each step: split on x_t vs M and S with exact truncations, push through ReLU with exact pairwise moments, and moment-match keys that meet. At the end, compute P(argmax = q) by sequential truncation. Code: [adf.py](../../az/estimators/adf.py), primitives in [gauss.py](../../../az/gauss.py).

## Why it died
Key masses come out right (1/90 ± 10%), but the conditional state does not: neuron 4 is exactly 0 for key (0,1) under Monte Carlo, while ADF gives 0.24 ± 1.25, and ADF's sds are inflated up to 4×. Raw output: [adf_diagnosis](../../results/experiments/adf_diagnosis.txt).

## What would revive it
A representation that keeps exact zeros: see [spike-slab-mixture](spike-slab-mixture.md). A better ReLU moment formula won't help, because the representation is the problem.
