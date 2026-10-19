# Strata on the failure modes

> Stratify on the modes' features with model-free masses, or sample one mode's exact stratum: a gain capped by the stratum's mass.
> The best partition per base reaches 2–5× at c = 2 and 9–64× at c = 8, split-half. The exact stratum "at most one positive, label 9" gives 937–1017× per sample on its two bases, but misses the 11% of example's c = 8 events outside it. Proposals that shrink with the level replace both: [mode-mixture](../../../research/angles/mode-mixture.md).

**Status:** parked

## Idea
Routes 2 and 3 of the rare-event brief. The modes of [tail.md](../../../research/findings/tail.md) are regions of a few features with masses that do not depend on the model: the order (p, q) of the top two, the number of positive inputs, and the direction's 2nd largest value S and tie gaps M − S, S − T. Stratify on them with Neyman allocation, or, inside one mode, sample its exact stratum. For the all-negative mode that is "at most one positive input, label at position 9", mass 11/10240, sampled exactly by sign patterns and an exchangeable swap.

## Measured
Raw output: [tail_strata](../../results/experiments/tail_strata.txt). Gain is split-half: allocation from one half of 2^28 inputs, cell rates from the other.

| Strata | c = 2 | c = 4 | c = 8 |
|---|---|---|---|
| (p, q), 90 cells | 1.1–1.5 | 1.3–3.7 | 2.4–7.9 |
| (p, q) × positive inputs | 1.3–2.0 | 1.7–9.7 | 2.6–64.4 |
| S × (M − S) × (S − T), 16 bins each | 1.6–3.8 | 2.4–8.9 | 2.7–15.5 |
| (p, q) × S × (M − S) | 2.0–5.3 | 5.7–34.0 | 4.4–41.7 |

The exact stratum with a 5% nominal share scores 1017× per sample at c = 8 on example/0 and 937× at c = 6 on s3/0, where it holds the tail. Its estimate on example sits at 0.89 of the truth: 11% of the c = 8 events lie outside the stratum, and the 5% share sees too few of them for the per-sample variance to show it.

## Why it is parked
A fixed stratum's gain is at most one over its mass, so it is flat in c once the mode holds the tail. The finer partitions grow with c until their cells run out of events, then fall (s3 at c = 8: 64 for (p, q) × positives, 4.5 for the finest). Both needs, a region that shrinks with the level and coverage of every mode, are what the gap proposals of [mode-mixture](../../../research/angles/mode-mixture.md) give, and that estimator holds these strata as special cases.
