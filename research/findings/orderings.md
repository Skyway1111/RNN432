# Order structure versus magnitudes, exactly

> At typical magnitudes the example model is exactly right on 3,627,397 of 3,628,800 orderings; nearly all its error comes from magnitudes.
> Its 1,403 order-level errors all have the 2nd max first and the max last. Every base model shares that mechanism: a late max (position 7–9) demotes an early 2nd max and the model picks the 3rd largest. It is 1–34% of each base's error, and the suite's noise frays it.
> Gaussian accuracy is the average over magnitude sets y of acc(y), the exact accuracy over all orderings of y: median 0.997, with a heavy tail at near-ties and ≤ 1 positive.
> acc(y) is low-dimensional: gaps and sign count explain 81–85% of its variance. Near-ties, which hold the error, also depend on the order of the lower ranks.

## Method
i.i.d. inputs = sorted magnitudes y × a uniformly random ordering. `exact_acc` (`experiments/common.py`) walks the prefix tree of all 10! orderings: 6.2M recurrent steps, 2.3e9 ops per model and y. The smallest top-2 logit gap, at least 2.9e-11 against float64 rounding near 1e-12 for logits O(100), certifies that no decision flips. Raw output: [permutation_exact](../../results/experiments/permutation_exact.txt), [order_exact_magnitude_quadrature](../../results/experiments/order_exact_magnitude_quadrature.txt).

## Order structure at fixed magnitudes
| Fact | Number |
|---|---|
| E[sorted N(0,1)^10], the "mean" magnitude set | 1.539, 1.001, 0.656, 0.376, 0.123, then mirrored |
| acc(mean) for example, s0, s1, s2, s3, s4 | 0.9996, 0.9922, 0.9592, 0.9987, 0.9990, 0.9793 |
| Example at the mean set: wrong orderings; all have | 1,403 of 3,628,800; 2nd max at position 0, max at position 9, prediction = 3rd largest |
| All six bases at the mean set: share of wrong orderings with the max at 7–9; at 9; that pick the 3rd largest; (p, q) cells holding 90% of them ([order_errors](../../results/experiments/order_errors.txt)) | 0.92–1.00; 0.56–1.00; 0.72–1.00; 1–16 of 90 |
| Order-level error at the mean set as a share of the model's total error: example, s0, s1, s2, s3, s4 | 0.008, 0.087, 0.34, 0.026, 0.015, 0.19 |
| Eight suite copies at the mean set: order-level error against the base's; share with the max at 7–9; that pick the max; (p, q) cells for 90% | 0.6–6.6×; 0.37–0.97; 0.00–0.60; 3–20 |
| Over Gaussian inputs, share of all errors with the max at 7–9 (30% of mass); at 9 (10%): bases; copies | 0.32–0.54; 0.14–0.25 / 0.20–0.44; 0.08–0.18 |
| acc(mean + 1), mostly positive inputs: example, s0 | 0.9961, 0.9895 |
| acc(mean − 1), mostly negative inputs: example, s0, s2, s3 | 0.900, 0.804, 0.670, 0.658 |
| acc(mean with S−T shrunk to 0.05): example, s0 | 0.944, 0.855 |
| acc(y) over 400 random magnitude sets, example: mean, median, share < 0.9 | 0.948 ± 0.007 (truth 0.953), 0.997, 11% |
| Same, s0 | 0.914 ± 0.007 (truth 0.910), 0.979, 23% |
| Example, mean acc(y) when S−T < 0.1 (22% of sets) / otherwise / ≤ 1 positive (3%) | 0.867 / 0.971 / 0.27 |

## acc(y) as a function of magnitudes
Grid rows hold T and the ranks below it at the mean set. "Families" are the 720 (p, q, r) position triples, 5,040 orderings each.

| Fact | Number |
|---|---|
| acc at S−T = 0.01, 0.05, 0.1, 0.5, with M−S = 0.5: example; s0 | 0.73, 0.97, 0.99, 1.00; 0.62, 0.90, 0.97, 0.99 |
| Example at S−T = 0.01, as M−S grows 0.01, 0.1, 1.0: acc peaks, then falls | 0.60, 0.77, 0.69 |
| sd of acc when y_4..y_10 are redrawn given T, at S−T = 0.02 / 0.1 / 0.5 (six models, 32 draws) | 0.028–0.043 / 0.002–0.011 / 0.0002–0.006 |
| At S−T = 0.02: acc at the mean set minus mean acc over those draws (six models) | +0.005 to +0.020 |
| Families holding 90% of errors at S−T = 0.05 / 0.01: example; s0 | 62 / 272; 171 / 351 |
| At S−T 0.05, S−T 0.01, M−S 0.05 near-ties: share of errors in families ≥ 95% wrong (example, s0) | 0.08–0.37 |
| Same near-ties: \|acc − acc if each family took its majority decision\| | 0.003–0.040 |
| Held-out variance of acc(y) explained, 2,048 random sets: gaps / gaps + signs (example, s0, noisy example, noisy s0) | 0.41, 0.67, 0.62, 0.79 / 0.81, 0.84, 0.85, 0.85 |
| Residual sd of acc(y) after gaps + signs, same four models | 0.060, 0.062, 0.063, 0.072 |

## Meaning for the challenge
The order logic is solved exactly: the model's ordering behavior at typical magnitudes is a finite, enumerable fact. Averaging over orderings makes the error far more low-dimensional than per input, where the same bins explain 16–28% ([errors.md](errors.md)). The rest of acc(y) sits at near-ties, where the order of the lower ranks decides, so no table of 720 families and no plug-in of the mean set reproduces it. That parks [order-exact-magnitude-quadrature](../../parked/research/angles/order-exact-magnitude-quadrature.md).

The one shared mechanism, the late max demoting an early 2nd max, is an enrichment, not the error: the max's position is 7–9 for a third to a half of all errors against 30% of the mass, and 69–87% of errors are near a tie at any position. Under the suite's noise the order-level error grows 2–7× and loses the mechanism's shape: it spreads to 12–20 (p, q) cells, the max is late in only 37–97% of it, and the fragile copies pick the max as often as the 3rd largest, so the noise breaks the max's self-suppression ([votes.md](votes.md)) in copy-specific ways. The example's copies even err less at p = 9 than elsewhere.

Across training, the same method splits each checkpoint's error into order-level and magnitude-level parts: [training.md](training.md).
