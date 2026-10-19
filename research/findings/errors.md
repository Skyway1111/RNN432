# Where the model errs

> Errors sit mostly near ties, yet depend on all ten inputs: low-dimensional features explain only about half of them.
> Inputs with at most one positive value are also mostly wrong. The example model errs on 4.7% of inputs: 64% of errors pick the 3rd-largest position, 20% the largest. Weight noise spreads errors away from ties.
> Even binned (tie gaps × top-3 positions × sign count, 141k bins), features explain only 47–68% of error variance. Any estimate that tracks only those features inherits the rest as bias.
> Inside a (family, T, M) cell, S carries half the indicator's variance and the largest lower input another tenth; at the tie, the two largest lower inputs' slots and values explain 55–71% of the margin's swing, the exact route itself about 60%.
> With the k largest lower inputs exact and the rest through one route, the tie decision's bias falls about 3× per added input; the lower inputs' interactions sit within three steps of each other and follow no rule of their 1-D effects.

## Evidence
Notation: [notation.md](../notation.md). Example model unless noted.

| Fact | Number | Raw output |
|---|---|---|
| Accuracy | 0.9533 | [accuracy_structure](../../results/experiments/accuracy_structure.txt) |
| Errors picking the 3rd-largest / the largest position | 64% / 20% | same |
| Accuracy with 0 / 1 positive inputs (1.1% of mass) | 0.13 / 0.23 | same |
| Accuracy with S−T in [0, 0.05) / [0.05, 0.1) / ≥ 0.5 | 0.77 / 0.94 / 0.99 | same |
| Accuracy with M−S in [0, 0.05) / ≥ 0.5 | 0.87 / 0.95 | same |
| Error rate near a tie (S−T or M−S < 0.1, 34% of mass) / elsewhere | 11.9% / 0.9% | [error_breadth](../../results/experiments/error_breadth.txt) |
| Same, three σ = 0.002 noisy copies | 16–27% / 2–6% | same |
| s0 clean / noisy: error rate near / far | 22% / 2.1%; 39–43% / 9–12% | same |
| Error variance explained by gaps; + (p,q); + (p,q,r); + sign count | 0.15; 0.24; 0.38; 0.53 | [error_set_dimension](../../results/experiments/error_set_dimension.txt) |
| Same for noisy example, clean s0, noisy s0 (all features) | 0.68, 0.47, 0.63 | same |
| Moving only S from T to M (s0/0, example/0, s4/0 copies): error share at the T end; lines with at most two correctness switches | 0.70, 0.84, 0.73; 0.99 | [sline_structure](../../results/experiments/sline_structure.txt) |
| Same: error mass on lines correct at both ends and wrong between, in accuracy; share of the error | 0.0029, 0.0028, 0.0031; 1.5–3% | [tent_law](../../results/experiments/tent_law.txt) |
| Inside a (family, record history) cell: midpoint of two error points is correct; activation patterns among a cell's error points, mean | 0.23–0.33; 134–193 | [error_cells](../../results/experiments/error_cells.txt) |
| Gate bits an input differs from its cell's majority route, mean; accuracy cost of forcing all of them to the majority | 6–10; 0.03–0.06 | [merge_ladder](../../results/experiments/merge_ladder.txt) |
| Twelve noisy copies: sd of the error rate across the 720 families; its share of the indicator's variance | 0.06–0.23; 0.04–0.28 | [family_variance](../../results/experiments/family_variance.txt) |
| Same: across-family variance explained by p, q, r main effects; the (p, q) pair; all position pairs | 0.21–0.55; 0.39–0.84; 0.62–0.94 | same |
| Inside a (family, T, M) cell, split-half share of the indicator's variance explained by the S bin (16); by S and the largest lower input's slot and value bin; the lower inputs' sign count adds (s0/0, example/0, s4/0) | 0.50, 0.57, 0.46; 0.58, 0.66, 0.59; +0.02 | [lower_ladder](../../results/experiments/lower_ladder.txt) |
| At the tie S = T, share of m_T's variance explained by the largest lower input's slot and value bin; by the two largest; by the full 160-bit gate pattern (split-half, a lower bound) | 0.42–0.57; 0.55–0.71; 0.50–0.59 | same |
| At the tie, P(wrong) per configuration from a Gaussian residual with oracle moments: no feature; the largest lower input exact; the two largest exact (1,344 cells per configuration), per-configuration rms | 0.040–0.053; 0.011–0.028; 0.006–0.017 | same |
| Variable gates at lower-input steps after the first of M, S, T: R² of the pre-activation on that step's own input, median; on all seven lower inputs linearly (s0/0, s4/0; example/0) | 0.97, 0.97; 0.005 / 0.99, 0.99; 0.89 | same |
| At the tie, the k largest lower inputs exact and the rest through the route at their conditional mean: P(m_T < 0) bias for k = 0, 1, 2, 3, 5 (s0/0; example/0; s4/0) | +0.052, +0.023, +0.008, +0.003, +0.0004; −0.017, −0.002, +0.0001, +0.0001, +0.0001; +0.061, +0.025, +0.009, +0.003, +0.0006 | [lower_ranks](../../results/experiments/lower_ranks.txt) |
| Same, with the lower inputs read before the first of M, S, T exact instead (six models): bias; share of draws whose sign it gets right | −0.008 to +0.025; 0.88–0.93 | same |
| Pairwise interaction of the lower inputs in m_T: share of its variance between slots 1, at most 2, at most 3 steps apart | 0.41–0.50, 0.63–0.70, 0.78–0.81 | same |
| Pairs whose 1-D effects both lower m_T: share with a positive pairwise term; its mean against the two effects' sum; R² of the term on the two effects | 0.57; +0.11 to +0.16 against −1.3 to −1.5; 0.002–0.019 | same |
| Confidently wrong at level c ([notation](../notation.md)), one copy per base: probability at c = 0, 2, 4, 6, 8 | 0.08–0.20; 1.6e-3–1.4e-2; 8e-5–2.1e-3; 1e-5–2.8e-4; 2.3e-6–4.8e-5 | [tail_structure](../../results/experiments/tail_structure.txt) |
| Events at c ≥ 4: share with at most one positive input (1.1% of mass); predicting the max; predicting rank 4 or lower; with the max at position 7–9 (30% of mass) | 0.00–0.90; 0.13–0.76; 0.03–0.59; 0.05–0.60 | same |

## Connections
- The rest of the error variance comes from the low ranks and from activation patterns: [linearity.md](linearity.md).
- This is why low-dimensional surrogates carry a bias: [low-dim-surrogates](../../parked/research/angles/low-dim-surrogates.md).
- Along S the error set is the two end wedges, so its integral is exact and cheap: [family-lattice](../../parked/research/angles/family-lattice.md).
- The confident-error tail by mode and level, and the ties that fix its margins: [tail.md](tail.md).
- For the zoo seeds a lower input's gates are thresholds on that input alone, so the route over the lower inputs is a chain of interval choices; the example's satellite gates read the accumulated state instead. What this bounds for an estimator: [requirements.md](../../parked/research/requirements.md).
