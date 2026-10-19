# Importance sampling over the failure modes

> Gap proposals per base and level with exact angular densities, orders learned in a first stage: above sampling from c = 2, rising with c.
> `az/estimators/modes.py` on `az/osp.py`. Each component is a mode of [tail.md](../findings/tail.md): orders (p, q, r), the 2nd largest's law, and the gaps to the max, to T and below T. Bench advantage over five realizations: 1.64 at c = 2, 92 at c = 8. The T gap carries most of it.
> Gaussians fitted to the same events gain with c too: in expectation, from each copy's exact mismatch, the structure's share is 4.3–4.7 through c = 6 and 7.0 at c = 8 on the bench suite, the rise all from s2's slab. Not an edge that grows with rarity.

**Status:** parked

## Idea
The tail's modes are regions of the top three's positions, the 2nd largest value S, and the gaps around it ([tail.md](../findings/tail.md)). A gap proposal draws the order (p, q, r) from π, S ~ N(μ, σ²), M = S + τ_M|z|, T = S − τ_T|z|, and each lower input T − τ_i|z_i| with its own scale per position. The order holds by construction, so along a ray the density is exp(C − A r² + B r). The event is scale-free, so the estimator weights directions, and the angular density is closed form: e^C (2A)^−5 K9(B/√(2A)), K9 tabulated once. Small τ_M or τ_T put mass on the M-end or T-end wedge, μ < 0 with small τ_T and τ_i on the all-negative mode, and a large τ_i on one position on a dominant negative input.

The library is structure found on the base, never on a suite copy. For each base, `events()` pools the events at c = 2, 4, 6, 8 of 16 fresh noisy copies (noise seeds 9000 + base index, 3,000 events per copy and level). `library()` fits K = 8, 4, 4, 4 components per level by EM. A copy is matched to its base by its weights. The estimate is importance sampling from 0.2 of the nominal law plus the base's components, with level shares 0.6, 0.2, 0.1, 0.1. The weights follow the balance heuristic, so every point serves every level. Each component gets a fixed count of points from a randomly shifted Kronecker lattice, through inverse normal CDFs.

Weight noise moves the modes between orders and keeps their shapes ([tail.md](../findings/tail.md)), so the run has two stages. The first 30% of every part's points re-estimate each component's order law from their soft events: Φ((margin − level)/1) times the LR and the component's responsibility, plus the same mass of the library's law. The other 70% are drawn with the new orders. Each stage is unbiased given the stages before it. A point costs 1.49 passes on average: the forward pass, 13 inverse CDFs, the shared gap statistics, 20 component densities, and in the first stage the responsibilities.

## Scores
One realization's score is a draw: a few copies' rare high-weight hits set the 192-copy RMSE. The score is therefore the harness's advantage, sampling's ops to 10% relative RMSE over these, as the geometric mean over seed salts 0 to 4 ([modes_score](../../results/experiments/modes_score.txt)). The bench suite is scored once, after the design was fixed on the held-out suite.

| Suite | c = 0 | c = 2 | c = 4 | c = 6 | c = 8 |
|---|---|---|---|---|---|
| held-out, geometric mean; range | 0.35; 0.30–0.39 | 1.68; 1.27–2.01 | 8.02; 7.58–8.49 | 27.8; 19.1–38.4 | 57.2; 18.5–110 |
| bench, scored once, geometric mean; range | 0.35; 0.32–0.38 | 1.64; 1.31–1.89 | 7.28; 5.99–9.21 | 34.3; 26.1–45.7 | 92.4; 74.5–107 |

Sampling's ops to 10% grow as 1/p, about 660-fold from c = 2 to c = 8 on the bench suite. These grow about 12-fold, so the advantage grows as the event gets rarer, in every realization.

## The copy's orders
The same five salts, paired, on the held-out suite ([modes_orders](../../results/experiments/modes_orders.txt)). Geometric means, with the per-salt range at c = 6:

| Orders | c = 2 | c = 4 | c = 6 | c = 8 |
|---|---|---|---|---|
| the library's | 1.69 | 7.52 | 17.0 (7.7–37.3) | 63.4 |
| learned from the first stage | 1.71 | 8.24 | 28.1 (19.1–38.4) | 59.3 |
| predicted from the weights, at the levels the budget covers tenfold | 1.61 | 7.38 | 22.7 (15.4–37.2) | 65.9 |
| both | 1.62 | 7.45 | 26.4 (18.3–32.7) | 57.3 |

Learning beats the library at c = 6 on all five salts and at c = 4 on four. Prediction weights the library's orders by exp(0.5 (f_copy − f_base)) at each component's mean point per order, 720 passes per component. It recovers a third of the log-distance to the oracle per component ([tail.md](../findings/tail.md)), but in the estimator it lifts only c = 6, and on top of learning it adds nothing, so the estimator does not carry it. Neither moves c = 2: there the first stage holds too few events for 720 orders, and the predictor would cost a third of the budget.

## What carries it
Each row changes one part of the design without the order stage, one realization on the held-out suite ([modes_design](../../results/experiments/modes_design.txt)):

| Change | c = 2 | c = 4 | c = 6 | c = 8 |
|---|---|---|---|---|
| none | 2.03 | 7.64 | 31.4 | 99.5 |
| T gap at the lower inputs' pooled scale | 0.51 | 1.41 | 5.51 | 8.62 |
| one lower-gap scale for all positions | 1.88 | 7.84 | 12.3 | 67.9 |
| independent draws instead of fixed counts and a lattice | 1.56 | 7.44 | 16.4 | 108 |
| K = 8 at every level | 1.68 | 9.36 | 22.9 | 110 |
| nominal share 0.1; 0.3 | 1.64; 1.88 | 7.79; 7.56 | 28.1; 31.9 | 104; 86.3 |

The T gap is the T-end wedge and the all-negative mode, where the lower inputs hug T. Without it the proposal is a two-gap law, and c = 8 drops elevenfold. A lower-gap scale per position is what lets one input dominate below zero. The other rows move by less than the spread of realizations.

## Against a generic proposal
`az/estimators/ce.py` keeps every part of the design but the components: full-covariance Gaussians fitted by EM to the same events, the cross-entropy method's family, with a first stage that re-learns each level's component weights. Its components per level and first stage were chosen on the held-out suite over three salts ([ce_score](../../results/experiments/ce_score.txt)). modes over ce, paired by salt 0 to 4:

| Suite | c = 0 | c = 2 | c = 4 | c = 6 | c = 8 |
|---|---|---|---|---|---|
| held-out, geometric mean; range | 1.59; 1.10–1.96 | 3.71; 3.12–4.28 | 4.63; 3.79–5.28 | 5.02; 3.32–7.83 | 3.95; 1.79–10.3 |
| bench, geometric mean; range | 1.57; 1.36–1.82 | 3.67; 2.93–4.75 | 5.20; 4.57–5.92 | 5.29; 4.03–9.26 | 5.36; 3.31–8.27 |

ce's own advantage grows from 0.45 at c = 2 to 17 at c = 8 on the bench suite ([benchmarks.md](../../benchmarks.md)), so most of modes' growth with c is importance sampling's. At c = 2 the factor of 3.7 is the difference between losing to sampling and beating it. More Gaussian components help at c = 8 and hurt at c = 2, where each component's density costs more per point than it covers.

A realization's score at c = 8 is set by single heavy draws on a few copies, so the ratio above is noisy there. The expectation has no such luck: each copy's exact mismatch with each estimator's library and learned mixtures, first stages included, gives its expected relative variance per point ([expected_score](../../results/experiments/expected_score.txt), within a factor of 1.8 of the errors realized over 64 salts). Expected ops to 10%, ce's over modes':

| Suite | c = 2 | c = 4 | c = 6 | c = 8 | s2's share of ce's expected error at c = 8 |
|---|---|---|---|---|---|
| held-out | 4.22 | 5.18 | 4.97 | 7.45 | 0.78 |
| bench | 4.25 | 4.70 | 4.45 | 6.97 | 0.70 |

The structure's share is flat through c = 6. It rises at c = 8 only on s2, whose tail is a slab across the top-two tie ([tail.md](../findings/tail.md)): a gap proposal follows the tie at every order, while a coordinate Gaussian must hold a thin axis per order. Averaged over 32 salts on the held-out suite, the realized ratio is 4.48, 5.14, 4.52, 5.75 at c = 2 to 8, and its c = 8 rests on single draws: the largest squared relative error of one copy in one realization is 99 for modes and 291 for ce.

## Limits
- **c = 2 is modest.** There the tail is the near-tie wedges at both ends, whose depth the lower inputs set through the margin at the tie. Noise moves them between orders, and the c = 2 budget, about 11,000 points at its 10% crossing, holds too few events to learn 720 orders.
- **A second tie mode sets the error at c = 8.** On the copies that dominate it, the heaviest tenth of the events, in a three-way tie (s2) or a near top-two tie (s1, s4) that no fresh copy showed, carries 59–96% of the variance. The copy's own orders lower it at most 2.7×, a refit on the copy's events 1.4–20×, and defensive changes at most 31% ([tail.md](../findings/tail.md), What a library misses).
- **Its gain is a property of the base's library.** A noise-created mode that no fresh copy showed is covered only by the nominal share.

## If resumed
Parked with stage 1, closed on the account ([plan.md](../../plan.md), Success criteria). The levers measured on the copies that set the error, each a constant factor on one class of copy ([heavy_modes](../../results/experiments/heavy_modes.txt)):
1. Learn the second mode's shape from the copy's own first-stage events, as a new component beside the library's rather than a refit of them, which collapses. A refit on the copy's events is worth 1.4–20× at c = 8.
2. A tie atlas, the copy's margins at every order's tie point, for the second modes it can see: 6–17× on s1/12, nothing on s2.
3. The exact S-line on short lines, a rule on the other inputs alone: 1.5–2× on s2 at c = 8 after its cost ([tail-lines](../../parked/research/angles/tail-lines.md)).
