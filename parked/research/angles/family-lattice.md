# Family-stratified lattice

> Stratified deterministic sampling over families and the top three's laws: 0.3–0.6 of random sampling's MSE on the accuracy control.
> `az/estimators/lattice.py`, one forward pass per node; `az/estimators/sweep.py` adds the last input's exact integral for a quarter of a pass and ties it within luck. They read nothing from the weights except by running them, and the same point set serves any model of the task, so their gain is a constant factor. Designed on a held-out suite; not yet scored on the rare-event yardstick.

**Status:** parked

## Idea
Input space factors exactly ([margin-gaussian](margin-gaussian.md)): the family (p, q, r), uniform over 720; T, the 3rd largest value; M given T; S given (T, M); and the seven lower inputs, i.i.d. N(0, 1) truncated above at T. The estimator enumerates the families and splits (T, M | T) into k_T × k_M strata of equal mass. One node per stratum takes its position in the stratum, S, and the lower inputs from Roberts' Kronecker sequence over the (family, stratum) index, through the exact inverse CDFs, and the model's correctness at the node is averaged. Below one node per family it uses the 90 families (p, q, L[p][q]) of an idempotent Latin square, which balance every pair of positions. No random numbers are drawn anywhere.

## Scores
On the bench suite, scored once at budgets fixed in advance: mse/samp 0.26–0.75 at eleven budgets from 2.7e5 to 3e10 ([benchmarks.md](../../../benchmarks.md)). Above 1e10 the RMSE meets the truth's own sampling error, about 8.5e-5, so those ratios are upper bounds.

The design was chosen on a held-out suite of the same six bases with other noise seeds ([lattice_validation](../../results/experiments/lattice_validation.txt)): there the estimator scores 0.32–0.68 from 2.7e5 to 1e9. Stratifying S as well ties the shipped design (0.41 each); strata graded toward low T or the S ends lose (0.56–0.60). Below one node per family, the Latin families score 0.49–0.51, while the family as one more lattice coordinate loses (1.04–1.26).

## What wins
At matched ops on the bench suite ([sline_ablation](../../results/experiments/sline_ablation.txt)):

| Variant | 1.1e8 ops | 7.9e8 ops |
|---|---|---|
| Plain lattice over all ten inputs | 1.28 | 1.11 |
| Families enumerated, strata, lattice (this estimator) | 0.44 | 0.32 |
| Same with the exact S-line integral at each node (`az/estimators/sline.py`) | 0.42 | 0.49 |

The gain is the exact structure: enumerating families and stratifying by the order statistics' laws. A lattice alone is about as good as sampling. Against a 2^30-input truth on twelve models both enumerated variants converge faster than sampling, slopes −0.60 and −0.64 against −0.5 ([sline_convergence](../../results/experiments/sline_convergence.txt)).

## The S-line
Hold everything but S and move it from T to M: the model is piecewise linear with about five pieces, and its error set is the two ends, the S−T and M−S wedges ([sline_structure](../../../results/experiments/sline_structure.txt)). So P(correct | everything but S) is exact for about six passes; `sline.py` does that at each node. It halves the indicator's variance but costs six passes, so at equal ops it ties the plain node.

| S-line fact | s0/0 | example/0 | s4/0 |
|---|---|---|---|
| Error at the T end, the M end | 0.70, 0.29 | 0.84, 0.14 | 0.73, 0.26 |
| Lines with 0, 1, 2, more than 2 correctness switches | 0.30, 0.41, 0.28, 0.01 | 0.28, 0.41, 0.30, 0.01 | 0.30, 0.41, 0.29, 0.01 |
| Exact integral against 2,000 nodes, max difference | 0.0007 | 0.0006 | 0.0005 |
| Passes per line, mean and 90th percentile | 4.9, 9.3 | 7.7, 14.5 | 4.4, 8.2 |

Tracking inward from the two ends until the first correctness switch, instead of the whole line, does not pay ([tent_law](../../../results/experiments/tent_law.txt), part 1): it costs 3.5–3.9 passes against 4.4–7.5, because the pieces are created late in the line's steps and the full tracking already shares the early ones; and it misses the lines that are correct at both ends and wrong between, 1.5–3% of the error mass, a bias of +0.0028 to +0.0031. A third of the lines need more than one piece from an end.

## Cheaper nodes: the last input's sweep, and prefix trees
Exact integration is cheap at the end of the sequence: given h_9 the last pre-activations are affine in x_9, so the logits are affine between at most 16 kinks and the correct set is an interval per segment, 0.27 of a pass on top of the nine-step prefix. `az/estimators/sweep.py` does that at each lattice root, whatever rank position 9 holds; it checks against 4,000 nodes per line to 2e-4. Prefix-sharing trees branch position 9, or 8 and 9, into stratified leaves under one prefix (leaves weighted exactly when the two are order-adjacent). Held-out suite, equal ops ([cheap_suffix](../../results/experiments/cheap_suffix.txt)):

| Variant | Passes per root | 1e7 | 1e8 | 1e9 |
|---|---|---|---|---|
| Lattice | 1.04 | 0.43 | 0.39 | 0.36 |
| Sweep | 1.27 | 0.40 | 0.40 (0.31 with (5, 6) strata) | 0.26 |
| Tree on 9, two or four leaves | 1.21, 1.51 | 0.39, 0.50 | 0.39, 0.37 | 0.34, 0.37 |
| Tree on 8 and 9, 2 × 2 | 1.66 | 0.67 | 0.47 | 0.38 |
| Tree on 8 with 9 swept | 1.81 | 0.45 | 0.45 | 0.41 |

The sweep removes about 45% of the variance for a quarter of a pass: 1.3× at 1e9 on the held-out suite, within the lattice's luck at 1e8, and on the bench suite a tie, 0.32–0.42 against the lattice's 0.26–0.52 over 3e7–1e10 ([benchmarks.md](../../../benchmarks.md)). The exact deduction is real and its gain is the size of a lattice shift. Trees give nothing: a shared step saves a tenth of a pass, and leaves under one prefix are correlated, so each costs more than it saves. Scale-freeness gives nothing either: with the prefix at r times its direction, the pair (r, x_9) integrates in closed form under the t_9 law at the sweep's cost, yet the scale carries almost no variance, 0.1022 to 0.1019 within families on s0/0 ([radial_sweep](../../results/experiments/radial_sweep.txt)). What this bounds, and what a solution must do instead: [requirements.md](../requirements.md).

## Lattice luck
With the lattice's offset replaced by 16 random Cranley-Patterson shifts, the shipped offset proves typical ([lattice_shifts](../../results/experiments/lattice_shifts.txt)):

| Budget | Bench suite: mean (range) | Held-out suite: mean (range) |
|---|---|---|
| 3e5 | 0.66 (0.47–1.02) | 0.65 (0.46–0.98) |
| 1e6 | 0.78 (0.60–1.03) | 0.75 (0.61–0.93) |
| 3e6 | 0.64 (0.35–1.41) | 0.63 (0.37–1.28) |
| 1e7 | 0.55 (0.43–0.86) | 0.51 (0.37–0.73) |
| 1e8 | 0.35 (0.27–0.49) | 0.35 (0.24–0.56) |
| 1e9 | 0.33 (0.26–0.39) | 0.32 (0.24–0.42) |
| 1e10 | 0.47 (0.36–0.60) | 0.48 (0.38–0.59) |

From 1e7 up every shift beats sampling on both suites; below, 1–2 of 16 lose. The S-line variant's lattice, chosen on the bench suite before this design, was luckier than typical ([sline_shifts](../../results/experiments/sline_shifts.txt)).

## Strict rules lose
Without a lattice the integral over the lower inputs keeps a bias. Gauss and graded rules in (T, M) with gauss_margin's 16-run design score mse/samp 11–795; the nearest strict variant, a 64-row design spread across families, scores 4.4, its RMSE stuck at 0.0011 ([sline_strict](../../results/experiments/sline_strict.txt)). The integrand turns sharply as the largest lower input nears T, and the other six need their spread too, so only point sets with a distinct, spread configuration at every node track it. Such a design is a Latin hypercube, a lattice again.

## Open
- Below 2.7e5 ops there are fewer nodes than the 90 Latin families, and no smaller design covers every (p, q) pair, which carries 39–84% of the across-family variance ([family_variance](../../../results/experiments/family_variance.txt)); that floor is structural.
- The constant: about 0.3–0.6 of sampling's MSE. More structure in the strata did not help; a cheaper exact integral per node might.
