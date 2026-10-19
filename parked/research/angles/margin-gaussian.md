# Gaussian margin over the lower inputs

> Given the family and top three magnitudes, the margin is Gaussian over the lower inputs; the estimator reaches RMSE 0.0021, not sampling.
> `az/estimators/gauss_margin.py` integrates (T, M − S, S − T) on a grid over all 720 families, with a 16-run design that places the largest lower input by its law and the cell's spread of the mean added to the variance.
> It cannot meet sampling at any budget, by measurement: the presumption removes only 30–41% of the indicator's variance, holds only on fine cells, and no family can stand for another, so its cost floor is about 1e10 ops.

**Status:** parked

## The chain it completes
The error set is a far part (whole failing orderings) and a near part (wedges around the tie surfaces). Region-by-region certification has the wrong rate on this model, so the near part must be integrated analytically over whole families of routes. That needs the wedge width as a function of few coordinates: the ages and top magnitudes, with the seven lower inputs summarized statistically. This angle tests that last link and builds the estimator on it.

## The presumption
Configurations: 720 families × a 6 × 6 grid over the two gaps, T at its mean; the lower inputs i.i.d. N(0,1) truncated above at T, their exact conditional law; 256 draws per configuration. Raw output: [margin_gaussianity](../../results/experiments/margin_gaussianity.txt), [margin_moments_design](../../results/experiments/margin_moments_design.txt).

| Near-tie error mass (mass-weighted over the grid) | example | s0 | s4 | example/0 | s0/0 | s4/0 |
|---|---|---|---|---|---|---|
| Empirical | 0.0293 | 0.0766 | 0.0965 | 0.0623 | 0.1885 | 0.1851 |
| Gaussian with the empirical moments | 0.0293 | 0.0781 | 0.0950 | 0.0622 | 0.1891 | 0.1834 |
| Gaussian, moments from one pass at the symmetric reference | 0.0319 | 0.0985 | 0.1186 | 0.0611 | 0.1888 | 0.1894 |
| Gaussian, moments from the 8-run Hadamard design | 0.0299 | 0.0772 | 0.0997 | 0.0595 | 0.1858 | 0.1881 |

One pass underestimates the sd by 10–40% because route flips add spread. Over all of gauss_margin's (7, 8) nodes, the presumption with the empirical moments errs by 0.0007 RMS in accuracy over six models ([design_variants](../../results/experiments/design_variants.txt)).

## The estimator
Cells of a (T, g1, g2) grid with exact masses, mass-weighted nodes and covariances; all 720 families; 16 forward passes per cell and family. P(wrong) = Φ(−μ / √(sd² + a′Ca)), where a is μ's gradient between neighbouring nodes and C the cell's covariance. Suite scores: [benchmarks.md](../../../benchmarks.md).

Where the error sits. The split is at level (12, 12) without the spread term, 12 models ([gauss_margin_bias](../../results/experiments/gauss_margin_bias.txt)); the spread term then removes most of the grid's part.

| Part of the error | Mean | Per model | What addresses it |
|---|---|---|---|
| Grid: Φ at the node ignores μ's spread across the cell | +0.0017 | +0.0013 to +0.0022 | the a′Ca term: against a fine grid, +0.0020 → +0.0003 at (9, 10) and +0.0012 → +0.0001 at (12, 12); level (5, 6) overshoots to −0.0067 ([cell_spread](../../results/experiments/cell_spread.txt)) |
| Presumption | 0.0000 | −0.0008 to +0.0010 | none |
| Design moments | +0.0011 | −0.0004 to +0.0042 | the order-statistic design; at the (7, 8) nodes, six models, H8 errs +0.0015 mean and 0.0026 RMS, this design +0.0007 and 0.0012 ([margin_design_mechanism](../../results/experiments/margin_design_mechanism.txt)) |

## Why the 8-run design missed the spread
The design's mean is right, but its sd runs 4–13% low, and no central design fixes that: the full 2^7 factorial, an OA(27, 7, 3, 2) and the 3^7 Gauss tensor rule all err +0.0035 to +0.0041, worse than H8. Korobov lattices reach the empirical moments only at 31–61 runs ([design_variants](../../results/experiments/design_variants.txt)). The cause is the largest lower input ([margin_design_mechanism](../../results/experiments/margin_design_mechanism.txt)):

| Fact | Number |
|---|---|
| Share of the margin's variance over the lower inputs that is linear in them; additive (s0/0; example/0) | 0.56, 0.80; 0.44, 0.75 |
| H8's median sd over the empirical sd, lower inputs truncated at T, T − 0.15, T − 0.4 (s0/0; example/0) | 0.86, 0.93, 0.96; 0.87, 0.91, 0.93 |
| H8's accuracy error on example/0 at the same truncations | +0.0050, +0.0011, +0.0004 |

H8 puts every lower input at its mean ± sd, so the largest sits about 0.36 below T. When it comes within about 0.1 of T, the margin drops and that input's own position often becomes the competitor. The 16-run design puts the largest lower input at 16 quantiles of its law (Φ(v) / Φ(T))^7, one slot per run in turn, and the other six at mean ± sd below it.

The presumption's own floor is the law's shape. With the draws' moments, the Gaussian errs 0.00067 RMS over six models, an Edgeworth expansion with the skewness 0.00030, and with the kurtosis too 0.00015. With the 16- or 64-run design's moments the expansion gains little, 0.0012–0.0013 to 0.0008–0.0010, because the design's moments are the larger error ([margin_higher_moments](../../results/experiments/margin_higher_moments.txt)).

## Why it cannot reach the bar
Sampling's MSE at C ops is about 346 / C in accuracy² (p(1 − p) ≈ 0.12, 2880 ops per input). The estimator's error falls with cells about as sampling's falls with ops, from a constant 300–800 times worse. Each way to share work was measured ([sharing_limits](../../results/experiments/sharing_limits.txt), [gap_plane_sharing](../../results/experiments/gap_plane_sharing.txt), [fold_cells](../../results/experiments/fold_cells.txt)):

| Limit | Number |
|---|---|
| Share of the indicator's variance inside (family, node), the most the presumption can remove (six models) | 0.30–0.41 |
| Oracle presumption over larger cells: accuracy error at 1, 9, 25, 75, 448 cells per family | −0.024, −0.0085, −0.0057, −0.0039, −0.0010 |
| The same with a mean linear in (T, g1, g2) per cell, at 1, 9, 25, 75 cells | −0.016, −0.0066, −0.0046, −0.0027 |
| Scale-free cells, gaps over \|T\| and the margin over \|T\|, 128 per family | −0.061 |
| A balanced 90-family Latin subset against all 720, on the true per-family rates, RMS | 0.0053 |
| Each run's logit differences interpolated from 25 gap nodes against a 24 × 24 lattice, RMS | 0.0022 |
| The fold (each cell's (T, g) spread as three more design factors, 12 runs), 3 × 3 cells, Latin 90 families | RMSE 0.0074 at 2.8e7 ops, mse/samp 4.5 |
| The fold with 5 × 5 cells, all 720 families | bias +0.0062, mse/samp 90 |

So with 16 runs per node, the quadrature over (family, T, g) would have to beat Monte Carlo by about 10 times in variance just to tie, and it cannot coarsen: the presumption fails on any cell much larger than the (7, 8) grid's. The fold comes nearest only by cancellation. Over 3 × 3 cells its presumption errs −0.0085 and its design about +0.011, and refining the cells undoes the balance.

## Do not repeat
Sharing the gap plane by interpolation, central designs over the lower inputs, the presumption or the fold on coarse cells, scale-free cells, Latin family subsets. Each is measured above. A presumption keyed on route features (the two largest lower inputs' slots and values, the sign count, the record history) is measured too: it reaches a per-configuration rms of 0.006–0.017 only with oracle moments in 1,344 cells per configuration ([errors.md](../../../research/findings/errors.md), [requirements.md](../requirements.md)).

## The wedge law, tested
The revival named here was a closed-form law for the wedge widths over the lower inputs, so that one evaluation covers a coarse cell. Measured at the T end: the tie margin m_T (l_q minus the best other logit at S = T) and its slope, 512 configurations (family, T, M) with 1,024 draws of the lower inputs each. Raw output: [tent_law](../../../results/experiments/tent_law.txt).

| Fact | s0/0 | example/0 | s4/0 |
|---|---|---|---|
| Error mass in interior wrong intervals of S-lines, which no end-wedge law sees | 0.0029 | 0.0028 | 0.0031 |
| First-order wedge max(0, −m_T / slope) minus the exact wedge, mass over the line | +0.023 | +0.045 | −0.001 |
| Draws whose margin falls as S rises from the tie; falling and wrong | 0.035; 0.018 | 0.044; 0.020 | 0.061; 0.029 |
| P(m_T < 0): Gaussian with the draws' moments, bias and per-configuration rms | +0.004, 0.053 | +0.007, 0.047 | +0.007, 0.045 |
| Same with the 16-run design's moments | +0.003, 0.071 | +0.002, 0.055 | +0.007, 0.067 |
| Largest lower input exact (slot and 16 value bins), Gaussian residual per bin, oracle moments | +0.001, 0.016 | +0.001, 0.010 | +0.001, 0.012 |
| Skewness of m_T over the lower inputs; share of its variance the largest lower input explains | −0.79; 0.49 | −0.71; 0.58 | −0.65; 0.53 |

So the wedge is not the first linear piece's margin over its slope, the tie margin is not Gaussian over the lower inputs (per-configuration rms 0.05 on a probability near 0.5), and the composition the brief names, the largest lower input exact with a Gaussian for the rest, is three to four times better only with oracle moments from a thousand draws per configuration. Nor can the cells coarsen: with S integrated exactly and the lower inputs on the Kronecker sequence, putting (T, M) at their strata's medians instead of spreading them biases the estimate by +0.019 at 9 and 36 strata per family ([sline_structure](../../../results/experiments/sline_structure.txt), part 3). Closed.
