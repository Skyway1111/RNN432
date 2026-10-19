# Tie-wedge integration

> Gap density integrated over designed inputs; built and scored: it loses to sampling at every budget, by a bias the lower magnitudes set.
> The two top gaps and the input mean are integrated exactly on a grid, with one reference configuration per (p, q, r) family. Even with all 720 families the estimate is +0.019 too high, and cheap designs are +0.05 to +0.08 too high, so no budget band is won.

**Status:** dead

## What was built
Inputs designed, not sampled: per (p, q, r) family, the seven lower values at the mean magnitude set in a fixed order, on a 12 × 12 grid over (M − S, S − T) and 7 values of the input mean, which are independent of the ordering. Bin masses are exact order-statistic facts. Estimate = 1 − Σ mass × share of families wrong. Scored on the 192-model suite. Raw output: [wedge_feasibility](../../results/experiments/wedge_feasibility.txt), [wedge_rest_variants](../../results/experiments/wedge_rest_variants.txt).

| Design | Families | Ops | RMSE | Bias | Sampling RMSE at those ops | mse/samp |
|---|---|---|---|---|---|---|
| Full grid 12 × 12 × 7 | 720 | 2.1e9 | 0.022 | +0.019 | 0.0004 | 3100 |
| Gaps only 12 × 12 | 720 | 3.0e8 | 0.026 | +0.024 | 0.0011 | 620 |
| Gaps 4 × 4 | 72 | 3.3e6 | 0.043 | +0.037 | 0.010 | 18 |
| S − T only, 4 points | 72 | 8.3e5 | 0.061 | +0.055 | 0.020 | 9.2 |
| S − T only, 4 points | 9 | 1.0e5 | 0.115 | +0.082 | 0.057 | 4.0 |

Every design loses to the constant 0.859 (RMSE 0.065) as well. The bias is positive everywhere: inputs with the lower values at their means are easier than Gaussian inputs.

## Why it died
The bias is the shape of the lower seven values. With all 720 families and the gaps-only grid: 4 orders of the lower values +0.020; three spreads about T (0.6, 1.0, 1.5 × the mean set's) equally weighted +0.018; spread 1.5 alone +0.002, RMSE 0.0125 at 3e8 ops where sampling has 0.0011. So the bias is a smooth function of a nuisance the design fixes, and removing it means integrating over the lower values' 7-D conditional distribution, which is the curse of dimension that sampling avoids. Picking the spread that zeroes the bias is fitting to the suite, not deduction.

## What this closes
Any estimator that integrates a few natural coordinates exactly and fixes the rest at representative values carries a bias of 0.02 to 0.08 on this suite, which is above sampling's RMSE at every budget above about 3e4 ops and above the constant's at every budget. Routes that remain: a weight-only closed form for the low band (unattempted, and at risk of being advice in disguise) and exact enumeration for the high band ([exact-accuracy](exact-accuracy.md)); advice plus the noise kernel for the middle band is measured dead ([advice-noise-kernel](advice-noise-kernel.md)).
