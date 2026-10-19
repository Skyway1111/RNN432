# What changing a weight does

> Accuracy hangs on a few recurrent weights into the comparators and the running max: a step of 0.02 on W_hh[7,2] costs 20 points.
> The reference routes predict the ranking of all 432 weights by their effect (Spearman 0.96, 26 of the top 30) from 270 inputs and a gradient, with no probing.
> No single weight can be moved by 0.1 to gain more than 0.002: the model is at a coordinate-wise optimum.

## Ground truth
Every weight stepped by ±0.02, ±0.1, ±0.5, all copies on the same 2^20 inputs. Table in [weight_whatif.npy](../../results/experiments/weight_whatif.npy), text in [weight_whatif](../../results/experiments/weight_whatif.txt). Example model, base accuracy 0.9533.

| Fact | Number |
|---|---|
| Mean drop per weight at ±0.1: W_hi, W_hh, W_oh | 0.002, 0.115, 0.002 |
| Mean drop per weight at ±0.5: W_hi, W_hh, W_oh | 0.034, 0.38, 0.046 |
| Drop at ±0.02 on W_hh[7,2] (comparator 7 from the running max, 0.674); on W_hh[7,7]; on W_hh[6,2] | 0.19; 0.14; 0.08 |
| Rows holding the 30 most sensitive weights | 7, 6, 1, 8 (the comparators) and 2 (the running max); all from columns 2, 7, 6, 8, 1, 10, 14, 11, 0 |
| Share of the total effect at ±0.1 in the top 10 / 30 / 100 weights | 0.16 / 0.41 / 0.82 |

Why W_hh[7,2]: comparator 7 computes W_hh[7,2]·n2 − x_t with n2 ≈ the running max, so its zero shifts by 3% of the max per 0.02, which is the size of a near-tie gap ([errors.md](errors.md)).

## Predicting it from the reference routes
270 reference inputs (one per (p, q) at the mean magnitude set, and at S − T = 0.05 and M − S = 0.05), each with its margin and the margin's gradient over all 432 weights. Raw output: [margin_sensitivity](../../results/experiments/margin_sensitivity.txt).

| Predictor | Spearman with the ground truth at ±0.1 / ±0.02 | Top-30 overlap |
|---|---|---|
| Mean relative margin sensitivity \|∂m/∂w\| / \|m\| | 0.963 / 0.854 | 26 |
| Linearized share of reference inputs that flip at 0.1 | 0.944 / 0.872 | 27 |
| Exact share that flip when the weight is stepped | 0.942 / 0.875 | 27 |
| Sampled smoothed accuracy gradient on 2^16 Gaussian inputs (not deductive) | 0.881 / 0.791 | 16 |

Within W_hh the deductive predictors reach 0.96–0.98; within W_hi and W_oh, whose effects are tiny, 0.2–0.8. Within W_oh the curvature of the readout's optimum ranks them at 0.93–0.96 ([circuit.md](circuit.md)). The size of a step's effect follows from the same gradients through the noise model in [noise.md](noise.md).
