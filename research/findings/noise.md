# Noise sensitivity and the suite

> W_hh is the fragile matrix: N(0, 0.002²) noise costs 2–11 points of accuracy, while W_hi and W_oh shrug it off.
> That fixes the benchmark noise at σ = 0.002 (ARC: "enough to significantly alter accuracy"). The suite spans 0.65–0.94, sd 0.065.
> Why W_hh: noise on a row shifts that neuron's pre-activation by ε·h, and the state is large and one-directional. Half of each row's damage lies along the state's principal direction, and the fragile seeds are the ones whose state runs 2.5× larger.

## Evidence
| Fact | Number | Raw output |
|---|---|---|
| Base accuracy example, s0, s1, s2, s3, s4 | 0.953, 0.910, 0.880, 0.949, 0.932, 0.890 | [noise_sensitivity](../../results/experiments/noise_sensitivity.txt) |
| σ = 0.002 noise, mean ± sd: example, s0, s1, s2, s3, s4 | 0.925±.016, 0.804±.038, 0.776±.053, 0.926±.009, 0.906±.009, 0.804±.031 | same |
| σ = 0.01 noise, mean: example | 0.64 | same |
| σ = 0.003 on W_hh only vs W_hi or W_oh only, example | 0.882 vs 0.953 | same |
| Suite (6 × 32 at σ = 0.002): mean, sd, min, max | 0.859, 0.065, 0.654, 0.945 | `python -m az.bench` (truth cache) |

## Where the damage sits
Noise on one row (a neuron's incoming recurrent weights) or one column at a time, 16 draws, 2^20 inputs. Raw output: [noise_fragility_map](../../results/experiments/noise_fragility_map.txt), [noise_fragile_direction](../../results/experiments/noise_fragile_direction.txt).

| Fact | Number |
|---|---|
| σ = 0.002 on all of W_hh, drop: example, s0, s1, s2, s3, s4 | 0.029, 0.111, 0.100, 0.020, 0.022, 0.073 |
| Example: the five core rows' share of the summed row damage; by block core←core / core←sat / sat←core / sat←sat | 0.030 of 0.036; 0.0135 / 0.0095 / 0.0030 / 0.0023 |
| Zoo seeds: largest single row's share of the summed row damage | 12–14% (diffuse) |
| Sum of single-row drops ÷ full-matrix drop | 1.1–1.8 |
| Per row, sd of the drop across draws ÷ mean drop (rows with drop > 0.003) | 0.7–1.2: a quadratic form of rank about 1–2 per row |
| A row's noise energy put along the state's PC1: damage ÷ isotropic full-row damage, six models | 5.8–8.1× (16× would be rank one) |
| Same along the state's PC2 / along W_hi | 1.0–3.2× / 0.6–5.0× |
| State variance share on PC1; RMS state norm √tr C over all steps: robust example, s2, s3 / fragile s0, s1, s4 | 0.75–0.78; 52–63 / 0.91–0.93; 129–146 |
| σ · √tr C, the per-step pre-activation bias a row's noise induces: robust / fragile | 0.10–0.13 / 0.26–0.29 |

So noise damage is bias sensitivity × state power along the noise, and the scale that matters is σ‖h‖ against the near-tie gaps that decide the answer ([errors.md](errors.md)). A per-neuron account of the damage exists only for the example, whose core rows carry it; the seeds spread it over every row, as they spread the core quantities ([circuit.md](circuit.md)).

## Noise as a smoothing of the margin, per input
To first order, isotropic noise on W_hh shifts an input's margin m (its predicted logit minus the best other) by N(0, σ²‖∂m/∂W_hh‖²), so P(flip | x) = Φ(−|m| / (σ‖∂m/∂W_hh‖)). Tested against 64 noisy copies on 2^14 inputs. Raw output: [noise_flip_model](../../results/experiments/noise_flip_model.txt).

| Fact | σ = 0.002 | σ = 0.005 |
|---|---|---|
| Predicted / empirical mean flip rate | 0.052 / 0.059 | 0.137 / 0.191 |
| Correlation of predicted and empirical flip probability across inputs | 0.935 | 0.871 |
| Share of flips on inputs predicted below 1% | 0.034 | 0.043 |
| Median \|m\|; median ‖∂m/∂W_hh‖; so the median σ‖∂m/∂W_hh‖ | 8.2; 1194; 2.4 | 6.0 |

At the challenge's σ the model is calibrated within about 20% in every bin and under-predicts slightly; at 0.005 second-order terms show. So the noisy suite's accuracy is the clean margin distribution smoothed by a per-input kernel whose width the weights give exactly, which is what lets any account of the clean model transfer to its noisy copies.

## Consequences
- Any estimator for the noisy suite must use W_hh exactly ([the estimator protocol](../../parked/README.md#the-rnns-estimator-protocol)).
- A constant guess of the suite mean has RMSE 0.065, which sampling beats from about 8×10^4 ops ([benchmarks.md](../../benchmarks.md)).
