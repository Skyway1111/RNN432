# Votes and tuning: the decision as exact coefficients on the inputs

> Per route the logits are exact sums of votes: each input votes for its own slot by its value, except the max, which votes against itself.
> Each neuron's coefficient on the max is a bump over the max's position, a different bump per neuron, and position alone explains 60–92% of it; the record history adds nothing. The second max has its own bumps.
> The two kinds of second max, a demoted max (q before p) and a runner-up (q after p), use the same neurons with opposite signs: every comparator's contribution to the deciding margin flips, while the net margin stays near +21.

## Method
`contributions` (`experiments/common.py`) returns P_t W_hi x_t for each step, the part of h_10 that input t contributes; the ten parts sum to h_10 exactly. Dividing by x_t gives the coefficient; W_oh times it gives the input's vote per unit for each position. Example model, Gaussian inputs. Raw output: [votes](../../results/experiments/votes.txt), [age_code](../../results/experiments/age_code.txt), [two_cases](../../results/experiments/two_cases.txt).

## Votes by rank, per unit input (mean, sd in brackets)
| Rank | For its own position | For q | For p | For r |
|---|---|---|---|---|
| 1 (the max) | −22 [36] | +9.5 | −22 | +1 |
| 2 (the 2nd max) | +35 [40] | +35 | +45 | −18 |
| 3 | +41 [43] | −16 | −4 | +41 |
| 4, 5, 6 | +27, +16, +10 | −8, −3.5, −1.6 | | |
| 7–10 | +6 to +7 [17] | −0.8 | | |

So l_q − l_r ≈ 53 S − 57 T + 8 M on average ([linearity.md](linearity.md) has the same from the margin side), and l_q − l_p ≈ 31 M − 10 S − 16 T: the max's slot is scored by how little it beats the 2nd max. The max's self-vote weakens from −24 at p = 0 to −9 at p = 7, then −28 at p = 8 and −51 at p = 9 (the increment neuron's veto). Positions explain only 10–13% of a self-vote's variance; the route sets the rest.

## What each neuron holds: coefficients by rank, and tuning over position
| Fact | Number |
|---|---|
| Running max n2: coefficients on M, S, T | 12.6, 4.1, 1.2 (sd 6, 6, 3) |
| Comparators n7, n6, n1, n8: on M; on ranks 6–10 summed | +7 to +9; −5 to −6.5 |
| Satellites n0, n10, n11, n12, n14: on M; on S; on T | +10 to +13 [10–14]; −0.4 to −3.6; −2.5 to −2.7 |
| n5, n9: fire only when the max or 2nd max sits at position 8 (coefficient 12.7, 2.4); n4 only at position 9 (10.1) | |
| Held-out variance of a neuron's coefficient on M explained by p alone; by p and the record history | 0.60–0.92 (n3, n9, n13: 0.07–0.24); +0.00 to +0.02 |
| Same for the coefficient on S by q; by (q, p) | 0.01–0.74; 0.25–0.80 |
| Bump centres of the coefficient on M over p: n12, n8, n15 / n0, n14, n1 / n6, n13 / n10, n11 (two bumps) | 0–2 / 3 / 5 / 0 and 7 |
| Valleys of the coefficient on S over q: n10, n11, n12 | 2–4, 3–5, 4–6 |

So the satellites and comparators are place cells for when the max and the 2nd max arrived, each tuned to an age, read by a linear decoder. The readout is tuned so that the position-explained parts combine into the robust votes above, and the route-dependent remainder is what the model errs on.

## The bumps are impulse responses
A single impulse x = e_p leaves every earlier state at zero, so its route is canonical and h_10 is the neuron profile of a unit max at age 10 − p, deduced from the weights alone. Raw output: [impulse_response](../../results/experiments/impulse_response.txt).

| Fact | Number |
|---|---|
| R² across positions between a neuron's impulse response and its measured coefficient on M: median; below 0.8 | 0.96; n3 0.38, n13 0.76 |
| Same for the 2nd max from x = 2e_p + e_q: median | 0.90 |
| The max's self-vote by p = 0 … 7 implied by the impulse responses; measured | −0.4 to −5; −24 to −9 |

The bumps come from the weights, but the readout cancels large terms, so a 10% context effect on the bumps becomes the whole of the max's self-suppression. A deductive account of the votes needs the typical route, not the single-impulse one.

## One reference route per (p, q)
Reference inputs: the mean magnitude set with M at p, S at q, and the eight lower values in 8 fixed orders. Each has an exact route and so an exact linear map from inputs to h_10. Raw output: [canonical_routes](../../results/experiments/canonical_routes.txt).

| Fact | Number |
|---|---|
| R² over the 90 pairs between the reference and the Gaussian-mean coefficient on M; on S (n3, n9, n13 excluded) | 0.92–0.99; 0.82–0.97 |
| Same for the max's and the 2nd max's self-votes | 0.07; −0.30 |
| Coefficient spread across the 8 reference orders vs across Gaussian inputs within a pair | 0.99 vs 3.62 |
| Surrogate "apply the pair's reference map to the input": agreement with the model; near a tie; elsewhere; its accuracy | 0.924; 0.849; 0.964; 0.957 |

So one route per (p, q) is the model on 92% of inputs, and the route's spread comes from magnitudes (gate flips at near-ties), not from the order of the lower values. The vote means do not transfer because a Gaussian average over routes is not the route of the average input.

## The two cases
| Fact | Demoted (q < p) | Runner-up (q > p) |
|---|---|---|
| Accuracy; by delay 1 → 9 | 0.955; 0.94 → 0.90 | 0.951; 0.96 → 0.91 |
| Worst (q, p) cells | (0, 9): 0.90 | (7, 0): 0.85 |
| n7, n8 mean contribution to l_q − l_r (sd ≈ 150, 130) | +30, −35 | −25, +38 |
| Readouts that matter mostly in one case (accuracy with that readout zeroed) | n15 0.49, n4 0.87 | n14 0.10, n8 0.17, n5 0.79 |
