# The sixteen neurons, one by one

> Every neuron is essential: silencing any one leaves 11–67% accuracy of 95.3%, and the 20 costliest links each cost 61–85 points cut.
> The timers point to the runner-up: each lifts its own veto on one or two positions when the runner-up lands there, and n7 does it for 9.
> Per unit of the runner-up's value, the ten timers add +36 to +59 to its own position's score, and the core six subtract up to 30.
> A link's push, |weight| times its source's mean activity, ranks the 256 links as cutting them does (Spearman 0.89), so the push can stand in for it.

## Method
Example model, 2^20 Gaussian inputs, accuracies on 2^18 of them. Silencing zeroes a neuron's incoming weights, so it never
fires; a vote cut zeroes its readout column; a feed cut zeroes what it sends to the others. A link's push is the mean of
|W_hh[i, j] h_j| over inputs and steps. The runner-up's part of a neuron is its exact contribution to h_10 per unit of its
value, by its position q ([votes.md](votes.md), Method); its vote for q is that part times W_oh[q, n]. Raw output:
[neuron_atlas](../../results/experiments/neuron_atlas.txt); the link-cut table, `results/experiments/neuron_atlas.npy`.

## The core six
Roles: [circuit.md](circuit.md). Accuracy when silenced; the links whose cut costs most, with the points lost.

| Neuron | Role | Silenced | Reads (cut cost) | Feeds (cut cost) |
|---|---|---|---|---|
| n2 | the record, one step behind | 0.18 | n4 +1.73 (76), itself +0.97 (74) | n7 +0.67 (81), n6 +0.53 (81), n1 +0.50 (76), n8 +0.63 (73) |
| n4 | the jump past the record; vetoes position 9 | 0.18 | input +10.2 (78), n2 −0.49 (35) | n7 +1.32 (80), n2 +1.73 (76), n12 +1.28 (50), n11 +1.71 (45) |
| n7 | how far below the record | 0.11 | input −13.2 (83), n2 +0.67 (81), n4 +1.32 (80) | n8 −1.09 (85), n1 −1.05 (81), n11 +0.38 (62) |
| n6 | below the record with the previous input left out (R² 0.97, [role_probe](../../results/experiments/role_probe.txt)) | 0.14 | n2 +0.53 (81), input −11.0 (79), itself +0.36 (68) | n1 +1.00 (69), n14 +0.77 (66), n0 +1.13 (61) |
| n1 | below the record with the input two back left out (R² 0.93, same) | 0.15 | n7 −1.05 (81), input −10.6 (80), n2 +0.50 (76), n6 +1.00 (69) | n8 +0.63 (63), n12 +0.47 (58), n0 +0.71 (55) |
| n8 | below the runner-up so far | 0.17 | n7 −1.09 (85), input −12.3 (74), n2 +0.63 (73) | n3 −1.07 (67), n15 −0.72 (59), n0 −0.79 (51) |

## The ten timers
Each holds a share of the record that depends on when it arrived (its impulse response, [votes.md](votes.md); "held
most" is within 80% of its peak), and the
runner-up pushes it down or up by where it lands. "Points to" is the q where its vote for the runner-up's own position is
largest, with that vote per unit; a lifted veto votes against q and is pushed down there, an added vote the reverse.

| Neuron | Record held most at | Points to (vote) | How | Silenced | Strongest links (cut cost) |
|---|---|---|---|---|---|
| n14 | 2–4 | 0 (+33) | lifted veto | 0.19 | from n6 +0.77 (66), n2 +0.34 (61) |
| n15 | 1–2 | 1 (+19), 2 (+16) | added vote | 0.64 | from n10 −1.09 (75), n8 −0.72 (59) |
| n10 | 6–7 | 3 (+43), 2 (+25) | lifted veto | 0.13 | from n11 +1.10 (82); to n15 −1.09 (75) |
| n11 | 0–1, 7–8 | 4 (+51) | lifted veto | 0.11 | from n12 +0.66 (72), n7 +0.38 (62); to n10 (82) |
| n12 | 0–2 | 5 (+37), 6 (+18) | lifted veto | 0.13 | from n1 +0.47 (58), n14 +0.34 (53); to n11 (72) |
| n0 | 2–4 | 7 (+36) | lifted veto | 0.19 | from n6 +1.13 (61), n7 −0.78 (57); to n3 −1.23 (53) |
| n5 | 8 | 8 (+45) | added vote | 0.21 | from n4 +1.19 (35); to n0 −0.91 (55) |
| n3 | 0–2, weakly | 1 (+9), 6 (+7) | added vote | 0.55 | from n8 −1.07 (67), n0 −1.23 (53) |
| n13 | 4–6, weakly | 5 (+8) | added vote | 0.30 | from n2 +0.38 (31), n8 −0.29 (28) |
| n9 | 8, faintly | none: cutting its vote leaves 0.95 | — | 0.67 | from n7 −5.03 (10); to n1 −0.69 (20), n6 −0.64 (17) |

So the timers tile the runner-up's positions 0–8, a relay n12 → n11 → n10 → n15 among them, and the comparator n7
covers 9: a runner-up arriving last lowers how far below the record the last number fell, lifting n7's vote against 9
(+42 per unit).

| Vote for the runner-up's own position, per unit of its value, by q = 0 … 9 | |
|---|---|
| All sixteen | +34, +29, +26, +26, +31, +37, +40, +42, +45, +40 |
| The core six | −16, −30, −26, −19, −13, −2, +4, −4, −1, +41 |
| The ten timers | +50, +59, +52, +45, +44, +39, +36, +46, +46, −1 |

## Ablation
| Fact | Number |
|---|---|
| Silencing one neuron: accuracy, the 13 most essential; n3, n15, n9 | 0.11–0.30; 0.55, 0.64, 0.67 |
| Cutting only its vote: n4, n5, n9; the other thirteen | 0.91, 0.86, 0.95; 0.24–0.76 |
| Cutting only what it feeds: n3, n9, n15; the other thirteen | 0.66, 0.67, 0.68; 0.10–0.41 |
| The 20 links whose cut costs most: points lost; overlap with the 20 largest pushes | 61–85; 12 of 20 |

## Open
- Why each timer holds the record best at the ages it does: the reading angle of its weights on the comparators' rotating
  mode does not predict it (the comparators themselves sit about 80° apart on that mode, two steps of the clock).
