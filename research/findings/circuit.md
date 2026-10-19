# Circuit of the example model

> A six-neuron core (running max, increment, four phase-wheel comparators) drives ten satellites; the satellites carry the decision.
> In canonical form the core takes ≤ 0.23 from the satellites, but satellites run at O(10–30). Cutting that feedback drops accuracy to 51%, and cutting their readout drops it to chance.
> Every zoo seed computes the same core quantities, but mostly spread across neurons: only the example stores them in clean single neurons.
> The readout follows from W_hi and W_hh: it is the cross-entropy-optimal decoder of the state (R² 0.985), and its top plane is a clock because the comparators' block of W_hh turns the state 42° per step, so positions 0–7 fill the circle. On Gaussian inputs the decision still uses all ten directions.
> Near a tie the decision is a cascade: gates as fixed thresholds on their own input get 89% of the seeds' decisions, and each round of gates retuning later gates adds the rest over three to five rounds; the example's satellites read the state, so no fixed-threshold account exists for it.

## Evidence
Canonical form: neurons with |W_hi| > 0.5 rescaled to W_hi = ±1 ([notation.md](../notation.md)). Raw matrices: [canonical_form](../../results/experiments/canonical_form.txt).

| Part | Neurons | Role | Evidence |
|---|---|---|---|
| Running max | 2 | ≈ max(0, x_0…x_{t−2}) | ARC ([prior-work.md](../prior-work.md)) |
| Increment | 4 (input +1) | ≈ new max minus old; votes −51 against position 9 | canonical form |
| Comparators | 7, 6, 1, 8 (input −1) | leave-one-out max minus last input; readout rows are phase-shifted sinusoids over position (n7: −37, 6, 62, 96, 77, 12, −38, −71, −66, −50) | canonical form; ARC; Keane |
| Satellites | 0, 3, 5, 9–15 | driven by the core with weights 2–66; readout weights 1–7 | canonical form |

| Ablation | Accuracy | Raw output |
|---|---|---|
| Full model | 0.953 | [core_ablation](../../results/experiments/core_ablation.txt) |
| Zero satellite → core weights (all ≤ 0.23 canonical) | 0.508 | same |
| Zero satellite readout | 0.112 | same |
| Core only | 0.103 | same |

W_hh eigenvalue moduli 1.27, 1.17, 1.12 (complex pair), 0.90 (complex pair), …: expanding and rotating modes.

## The zoo seeds
A probe fits each neuron against candidate quantities and finds roles without knowing indices; on the example it recovers ARC's five neurons (R² 0.93–0.99). Raw output: [role_probe](../../results/experiments/role_probe.txt).

| Model | Clean single neurons (R² > 0.9) | Running max, decode R² | "max − last", decode R² | 2nd max, decode R² |
|---|---|---|---|---|
| example | 5 (new-record, 3 comparators, running max) | 0.998 | 0.999 | 0.91 |
| s0, s1, s4 | 1–3 (new-record, comparators) | 0.97–1.00 | 0.99–1.00 | 0.68–0.78 |
| s2 | 1 (new-record, R² 1.000) | 1.000 | 0.87 | 0.96 |
| s3 | 4 (comparators, last input; no new-record) | 0.995 | 0.99 | 0.74 |

So legibility at the single-neuron level is a property of this one model, not of the task. s2 relies on a running 2nd max instead of comparators ([training.md](training.md) shows when it switched).

## The phase wheel
The top two singular directions of W_oh span a plane in state space; each position's readout row has an angle in it, and so does h_10. Two-impulse inputs are Keane's x_p = 2, x_q = 1, rest 0 ([prior-work.md](../prior-work.md)). Raw output: [readout_wheel](../../results/experiments/readout_wheel.txt).

| Fact | Number |
|---|---|
| Share of ‖W_oh‖² in the top plane: example, s0, s1, s2, s3, s4 | 0.78, 0.81, 0.72, 0.84, 0.74, 0.85 |
| Example: readout angle step per position, 0 → 7; then 7 → 9 | −46, −40, −52, −42, −41, −45, −53; +17, +9 |
| s0, s3, s4: step per position (a near-flip with drift); s1 alternates ±150; s2 has no clean wheel | −147 to −175 |
| Two-impulse inputs, mean \|state angle − readout angle of q\|: example, s0, s1, s2, s3, s4 | 22°, 34°, 40°, 52°, 31°, 25° |
| Two-impulse accuracy, same order | 1.00, 0.82, 0.83, 1.00, 1.00, 0.99 |

On two-impulse inputs, then, position is a phase read by matched filters. On Gaussian inputs it is not the decision. Raw output: [phase_plane](../../results/experiments/phase_plane.txt).

| Fact | Number |
|---|---|
| Agreement with the full model when W_oh is truncated to its top 1 / 2 / 3 / 4 singular directions, example | 0.20 / 0.37 / 0.45 / 0.46 (zoo seeds 0.15–0.49) |
| Plane vector of h_10 by exact per-input contribution: mean size of the 2nd max's part; of the other nine; of the sum | 5.5; 4.3; 2.5 |
| The other nine inputs' part along the 2nd max's: mean | −3.8 (they cancel most of it) |
| Variance of a step's push per unit input explained by its offset from q; by offset, side of p, and rank | 0.01; 0.14 |

So the plane vector is a small residual of large cancellations, and the decision uses all ten readout directions: the satellites, read at 1–7 but running at O(10–30), carry as much of every logit as the comparators do. Positions 8 and 9 fall off the wheel even on two-impulse inputs and lean on the increment neuron's vote.

## Why the readout takes its shape
Deduced from W_hi and W_hh, then checked against W_oh. Raw output: [readout_derivation](../../results/experiments/readout_derivation.txt).

| Fact | Number |
|---|---|
| The core block of W_hh alone (n2, n7, n6, n1, n8): eigenvalues | 0.96 at 0°; 0.93 at ±42.1° |
| Two-impulse inputs, each gate pattern after both impulses: angle of its largest eigenvalue other than the positive reals, step-weighted median; share of steps within 10° of it | 46.1°; 0.82 |
| Readout rows' angle step per position 0 → 7, median and range; the eight steps with the closing one, 7 → 0, sum to | 44.5°, 40–53°; 360° |
| The commonest pattern with one gate shut: a comparator (n1, n6, n7, n8); any satellite | 35–46° at modulus 0.52–0.85; 42–49° |
| Core gates settled (n2 and the comparators on, n4 off): at the later impulse's own step; one step after | 0.16; 1.00 |
| The cross-entropy-optimal readout of h_10, fitted without W_oh, against W_oh with rows centered: per-row cosine; R² | ≥ 0.994; 0.985 |
| Accuracy with W_oh; with the derived readout | 0.953; 0.957 |
| A step ε on W_oh[j, n]: Spearman of ε²/2 · E[p_j (1 − p_j) h_n²] with the 160 measured accuracy drops at ±0.1 and ±0.5 ([whatif](whatif.md)); drop at 0.5 over drop at 0.1, median | 0.93–0.96; 21 (quadratic: 25) |
| Zoo seeds, the derived readout: per-row cosine; R² | ≥ 0.984; 0.86–0.99 |
| Zoo seeds, the turning angle against the readout step: s0, s3, s4; s1; s2 | 168° vs 159°, 153° vs 163°, 166° vs 168°; 180°, a flip, vs 136°; 110° with no wheel |

So the clock is the comparators. Their block of W_hh turns the state 42° per step, and the satellites carry the turning state as rectified place cells without changing its angle. Eight steps of 45° close the circle, so the wheel holds positions 0–7, equally spaced. Position 8 would land on position 0's phase, and position 9 arrives before the core's gates settle, so each has its own neurons: n5 and n9 for position 8, n4 for position 9 ([votes.md](votes.md)).

The rest of each row is the decoder. Given the states the recurrence computes, the cross-entropy-optimal linear readout is unique up to a common shift of its rows, and W_oh lies within R² 0.985 of it (cross-entropy 0.147 against the optimum's 0.135), so it holds almost nothing beyond W_hi, W_hh and the task. At that optimum a readout step costs second order, ε²/2 times the neuron's power where the answer is in doubt, so the costliest readout weights are those on the largest neurons, n2 and n7.

## The cascade near a tie
Treat each (step, neuron) gate as a station. Round 0 freezes every gate at a reference route (lower inputs at equally spaced quantiles of their law), so the model is linear; round 1 computes each gate from the reference state and the actual input, a fixed threshold on that step's input; round k computes the gates from round k − 1's states, so stations shift later stations' thresholds; the rounds converge to the model. 256 configurations (family, T, M) at the tie S = T, 4,096 draws of the lower inputs each. Raw output: [station_rounds](../../results/experiments/station_rounds.txt).

| Share of draws whose correctness the round gets right; its P(wrong) bias | s0/0 | example/0 | s4/0 |
|---|---|---|---|
| Round 0, the reference route | 0.83; +0.08 | 0.84; −0.01 | 0.80; +0.10 |
| Round 1, fixed thresholds on each step's input | 0.89; +0.04 | 0.80; +0.00 | 0.86; +0.08 |
| Round 2 | 0.96; +0.01 | 0.88; +0.02 | 0.92; +0.04 |
| Round 3 | 0.99; +0.001 | 0.94; +0.02 | 0.97; +0.02 |
| Round 4; round 6 | 0.999; 1.000 | 0.97; 0.99 | 0.99; 0.9998 |
| Draws on the reference route in every gate | 0.10 | 0.05 | 0.14 |

So for the seeds a fixed set of stations, each a threshold on one input with a broadcast fixed by the reference route, accounts for 89% of near-tie decisions, and the rest is stations retuning stations, three rounds for 99% and five for 99.9%. Round 1 overestimates the error by 4–8 points: the later gates compensate for earlier deviations, so the cascade is self-correcting. The example's satellites take almost no direct input (|W_hi| 0.05–0.3) and read the core's state instead, so round 1 is worse than round 0 for it and the cascade runs past six rounds. Round 1 alone is a box partition of the lower inputs and could be integrated in closed form; every later round needs the actual states, so the account is as long as the computation ([requirements.md](../../parked/research/requirements.md)).

Which stations carry it, 2^19 draws at the tie, clean example and three copies ([cascade_stations](../../results/experiments/cascade_stations.txt)); the explorer shows the clean example's maps and, for any input, its own cascade (`why this answer`).

| Fact | example | s0/0 | s4/0 |
|---|---|---|---|
| Gates off the reference per draw: flipped at round 1; retuned later | 7.8; 8.8 | 6.6; 4.1 | 6.3; 4.0 |
| Stations firing at round 1 per draw, by step 0, 1, 2 … 9 | 3.7, 1.2, 0.9, 0.6, 0.6, 0.4, 0.3, 0.1, 0.1, 0.0 | 3.7, 0.9, 0.6, 0.4, 0.4, 0.3, 0.1, 0.1, 0.1, 0.1 | 3.7, 1.1, 0.4, 0.3, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1 |
| The first input's sign, when position 0 is a lower input: share of draws it flips every step-0 gate on; retuned gates with it, without it | 0.23; 16.3, 6.5 | 0.23; 7.2, 3.2 | 0.23; 7.0, 3.1 |
| Share of retunings at the steps that read M, S or T | 0.40 | 0.45 | 0.47 |
| Ten most active stations after step 0, their share of round-1 flips | core gates n6, n8, n1, n4, n7 at steps 1–4; 0.41 | n3, n6, n12, n5 at steps 1–4; 0.52 | n4, n11, n12, n15 at steps 1–4; 0.64 |
| Ten most retuned gates, their share of retunings | n2, n13, n0, n5, n9 at step 1, n15 at 4–5; 0.29 | n14 at steps 1–4, n11, n3; 0.38 | n3, n4, n11, n14 at steps 1–2; 0.39 |
| Distinct round-1 flip sets; sets holding 50% / 90% of draws | 39,712; 62 / 7,457 | 23,294; 46 / 3,380 | 20,278; 17 / 2,101 |

The stations are the early steps: before the max arrives the state is small and each lower input's sign and size set the gates, and once it has arrived the lower inputs flip almost nothing directly (0.1 per step from step 6 on). The largest station is the first input's sign, which flips all sixteen step-0 gates and doubles the cascade. On the example the stations are the core's comparator and record gates at steps 1–4 and the retuning lands on the satellites; on the seeds a few satellites fire and one or two take the retuning. Two fifths of the retuning falls on the steps that read M, S and T: the lower inputs' early stations change how the top three are processed. The flip sets have a long tail, so there is no short list of stations: 17–62 sets cover half the draws and thousands the rest.

## Open
- Why training chose a 45° clock with two dedicated positions, where s0, s3 and s4 turn about 165° per step: the weights show each choice, not its reason.
