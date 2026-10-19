# The confident-error tail

> The tail sits at ties: where inputs tie, the logits are fixed but the label is not, so one forward pass gives every label's margin.
> Each base fills its tail from one or two modes, each read from the weights: all inputs below zero but the max with the last one 2nd (example, s3), one dominant input (s0, s1, s4), the top two tied (s2). At c = 2 the tail is the near-tie wedges at both ends.
> Weight noise keeps each mode's shape and moves it between orders (p, q, r): re-estimating only the orders on a copy turns a 0.6–6.2× proposal into 3.0–7.6× at c = 2. The copy's margin at one point per order predicts about a third of that move.
> At c = 8 the copies that set the error hold a second tie mode the fresh copies never showed: a three-way tie on s2, a near top-two tie under s1's and s4's dominant input. Its shape, not its order, is what a library misses.

## Ties fix the margin
The model is continuous in its input and the label is not. At a point where inputs tie, every way of breaking the tie keeps the same logits, so the model is wrong for all but one of them, and its margin there, max_{j≠q} l_j − l_q over |x|, is a property of the weights. One forward pass at the tie gives it for every label the tie allows. As c rises the events move toward such points: their median distance to the nearest direction whose inputs take only two values falls from the nominal 0.49 to 0.25–0.36 at c = 8 on example, s3 and s4 ([tail_modes](../../results/experiments/tail_modes.txt)). On s0, s1 and s2 the event regions stay broad at c = 8 (0.41–0.47); their modes are a dominant input or a tie of two, with the other inputs free.

## The modes, base by base
Each event is assigned to the first mode it fits ([tail_modes](../../results/experiments/tail_modes.txt), held-out copy /0 of each base, 2^28 inputs).

| Base | Mode holding the tail at c ≥ 6 | Share at c = 6, 8 | Its margin from the weights |
|---|---|---|---|
| example | at most one positive input, label at position 9 | 0.82, 0.97 | D = 62.2 from Wo relu(−Wi); measured slope 55.5 |
| s3 | same | 0.97, 1.00 | D = 40.4; measured 41.1 |
| s0 | one positive input above 0.7 of the norm | 0.46, 0.58 | margin 47.0 at the best e_k |
| s1 | same | 0.32, 0.62 | 32.8 at the best e_k |
| s2 | top two within 0.02, median M − S 0.001 | 0.92, 0.92 | 16.2 at the best tied pair |
| s4 | at most one positive (label 9 in a quarter at most), then a dominant input of either sign | 0.79, 0.48 + 0.48 | 30.4 at +e_k, 45.0 at −e_k |

The first mode is deduced. When every input but the max is below zero, the recurrence keeps almost nothing of the early inputs, so the logits are about |x_9| · Wo relu(−Wi), the response to the last input alone. If x_9 is the 2nd largest, the model picks the slot that response favors, with margin D |u_9|, where D = max_{j≠9} L_j − L_9 and L = Wo relu(−Wi). The ordering caps |u_9| at 1/3, so the vertex −(1 − e_p)/3 bounds the mode: its margin is 20.9 on example and 13.8 on s3. The measured slope f / |u_9| on the mode's events is 55.5 against D = 62.2 on example and 41.1 against 40.4 on s3. On s0, s1 and s4 the same stratum holds a different mechanism, and the slope misses D (26.0 against 41.1 on s0).

A dominant input works the same way to first order. With one input carrying most of the norm, the state after it is its impulse response, the small inputs decide the label, and the margin is about u_k times the impulse's logit gap, the +e_k or −e_k margin in the table. The top-two tie is the M-end wedge at its thinnest: on s2 the events at c ≥ 4 sit at a median M − S of 0.001, with the max late in 60% of them ([tail_structure](../../results/experiments/tail_structure.txt)).

## At c = 2, the wedges
At c = 2 the median gap at the erring end is 0.016–0.047, against the nominal 0.087 for S − T and 0.139 for M − S. example, s2 and s4 err at the T end. s0, s1 and s3 err at the M end, predicting the max. Near ties hold 41–70% of the events at c = 2, and at most one positive input 4–36% ([tail_modes](../../results/experiments/tail_modes.txt)). The wedges are thin, but the margin at the tie also depends on the lower inputs, and only 6–26% of S-lines contain an event at c = 2 ([tail_lines](../../results/experiments/tail_lines.txt)).

## Noise moves the modes between orders
A mixture of order-statistic gap proposals fitted on fresh noisy copies of a base covers a held-out copy's tail with the shapes right and the orders wrong. On eleven held-out copies, the per-sample gain over sampling ([tail_orders](../../results/experiments/tail_orders.txt), part 1):

| Level | Library shapes and orders | Library shapes, the copy's orders | Full refit on the copy |
|---|---|---|---|
| c = 2 | 0.58–6.20 | 3.03–7.58 | 3.91–9.21 |
| c = 4 | 0.59–58.8 | 2.29–228 | 5.39–87.6 |

Re-estimating only π(p, q, r) recovers most of a full refit at c = 2. From 1,000 events, as a first stage of an estimator would hold, it lifts c = 2 from 0.6–4.4× to 2.0–6.5×. At c = 4 the result is mixed, and at c = 8 it can lose (555 to 156 on example), because a few events overfit 720 orders ([tail_orders](../../results/experiments/tail_orders.txt), part 2). The estimator that uses these modes: [mode-mixture](../angles/mode-mixture.md).

## The weights predict part of the move
The copy's own margin says where its tail went, but one point per order says only part of it. Weighting the library's orders by exp(0.5 (f_copy − f_base)), with f the margin at each component's mean point placed at each of the 720 orders, closes a tenth to three fifths of the log-distance to the oracle, about a third on average at both levels ([order_prediction](../../results/experiments/order_prediction.txt), eight held-out copies):

| Per-sample gain, the copies where the library is weakest | c = 2 | c = 4 |
|---|---|---|
| library orders | 0.58–1.08 | 0.59–3.94 |
| orders from the margin at the mean point | 0.80–1.76 | 0.91–6.55 |
| orders from the margin at the tie point | 0.77–1.52 | 0.74–3.17 |
| orders from 4,000 fresh-copy failure points run on the copy | 0.59–1.07 | 0.70–4.19 |
| oracle orders, from 20,000 of the copy's events | 1.71–4.57 | 1.96–11.5 |

The tie point does no better than the mean point, and the fresh copies' failure points do no better than the library: the copy's events sit at orders where the pool holds few points. The margin at a tie depends on the lower inputs, so the event mass at an order takes many evaluations to see, and one evaluation per order already costs 720 passes per component: for the eight c = 2 components, 40% of the c = 2 budget at its 10% crossing. Inside the estimator the prediction lifts only c = 6. Learning the orders from the estimator's own first stage does better and costs no passes ([mode-mixture](../angles/mode-mixture.md)).

## What a library misses at c = 8
The modes library's exact mismatch m = E_{p|A}[p / q] / P(A) is its relative variance per point plus one, read from a copy's own events among 2^31 inputs. On six held-out copies, four of them the ones whose realized errors are largest, m at c = 8 runs from 224 (example/0) to 14,968 (s2/28), where a proposal matching the copy would sit near 10. The heaviest tenth of the events carries 59–96% of it, in a tie mode beside the base's main one ([heavy_modes](../../results/experiments/heavy_modes.txt)):

| Copies | Heaviest tenth of the events at c = 8 | The bulk |
|---|---|---|
| s2/15, s2/28 | a three-way tie at orders (8, 5, ·): M − S 0.005–0.008, S − T 0.008–0.010 | the top-two tie, S − T 0.12–0.33 |
| s1/12, s1/16, s4/3 | a near top-two tie: M − S 0.024–0.052 | one dominant input: M − S 0.70–0.80 |

Giving every component the copy's own order law lowers m at c = 8 at most 2.7× (s1/12) and raises it on s1/16 and s4/3; refitting four components per level on half the copy's events lowers it 1.4–20×, to 160–1,628. So the shapes carry most of the miss, and only the copy's own events show them. Defensive changes move m by at most 31% ([heavy_modes](../../results/experiments/heavy_modes.txt), part 3).

The local geometry says why a fitted shape misses. On s2 every event of an order shares one route gradient (cosine 1.000) of norm 1,444, so the margin falls a thousand times faster across the top-two tie than along it, a slab 0.005 wide whose depth the other inputs set; at the three-way tie (8, 5, 0) the gradient takes two opposite directions. On example's all-negative mode the margin is one linear map of norm 61.7, the deduced D = 62.2, within 2%.

The weights find one of the two second modes. A tie atlas, each tie type's order law set by the copy's margins at its 720 tie points (2,880 passes per type), ranks s1/12's commonest c = 6 orders (5, 0, 9) and (5, 0, 7) first and second of 720 and lowers its m 17× at c = 6 and 6× at c = 8. On s2 the margin at a three-way tie depends on how the lower inputs are placed, the atlas misses (8, 5, ·), and m moves by −9% to +19%.

## Evidence
| Fact | Number | Raw output |
|---|---|---|
| Events at c ≥ 4 per base: at most one positive; predicting the max; max at positions 7–9 | 0.00–0.90; 0.13–0.76; 0.05–0.60 | [tail_structure](../../results/experiments/tail_structure.txt) |
| Nominal mass: at most one positive; a dominant input; top two within 0.02; 2nd and 3rd within 0.02 | 0.011; 0.163; 0.077; 0.103 | [tail_modes](../../results/experiments/tail_modes.txt) |
| Vertex margins, best over positions: all-negative tie; +e_k; −e_k; top two tied (example, s0, s1, s2, s3, s4) | 20.9, 8.5, 9.5, 2.5, 13.8, 9.8; 8.6, 47.0, 32.8, 5.4, 4.7, 30.4; 62.2, 41.1, 39.7, 5.9, 40.4, 45.0; 4.5, 4.9, 7.6, 16.2, 7.4, 8.1 | same |
| Per-line gain of the exact S-line at c = 2, 4, 6, 8, five bases; s2 | 2.8–6.1, 4.7–8.1, 3.1–12.2, 3.3–17.5; 6.3, 56, 107, 208 | [tail_lines](../../results/experiments/tail_lines.txt) |

## Connections
- Errors at c = 0 and the lower inputs' role near the tie: [errors.md](errors.md).
- The cascade of gate stations that decides a near-tie: [circuit.md](circuit.md).
