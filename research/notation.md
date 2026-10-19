# Notation

> Symbols and terms every research file uses without redefining them.

| Term | Meaning |
|---|---|
| x = (x_0 … x_9) | input, i.i.d. N(0, 1); the model and correctness are scale-free, so only its direction matters |
| p, q, r | positions of the largest, 2nd-largest (the label), and 3rd-largest input |
| M, S, T | the values at p, q, r; "gaps" are M − S and S − T |
| y_1 > … > y_10 | sorted inputs; "rank k" is y_k |
| l = W_oh h_10 | logits; the margins l_q − l_r and l_q − l_p decide most errors |
| positives | number of x_i > 0; at most one positive means 1.1% of mass, mostly wrong |
| canonical form | neurons with \|W_hi\| > 0.5 rescaled so W_hi = ±1 (ARC's standardization) |
| core, satellites | example model's neurons {2,4,7,6,1,8} and the other ten ([circuit.md](findings/circuit.md)) |
| pattern, route | the 160 on/off bits of all neurons over 10 steps; within a route the model is linear |
| record history | at each step, whether x_t is a new running max, a new running 2nd max, or neither |
| key | the discrete state an estimator conditions on, e.g. (p, q) so far |
| suite | 6 base models × 32 copies with N(0, 0.002²) on every weight |
| ops | multiply-adds; one forward pass = 2880 (`SAMPLE_OPS`) |
| mse/samp | estimator MSE over sampling's MSE at the same ops; below 1 beats sampling (accuracy control) |
| level c | the rare event max_{j≠q} l_j − l_q ≥ c·\|x\|, confidently wrong; scale-free; c = 0 is the error rate; levels 0, 2, 4, 6, 8 ([az/rare.py](../az/rare.py)) |
| mode | a region of the confident-error tail named by its mechanism: a tie wedge, the all-negative inputs, a dominant input ([tail.md](findings/tail.md)) |
| gap proposal | a law on directions: orders (p, q, r), the 2nd largest S, and half-normal gaps to M, to T and below T, with exact angular density ([az/osp.py](../az/osp.py)) |
| ops to 10% | ops an estimator needs for 10% relative RMSE on a level; its advantage is sampling's ops to 10% over its own |
| mismatch m | E_{u ~ p\|A}[p(u) / q(u)] / P(A) of a proposal q at a level: an importance sampler's relative variance per point plus one, exact from a copy's own events ([expected_score](../experiments/expected_score.py)) |
| mechanistic | deduced from the weights and the circuit, not read off samples; an estimator's mechanism is the structure it reads, named in its benchmarks.md row |
