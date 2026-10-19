# Exact accuracy by deductive brute force

> Compute each base model's accuracy exactly, region by region: mechanistic, and it beats sampling outright above its cost.
> Feasibility depends on the region count, which is unknown (≥ 5.5×10^5 seen). It anchors the high-compute end of the MSE curve and gives a truth with no sampling error. ARC would not call it understanding: surprise grows with the region count.
> Organizing the error set by (family, record history) does not shrink it: inside a cell it is not convex and spans 130–190 activation patterns, so there are no few polytopes to measure.

**Status:** parked

## Plan
1. Estimate the total number of regions from pattern-discovery curves (species-richness extrapolation on [accuracy_structure](../../../results/experiments/accuracy_structure.txt)-style counts). Stop if above about 10^9.
2. Enumerate by exact 1-D sweeps through the network (find each breakpoint along a line), recursively over input dimensions, with GPU batching.
3. Measure each cell exactly: on an ordering cone, a Gaussian cone measure with few active facets.

## The error set by (family, record history)
The hope: regions are too many, but the error set inside a (family, record history) cell might be one convex polytope with few facets, so a deterministic solid-angle series over about 4,000 cells would replace enumeration over a million regions. Three suite models, 2^22 inputs. Raw output: [error_cells](../../../results/experiments/error_cells.txt).

| Fact | example/0 | s0/0 | s4/0 |
|---|---|---|---|
| (family, record history) cells holding 90% / 99% of the error mass | 3,755 / 10,151 | 3,931 / 10,444 | 3,746 / 10,443 |
| Midpoint of two error points in one cell is correct | 0.33 | 0.25 | 0.23 |
| Segments between two error points with 2 / more than 2 correctness switches | 0.40 / 0.02 | 0.32 / 0.01 | 0.30 / 0.01 |
| Activation patterns among a cell's error points, error-mass weighted mean; error mass in cells with more than 10 | 193; 0.92 | 155; 0.96 | 134; 0.96 |
| (cell, pattern) keys for 90% of the error mass; share of error mass in keys seen once | 234k; 0.61 | 323k; 0.39 | 248k; 0.32 |
| Competitors (positions the model picks) per cell, mean | 4.1 | 4.2 | 4.5 |

The cells themselves are convex cones, and the error set inside them is not: a quarter to a third of error-pair midpoints are correct, and the error points of one cell lie in over a hundred linear pieces, which two random error points share 1–7% of the time. At 2^22 inputs a third to two thirds of the error mass sits in (cell, pattern) keys seen once, so the pieces are not even enumerable by sampling at this scale. The same count rules out a perturbative expansion around each cell's majority route: an input deviates from it in 6–10 gate bits on average ([merge_ladder](../../../results/experiments/merge_ladder.txt)).

## The state-machine route, and what replaced it
`Learn2Fly/verify.py` (`prove`) computes exact accuracy as a `Fraction` by counting a model's paths, but only for discrete inputs. A rank-event machine for this model's "skeleton" fails twice over: the six-neuron core alone scores chance ([circuit.md](../../../research/findings/circuit.md)), and decisions depend on magnitudes, not just order.
What does work exactly: the full model's accuracy over all 10! orderings of a fixed magnitude set, as a fraction certified against float rounding ([orderings.md](../../../research/findings/orderings.md)). Gaussian accuracy is that quantity averaged over magnitude sets, so exact accuracy now reduces to the magnitude integral: [order-exact-magnitude-quadrature](order-exact-magnitude-quadrature.md). `V3R1T4S/adderboard/submission/tiny2.py` (`prove`) adds per-region margins to the nearest readout breakpoint, a way to bound how much W_hh noise each region survives. Paths are relative to this repository's parent folder.

## Success criterion
Accuracy of `example` to 10^-6, at a stated op count. That count is the budget above which this estimator beats sampling.
