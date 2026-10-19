# Route mixture: Gaussians over fixed activation sequences in input space

> Label splits alone give 3^10 components; with ReLU splits the mixture passes 2×10^5 components by step 3.
> Each component holds an exact linear map (a "route") and a Gaussian over the inputs that take it. It is exact within a route, with no merging possible across routes, and merging with a measured error needs over a million components for an error of 10^-3.

**Status:** dead

## Idea
Split a component whenever a ReLU's sign is uncertain beyond eps, or a label event (new max or new 2nd max) is possible, and keep h = A x exact per component. Accuracy is the mass-weighted sum of each route's P(correct). This is Cammarata's "routes" in ARC's currency.

## Why it died
Component counts per step for eps = 0.1, 0.03 and 0.01 are in [route_mixture_proto](../../results/experiments/route_mixture_proto.txt). Components can't merge, because routes carry different linear maps.

## Merging with a measured error
The revival named here was merging routes whose readout-visible futures agree within a bound, carrying the bound as error. The cheapest merge with a measurable error: inside each (family, record history) cell, force every gate whose deviation from the cell's majority is rarer than τ to the majority and keep the rest as the input's own bit; the components are the distinct (cell, reduced pattern) keys left, and the error is the merged model's accuracy against the true one on the same 2^22 inputs. A certificate would bound that error from above, so the ladder bounds the components a certified mixture needs from below. Raw output: [merge_ladder](../../../results/experiments/merge_ladder.txt).

| τ | Components, example/0; s0/0; s4/0 | Accuracy error |
|---|---|---|
| 0.5 (every gate at its cell's majority) | 31k; 28k; 26k | −0.030; −0.060; −0.037 |
| 0.2 | 1.2M; 0.7M; 0.6M | −0.005; −0.011; −0.007 |
| 0.1 | 1.7M; 1.1M; 0.9M | −0.002; −0.003; −0.003 |
| 0.05 | 2.0M; 1.4M; 1.1M | −0.0007; −0.0004; −0.0006 |

An input deviates from its cell's majority route in 6–10 gate bits on average. An error of 10^-3, which sampling reaches at 3×10^8 ops, needs 1–2 million components even counted on 2^22 inputs, a lower bound; at 10^4 ops for one polytope measure that is 10^10 ops, where sampling's error is 2×10^-4. Closed.
