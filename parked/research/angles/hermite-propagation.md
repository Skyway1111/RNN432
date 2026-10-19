# Hermite and cumulant propagation in input space

> ARC's random-MLP tool: its ReLU step assumes near-Gaussian pre-activations, but here they are kinked functions of few inputs.
> With 10 inputs a degree-d basis has C(10+d, d) terms (1001 at d = 4, 8008 at d = 6), so the size is affordable. The obstacle is accuracy, not cost.

**Status:** parked

## Cost argument
The expectation of a degree-d projection equals E[f]; errors enter only through the projections of intermediate ReLUs. ARC's ReLU step expands around a Gaussian, while this network's pre-activations are kinked functions of a few inputs, so truncation error compounds over 10 steps, as Gaussian projection does in [adf](adf.md). Revive it if a ReLU projection is found that is exact for piecewise-linear arguments.
