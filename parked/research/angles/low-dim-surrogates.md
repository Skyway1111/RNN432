# Low-dimensional surrogates

> Every surrogate in natural low-dimensional coordinates misses accuracy by 2–5 points, exactly on the error set.
> Tried: record-history masks, logits linear in order statistics or raw inputs per position configuration, and top-k truncation. Each would have reduced the estimate to a cheap low-dimensional integral.

**Status:** dead

## Why it died
The table in [linearity.md](../../../research/findings/linearity.md). Sampling reaches 0.58% RMSE at 10^7 ops ([benchmarks.md](../../../benchmarks.md)), so a 2-point bias loses at any budget above about 10^5. The error set itself resists low dimension: [errors.md](../../../research/findings/errors.md).

## What would revive it
A surrogate whose residual is mechanistically bounded rather than ignored, for example a surrogate plus a correction term computed per error mode. Untested.
