# Region enumeration with exact volumes

> Linear regions are too many and too evenly weighted: the top 10^5 activation patterns cover only 56% of the mass.
> The M_{2,2} method (enumerate linear pieces, compute each one's correct share exactly) scaled up. Exact in the limit, but each region is a 10-D polyhedral cone.

**Status:** dead

## Why it died
See the pattern counts in [linearity.md](../../../research/findings/linearity.md). With each cone needing a 10-D Gaussian measure, covering 99.5% of the mass costs far more than sampling at matching precision.

## What would revive it
Regions far coarser than linear pieces on which correctness is still provable, e.g. [verification-bnb](verification-bnb.md); or the brute-force endpoint itself, [exact-accuracy](exact-accuracy.md).
