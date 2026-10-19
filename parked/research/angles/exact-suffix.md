# Exact integrals along lines

> Correctness is piecewise constant along any line; integrating the last inputs or a great circle exactly is cheap, but what is left is not.
> The last input integrated exactly leaves 46–58% of the indicator's variance, the last two 29–34%. The prefix left over is a near-indicator in 8–9 dimensions, and no deterministic design of it beats sampling.

**Status:** dead

## Idea
The last input enters only the last step, so given the prefix the logits are affine in it between at most 16 kinks: an exact 1-D integral for about two passes ([value-sketch](value-sketch.md) has the method). A great circle through 0 is exact too, since correctness depends on direction only. Either replaces one quadrature direction by an exact integral.

## Why it died
Raw output: [suffix_integration](../../results/experiments/suffix_integration.txt).

| Fact | Number |
|---|---|
| Share of the indicator's variance left after integrating the last input; the last two (four models) | 0.46–0.58; 0.29–0.34 |
| Great circles: sectors per circle; MSE per op against sampling at two passes per sector | 210–330; 12–38 times worse |
| Rank-1 lattice over the prefix with the last input integrated, 127, 509, 2,039 prefixes: mse/samp | 1.40, 1.74, 4.34 |
| Inside gauss_margin, x_9 a lower input: coarse-minus-fine grid error at (7, 8), for the 16-run design, the suffix integrated exactly, and 512 draws | −0.0015 to −0.0031 for all three |

With random prefixes the last-input integral halves the variance for about twice the cost, near break-even with sampling, and it is conditional Monte Carlo, not deduction. With a deterministic prefix design the error is a bias shared by every model and falls slower than sampling's, so mse/samp grows with budget; and a lattice is equidistribution, the edge of the rules. Inside gauss_margin the grid error is the same whatever integrates the lower inputs. The sharpness lies in the top three gaps, so no exact integral over lower inputs reduces the cells.

## What this closes
Exact integration over a few directions as the main lever, deterministic prefix designs, and exact suffixes as a way to fewer cells or runs in [margin-gaussian](margin-gaussian.md).
