# Sequential Monte Carlo over levels

> Subset simulation needs no proposal density, but random-walk moves fall out of the thin tie wedges: worse than sampling at every level.
> Collapsing S's line exactly makes the chain move along a smooth target, and it still loses to modes 16–370× at c = 6 and 8, at seven passes a line.

**Status:** dead

## Idea
Rare-event estimators that adapt to the copy without a fitted shape: N particles climb the ladder c = 0, 1, …, 8; at each step the share clearing the next level estimates the conditional probability, the survivors are resampled, and moves that keep the level refresh them. The cost grows with the number of levels, the log of 1/p, rather than with a proposal's misfit, which [mode-mixture](../../../research/angles/mode-mixture.md) shows sets modes' error ([tail.md](../../../research/findings/tail.md), What a library misses).

Two move kernels. Generic: pCN on all ten inputs, accepted when the input stays in the event. Line-collapsed: a particle is the nine inputs other than S, with S's slot fixed, and its target is the S-line's exact event mass ([tail-lines](tail-lines.md)), so the thin direction across a tie is integrated and every ratio is Rao-Blackwellized.

## Measured
Three held-out copies, 16 runs per setting; advantage is sampling's ops to 10% relative RMSE over the method's, scaling as 1/ops ([level_smc](../../results/experiments/level_smc.txt)):

| Copy | Generic, N 4,000, 10 moves: advantage at c = 2, 4, 6, 8 | Line-collapsed, N 2,000, 3 moves |
|---|---|---|
| s2/15 | 0.05, 0.11, 0.13, 0.15 | 0.08, 0.57, 2.17, 5.49 |
| s1/12 | 0.12, 0.15, 0.17, 0.19 | 0.03, 0.19, 0.46, 0.34 (bias −0.41 at c = 8) |
| example/0 | 0.08, 0.08, 0.11, 0.17 | 0.12, 0.13, 0.14, 0.25 |

modes' advantage on the bench suite is 34 at c = 6 and 92 at c = 8.

## Why it died
The tail is a union of wedges 0.001–0.01 wide around ties ([tail.md](../../../research/findings/tail.md)). A move that changes all ten inputs leaves a wedge unless its step is smaller than the wedge, so the chain freezes and the particles that survive stay where they started. Integrating S along its line removes that direction, and the chain then mixes, but each move costs a line, seven passes, against 1.5 passes for one of modes' points that already lands in the wedge.
