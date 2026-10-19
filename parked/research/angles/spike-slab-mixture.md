# Spike-and-slab state mixture

> ADF keyed on the running top three and the activation pattern: the cells are few, but a Gaussian per cell loses the decision.
> Only 2,500–7,200 cells hold mass above 1e-6 at step 10, so the bookkeeping is cheap. Yet with the deduction exact, one projection after step 3 costs 2–3 points of accuracy, and projecting at every step costs 21–56.

**Status:** dead

## Design
Key = positions of the running top three × the 16-bit activation pattern, with a Gaussian over the active coordinates of h and the running top values; inactive neurons stay exactly 0. Each step deduces exactly, then merges each key's inputs into one Gaussian. This fixes ADF's smeared zeros ([adf](adf.md)).

## Why it died
Raw output: [state_projection](../../results/experiments/state_projection.txt).

| Fact | s0/0 | example/0 | s4/0 |
|---|---|---|---|
| Cells with mass above 1e-6 at step 10 (per key) | 2,537 (4) | 7,215 (10) | |
| Accuracy error, Gaussian per cell after every step: keys only; keys × patterns | −0.28; −0.21 | −0.56; −0.27 | −0.32; −0.24 |
| Same, once after step 3, 6, 9 | −0.030, −0.066, −0.141 | −0.024, −0.063, −0.104 | −0.033, −0.073, −0.159 |
| Control: redraw from the cell's own inputs after every step | +0.0008 | −0.0002 | −0.0005 |

The control shows the loss is the Gaussian shape, not the bookkeeping. Within a cell the state is a mixture of linear images of truncated Gaussians, one per route history. The decision rests on margins of a few units between logits of tens, carried by exact linear ties between h and the running top values, and a Gaussian with the right first two moments loses those ties. Exact zeros are kept and are not the binding problem.

## What this closes
Any forward estimator that projects the state's law onto Gaussians per key, ADF included, misses by 20–500 times the error it would need. A forward representation would have to keep the state as exact linear functions of the inputs, which is region enumeration ([region-enumeration](region-enumeration.md)).
