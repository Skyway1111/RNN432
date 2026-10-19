# Training dynamics: explain the model by watching it form

> Watch structure form across the zoo's checkpoints, so each part of the explanation is tied to when and why training added it.
> This is ARC's "train and explain" and Cammarata's "watch neurons grow". First result: the skeleton is done by 2^24 and accuracy's last 15–20 points come after ([training.md](../../../research/findings/training.md)).

**Status:** parked

## Why it could help the estimate
Phase 3 adds the accuracy that the legible skeleton doesn't explain. If the phase-3 changes are few or structured (one weight block, one mechanism), they name what the estimator must capture, and the 2^24 checkpoint is a simpler model to explain first.

## Next experiments
1. **Localize phase 3 with dense checkpoints.** Splicing and straight-path attribution both fail between zoo checkpoints, which are too far apart ([training.md](../../../research/findings/training.md)). Retrain seeds with checkpoints every few hundred steps, then attribute along fair segments.
2. **Error set over time:** rerun [error_set_dimension](../../../experiments/error_set_dimension.py) per checkpoint. Does the share of error variance that low-dimensional features explain fall as accuracy rises?
3. **Seed 2's switch at 2^25:** which neurons change role, and does its error set differ from the other seeds'?

## Tools to borrow (sibling projects, paths from this repository's parent folder)
- **Dense retraining:** the zoo's 12 log-spaced checkpoints are too coarse to time a neuron forming. `V3R1T4S/adderboard/submission/tiny2.py` (`train`) and `Learn2Fly/ensemble.py` (`Stack`) train thousands of seeds at once, each weight with a leading model axis, replayed as a CUDA graph (4096 seeds in 80 s on this GPU). Use ARC's settings from `_ext/alg-zoo/alg_zoo/training.py`, which has no gradient clipping. These would be new seeds, not the zoo's.
- **Timing method:** `Learn2Fly/results/supervision.md` checkpoints every 500 steps and swaps one seed's init onto another seed's data, separating the effect of init from data order. It would tell us whether single-neuron R² or decode R² rises first.
- **Pre-registration:** `Learn2Fly/results/records/specs/` states each prediction as a direction against a baseline before the run, then repeats on fresh seeds. With 5 seeds, stories fit after the fact are cheap.

## Data
`data/checkpoints.npz` (from `python -m az.zoo`); `az.zoo.checkpoint(seed, n)` loads one. The example model has no checkpoints.
