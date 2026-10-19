# Parked

Work outside the current focus (`plan.md`), kept whole so it can come back. Each file sits at its old path under
`parked/`: `parked/az/estimators/adf.py` was `az/estimators/adf.py`. Everything here last ran in the Interpretability
repository, before the RNN track split from it, at its commit `f734363`. Markdown links resolve where the files sit now;
code, run commands and `Finding:` lines still name the old paths.

**To revive** a piece of work, `git mv` its files back, with the experiments and raw outputs its findings cite, fix
any import of a parked module, and run `python -m az.check --fix`.

| Area | What is here | Why it is parked |
|---|---|---|
| The RNN's estimators | 23 estimator ideas (`research/angles/`), why structure loses on accuracy (`research/requirements.md`), their experiments, the accuracy-control estimators `adf` and `sweep`, and the accuracy scores ([benchmarks](benchmarks.md)) | The estimator bar is unmet with an edge over `ce` that stays flat, and accuracy is a closed control. Active: the account, the tail, and `modes` |

## The RNN's estimator protocol
For whoever revives the estimator bar; `az/rare.py`, `az/bench.py` and the active estimators implement it.
- An estimator reads the copy's weights and structure found on a base model (roles, routes, families, failure modes),
  never truth files or event probabilities. It may deduce, use point sets, or sample; a randomized estimator seeds from
  the weights so its runs repeat. W_hh is fragile, so an estimator for the noisy suite uses it exactly.
- It beats sampling on a level when it reaches 10% relative RMSE in fewer ops, and counts as progress only when its
  edge over `ce`, paired by seed salt on the same suite, grows as c rises. Its `benchmarks.md` row names the structure
  it reads.
- Ops are multiply-adds, counted by the estimator itself; transcendental and CDF costs live in `az/ops.py`. Sampling
  costs `SAMPLE_OPS` = 2880 per input, Hilton's convention, plus 10 for |x| on the rare-event levels (`TAIL_OPS`).
- Suite: 6 base models × 32 draws of N(0, 0.002²) on every weight. Accuracy truth is 2^24 float64 samples per model.
  Rare-event truth is float32 counts on 2^31 inputs per bench model and 2^30 per held-out model (noise seeds 5000 + i);
  settle designs on the held-out suite, then score the bench suite once.
- Gates: `python -m az.rare sampling --budgets 1e6,1e8,1e10` (relative MSE over sampling's ≈ 1 where it expects hits)
  and `python -m az.bench sampling --budgets 1e5,1e6` (mse/samp ≈ 1).

## Index
<!-- index:start -->
### Angles
| Doc | Status | Summary |
|---|---|---|
| [agent-loop](research/angles/agent-loop.md) | proposed | Agents write structural claims about neurons as estimator code; `az.rare` keeps whatever widens the edge over `ce`. |
| [exact-accuracy](research/angles/exact-accuracy.md) | parked | Compute each base model's accuracy exactly, region by region: mechanistic, and it beats sampling outright above its cost. |
| [family-lattice](research/angles/family-lattice.md) | parked | Stratified deterministic sampling over families and the top three's laws: 0.3–0.6 of random sampling's MSE on the accuracy control. |
| [hermite-propagation](research/angles/hermite-propagation.md) | parked | ARC's random-MLP tool: its ReLU step assumes near-Gaussian pre-activations, but here they are kinked functions of few inputs. |
| [margin-gaussian](research/angles/margin-gaussian.md) | parked | Given the family and top three magnitudes, the margin is Gaussian over the lower inputs; the estimator reaches RMSE 0.0021, not sampling. |
| [mode-strata](research/angles/mode-strata.md) | parked | Stratify on the modes' features with model-free masses, or sample one mode's exact stratum: a gain capped by the stratum's mass. |
| [order-exact-magnitude-quadrature](research/angles/order-exact-magnitude-quadrature.md) | parked | Accuracy = E over magnitude sets y of acc(y): orderings combinatorially, magnitudes by quadrature; acc(y) ≈ 1 off near-ties. |
| [quantization-tree](research/angles/quantization-tree.md) | parked | K deterministic atoms for the state, each new input integrated exactly in 1-D: error falls like K^(-1/t), not K^(-1/2). |
| [tail-lines](research/angles/tail-lines.md) | parked | Integrate S exactly over (T, M) at each draw: inside the modes mixture the exact tie wedges cut the worst copies' variance 4–23×. |
| [training-dynamics](research/angles/training-dynamics.md) | parked | Watch structure form across the zoo's checkpoints, so each part of the explanation is tied to when and why training added it. |
| [verification-bnb](research/angles/verification-bnb.md) | parked | Certify input boxes correct by bound propagation, split the rest: boxes straddling the 9-D boundary shrink like N^(-1/9). |
| [additive-presumption](research/angles/additive-presumption.md) | dead | Each lower input's exact 1-D effect on the tie margin, summed: the margin's law would be a convolution of seven exact laws. |
| [adf](research/angles/adf.md) | dead | Gaussian projections of the state smear exact zeros and exact linear relations, and the error compounds over 10 steps. |
| [advice-noise-kernel](research/angles/advice-noise-kernel.md) | dead | Reuse work done on a copy's base model and pay per copy only for the noise's effect: no cheap per-copy form follows the copy. |
| [exact-suffix](research/angles/exact-suffix.md) | dead | Correctness is piecewise constant along any line; integrating the last inputs or a great circle exactly is cheap, but what is left is not. |
| [level-smc](research/angles/level-smc.md) | dead | Subset simulation needs no proposal density, but random-walk moves fall out of the thin tie wedges: worse than sampling at every level. |
| [low-dim-surrogates](research/angles/low-dim-surrogates.md) | dead | Every surrogate in natural low-dimensional coordinates misses accuracy by 2–5 points, exactly on the error set. |
| [region-enumeration](research/angles/region-enumeration.md) | dead | Linear regions are too many and too evenly weighted: the top 10^5 activation patterns cover only 56% of the mass. |
| [route-mixture](research/angles/route-mixture.md) | dead | Label splits alone give 3^10 components; with ReLU splits the mixture passes 2×10^5 components by step 3. |
| [route-polytope](research/angles/route-polytope.md) | dead | One linear route per (p, q) makes the accuracy a sum of polytope masses; measured first, it sits 0.09–0.30 below the zoo seeds' copies. |
| [route-screen](research/angles/route-screen.md) | dead | Run the model only where the reference route's linear margin predicts an event, weighted by 1/acceptance: the route misses the tail. |
| [spike-slab-mixture](research/angles/spike-slab-mixture.md) | dead | ADF keyed on the running top three and the activation pattern: the cells are few, but a Gaussian per cell loses the decision. |
| [tie-wedge](research/angles/tie-wedge.md) | dead | Gap density integrated over designed inputs; built and scored: it loses to sampling at every budget, by a bias the lower magnitudes set. |
| [value-sketch](research/angles/value-sketch.md) | dead | P(correct | the first t inputs) is exact for one input left, but near-binary in the state: no low-dimensional sketch holds it. |

### Notes
| Doc | Summary |
|---|---|
| [requirements](research/requirements.md) | Why no estimate built from the model's structure beats sampling on accuracy, the common-event control, each constraint pinned to a number. |
<!-- index:end -->
