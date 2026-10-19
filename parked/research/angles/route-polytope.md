# Route polytopes: the reference-route surrogate's accuracy by deterministic quadrature

> One linear route per (p, q) makes the accuracy a sum of polytope masses; measured first, it sits 0.09–0.30 below the zoo seeds' copies.
> The example's +0.004 agreement ([votes.md](../../../research/findings/votes.md)) was cancellation: one route scores 0.870, eight averaged 0.957, the best possible per-pair linear fit 0.984, against the model's 0.953. No per-pair linear surrogate has the model's accuracy, so there is nothing to integrate.

**Status:** dead

## Idea
For each (p, q), take the model's exact linear map at a reference input (the mean magnitude set, lower values in fixed orders) and apply it on the whole cone x_p > x_q > rest. The surrogate's correct set per cone is a polytope in 10-D, whose Gaussian mass a Genz-style sequential-conditioning quadrature with a deterministic lattice computes without sampling, at about 1e5–1e6 ops per cone. The advice is the recipe, not a table. If the surrogate's accuracy tracked the model's to 0.004, this would beat sampling near 1e7 ops.

## Why it died
Surrogate minus model on the same 2^18 inputs, 192 suite models. Raw output: [surrogate_bias_suite](../../results/experiments/surrogate_bias_suite.txt).

| Surrogate | Mean bias | RMS bias | Mean bias per base: example, s0, s1, s2, s3, s4 |
|---|---|---|---|
| per (p, q), 1 order, mean set | −0.206 | 0.228 | −0.06, −0.23, −0.28, −0.11, −0.22, −0.34 |
| per (p, q), 8 orders, mean set | −0.179 | 0.216 | +0.02, −0.28, −0.26, −0.09, −0.16, −0.30 |
| per (p, q, r), 1 order, mean set | −0.142 | 0.145 | −0.11, −0.17, −0.13, −0.13, −0.14, −0.18 |
| per (p, q), 8 orders, gaps 0.1 | −0.257 | 0.279 | −0.11, −0.38, −0.26, −0.14, −0.27, −0.38 |

On the clean example, one route per pair scores 0.870 and agrees with the model on 85% of inputs; the eight-route mean scores 0.957 and agrees on 92%. The zoo seeds' routes depend on the magnitudes too strongly for any reference route to stand for a pair: at their reference inputs they are right, and on the pair's Gaussian inputs the same map is wrong a quarter of the time. The example's near-agreement is a coincidence of a model whose routes are unusually stable ([circuit.md](../../../research/findings/circuit.md)), and even there the surrogate's accuracy is a different number from the model's: the best per-pair linear fit reaches 0.984 ([linearity.md](../../../research/findings/linearity.md)).

## What this closes
Any estimator whose per-copy work is "evaluate a compact structure fixed on the base model" needs that structure to hold the copy's behavior to 0.005. Reference routes hold it to 0.02 on the example and 0.1–0.3 on the seeds. The middle band needs a structure that is exact, which is the region decomposition and the high band ([exact-accuracy](exact-accuracy.md)), or advice that is itself a sampled table, which is sampling moved into the explainer.
