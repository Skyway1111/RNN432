# How linear and low-dimensional the model is

> Locally linear, not low-dimensional: patterns stay diverse with the input ordering fixed, and no natural coordinates linearize margins.
> Margins lean on the top 3–4 ranks, but their coefficients swing with which neurons fire, and positions explain at most 26% of that swing.
> Every surrogate tried here (record masks, order statistics, positional maps, top-k truncation) misses accuracy by 2–5 points, and the misses sit where the errors are.

## Evidence
Notation: [notation.md](../notation.md). Example model unless noted.

| Fact | Number | Raw output |
|---|---|---|
| Distinct 160-bit activation patterns in 2^20 inputs | 554,894 | [accuracy_structure](../../results/experiments/accuracy_structure.txt) |
| Mass covered by the top 10^3 / 10^4 / 10^5 patterns | 14% / 29% / 56% | same |
| Distinct patterns per 10^4 inputs with the full permutation fixed | 700–3500 | [pattern_given_permutation](../../results/experiments/pattern_given_permutation.txt) |
| Within a permutation: R² of one linear map; accuracy bias | 0.95; +1.8 points | [linearity_within_permutation](../../results/experiments/linearity_within_permutation.txt) |
| Record histories (when max / 2nd max change): distinct; top 3000 cover | 12,402; 91% | [event_masks](../../results/experiments/event_masks.txt) |
| Mask bits fixed by record history; accuracy with those masks | 93.6%; 0.922 | same |
| Margin l_q − l_r, contribution sd by rank 1…6 | 50, 77, 56, 13, 3.5, 1.4 | [margin_by_rank](../../results/experiments/margin_by_rank.txt) |
| Margin l_q − l_r, mean coefficient on M, S, T | +8, +53, −57 (sd 36, 65, 55) | same |
| Coefficient variance explained by (p,q,r) / by (p,q,r,4th), ranks 1–3 | ≤ 26% / ≤ 26% | [coef_by_config](../../results/experiments/coef_by_config.txt) |
| Logits linear in order statistics per (p,q,r): best R²; accuracy | 0.64; 0.94 vs 0.953 | [order_stat_surrogate](../../results/experiments/order_stat_surrogate.txt) |
| Logits linear in raw x per (p,q): R²; accuracy | 0.77; 0.984 | [positional_linearity](../../results/experiments/positional_linearity.txt) |
| Keep top k, zero the rest: per-input agreement, k = 3 / 8 | 96.5% / 98.6% | [topk_truncation](../../results/experiments/topk_truncation.txt) |
| Keep top k, push the rest 10 below: accuracy, k = 3 | 0.82 | same |

## Notes
- By homogeneity the margin equals Σ coefficient × value exactly (Euler check 2e-12), so "contribution by rank" is a decomposition, not a fit.
- Low ranks barely move the margin directly, but they decide which neurons fire, and that moves the coefficients on the top ranks.

## Connections
- Rules out [region-enumeration](../../parked/research/angles/region-enumeration.md) and [low-dim-surrogates](../../parked/research/angles/low-dim-surrogates.md).
- Why the error set is high-dimensional: [errors.md](errors.md).
