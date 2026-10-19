# Additive presumption over the lower inputs

> Each lower input's exact 1-D effect on the tie margin, summed: the margin's law would be a convolution of seven exact laws.
> Its P(m_T < 0) lands within 0.007 of the truth, but its wedge depth E[relu(−m_T)] is 2–16% too large in all six models. The cascade is sub-additive, and the pairwise terms follow no rule of the two effects, so no cheap correction exists.

**Status:** dead

## Idea
At the tie S = T, write the tie margin m_T as m(z*) plus, for each lower input, the change from moving that input alone from the anchor z* to its value. Each change is a 1-D piecewise-linear function, exact by tracking its line through the network, so the sum's law is a convolution of seven exact 1-D laws. The presumption is that the lower inputs' effects add.

## Why it died
Raw output: [lower_ranks](../../../results/experiments/lower_ranks.txt), part 2, 2,048 configurations × 256 draws per model.

| Fact | s0/0 | example/0 | s4/0 |
|---|---|---|---|
| P(m_T < 0) bias; anchor at the conditional mean, at the expected order statistics | −0.003, −0.007 | −0.005, −0.001 | −0.003, −0.002 |
| E[relu(−m_T)] relative bias, same anchors | +0.046, +0.020 | +0.076, +0.038 | +0.030, +0.021 |
| All 21 pairwise terms added, order-statistic anchor: P bias; E[relu(−m_T)] | −0.001; +0.043 | +0.002; +0.013 | −0.001; +0.034 |

Over six models the wedge depth comes out 3–16% too large at the mean anchor and 2–4% at the order-statistic anchor. Through the wedge, that bias carries into the T-end error mass, 0.07–0.15 of accuracy, so the estimate errs by 0.002–0.005. Seven exact lines per configuration put the estimate near 1.6e9 ops, where sampling's RMSE is 4.6e-4 ([requirements](../requirements.md)).

The sum is too deep because the cascade is sub-additive. When two inputs each lower the margin, their pairwise term is positive in 57% of pairs and adds back about a tenth of their sum. That tilt follows no rule of the two effects (R² 0.002–0.019), so a product form such as m0 + (1 − Π(1 − c g_i)) / c gains nothing.

## What this closes
Presumptions that add the lower inputs' exact effects, at any anchor, with or without a correction keyed on the effects. The deduced ladder in [errors.md](../../../research/findings/errors.md) prices the alternative: each lower input made exact cuts the bias about 3×, but adds its slot and value to the configurations a deterministic estimate must cover.
