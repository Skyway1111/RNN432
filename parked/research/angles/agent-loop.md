# Agent loop: Cammarata's method, ARC's scoring

> Agents write structural claims about neurons as estimator code; `az.rare` keeps whatever widens the edge over `ce`.
> Cammarata's "normal science" with ARC's scoring: explanations that can lose. This is the most direct answer to the tweet. It needs the multi-agent workflow tool, which the user must opt into.

**Status:** proposed

## Loop
1. An agent reads [research/README.md](../../../research/README.md), picks an open question (for example in [circuit.md](../../../research/findings/circuit.md)), and states a falsifiable structural claim.
2. It turns the claim into an experiment (`experiments/`, output in `results/experiments/`) and, if the claim survives, into an estimator change.
3. `python -m az.rare NAME --suite heldout` judges the change against `ce`, paired by seed salt. Findings and angle files record the outcome, and `python -m az.check --fix` reindexes.

## Success criterion
An estimator whose edge over `ce` grows with c on the bench suite, the bar of `CLAUDE.md`.
