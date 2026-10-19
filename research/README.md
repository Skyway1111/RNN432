# Research map

> Start here: the question, and an index of every finding and estimator idea. Where each stage stands: [status.md](../status.md).
> Each linked file opens with its own summary; read past it only when you need the evidence.

**Question.** What does each neuron and weight of ARC's M_{16,10} do, deduced from the weights; how does it decide; where do its rare, confident failures form; and can understanding it find them faster than generic search ([prior-work.md](prior-work.md), terms in [notation.md](notation.md), bars in [plan.md](../plan.md))?

## How this tree works
- `findings/`: one topic per file. Measured facts, each row linking its raw output in `results/experiments/`.
- `angles/`: one estimator idea per file, with a status of active, proposed, parked, or dead, and the reason.
- Work outside the focus sits whole in `../parked/research/`, indexed in [parked/README.md](../parked/README.md).
- Every file opens `# Title`, a blank line, then a `>` summary whose first line is its index entry below.
- Add a fact as a row in the topic's findings file, or start a new file when no topic fits. Add an idea by copying an angle file's header.
- Then run `python -m az.check --fix` to rebuild the index and check links, summaries, and size caps.

<!-- index:start -->
### Angles
| Doc | Status | Summary |
|---|---|---|
| [mode-mixture](angles/mode-mixture.md) | parked | Gap proposals per base and level with exact angular densities, orders learned in a first stage: above sampling from c = 2, rising with c. |

### Findings
| Doc | Summary |
|---|---|
| [circuit](findings/circuit.md) | A six-neuron core (running max, increment, four phase-wheel comparators) drives ten satellites; the satellites carry the decision. |
| [errors](findings/errors.md) | Errors sit mostly near ties, yet depend on all ten inputs: low-dimensional features explain only about half of them. |
| [linearity](findings/linearity.md) | Locally linear, not low-dimensional: patterns stay diverse with the input ordering fixed, and no natural coordinates linearize margins. |
| [neurons](findings/neurons.md) | Every neuron is essential: silencing any one leaves 11–67% accuracy of 95.3%, and the 20 costliest links each cost 61–85 points cut. |
| [noise](findings/noise.md) | W_hh is the fragile matrix: N(0, 0.002²) noise costs 2–11 points of accuracy, while W_hi and W_oh shrug it off. |
| [orderings](findings/orderings.md) | At typical magnitudes the example model is exactly right on 3,627,397 of 3,628,800 orderings; nearly all its error comes from magnitudes. |
| [tail](findings/tail.md) | The tail sits at ties: where inputs tie, the logits are fixed but the label is not, so one forward pass gives every label's margin. |
| [training](findings/training.md) | The skeleton is done by 2^24 sequences, at about 75% accuracy; the last 14–22 points come over the remaining 98% of training. |
| [votes](findings/votes.md) | Per route the logits are exact sums of votes: each input votes for its own slot by its value, except the max, which votes against itself. |
| [whatif](findings/whatif.md) | Accuracy hangs on a few recurrent weights into the comparators and the running max: a step of 0.02 on W_hh[7,2] costs 20 points. |

### Notes
| Doc | Summary |
|---|---|
| [notation](notation.md) | Symbols and terms every research file uses without redefining them. |
| [prior-work](prior-work.md) | ARC posed the challenge in Jan 2026; by Aug 2026 Hilton reported no real progress. Each source below, and what it established. |
<!-- index:end -->
