# The runner-up

A 432-parameter recurrent network, taken apart. The Alignment Research Center trained a 16-neuron ReLU RNN to read
ten numbers one at a time and point to the position of the second-largest, published its weights, and
[asked whether anyone could explain them](https://www.alignment.org/blog/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters/).
This repository holds an account of what every neuron and weight does, deduced from the weights, and a site that
tells it to a first-year student with figures computed live from those weights.

**The site:** https://skyway1111.github.io/RNN432/

## Where it stands
ARC counts a model as explained when a mechanistic estimate of its accuracy matches random sampling's error at equal
compute. This account does not meet that bar. Every neuron has a role that predicts what silencing or nudging it does,
but near a tie the decision is a cascade of ReLU gates retuning one another, and predicting it costs about as much as
running the network. On accuracy, every estimator built from the structure lost to plain sampling. On rare confident
failures, an importance sampler aimed at the failure modes the weights name beats sampling, but a structure-blind
sampler gains almost as fast. The scores are in `benchmarks.md`; the 24 estimator ideas and why each fell short are
indexed in `parked/README.md`.

## What is here
- `web/` is the site. It opens from disk with no build step: `web/index.html` is the story, `web/explorer/` a 3-D
  sandbox for clicking any neuron or weight. `python -m web.build` regenerates `web/data.js` from `results/`;
  `python -m web.check` is the site's gate (headless Chrome via Playwright); `python -m web.card` renders the social
  card.
- `research/` holds the account: `research/README.md` indexes the findings, one topic per file, each fact linked to
  its raw output.
- `az/` is the model's library: weights, batched forward pass, traces, the rare-event yardstick. `python -m az.check`
  runs its gate.
- `experiments/` holds one script per measurement, each opening with the question it answers; `results/` holds their
  committed outputs, which every number on the site is read from.
- `data/` holds the example model and the five sibling networks with their training checkpoints, as ARC released
  them.

## Credits
The network and its siblings are ARC's. Earlier analyses found the record keeper, the jump and three of the
comparators (ARC), the fourth comparator
([Chen](https://drive.google.com/file/d/1USCD1LPa6hXMTj7PAO6Jmc4dVdxemZY3/view)), a linear fit for the comparators
([Goldowsky-Dill](https://gist.github.com/NixGD/2651be79bfd22bcb13d1f609f23439a4)) and the clock-like readout
([Keane](https://gist.github.com/mrsirrisrm/d6850ff8647d1ed2f67cc92d5bce3ed0)); this account builds on theirs. The
site's typefaces are Fraunces and Source Serif 4, under the SIL Open Font License (`web/fonts/OFL.txt`).
