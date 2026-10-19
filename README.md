# The runner-up

A 432-parameter recurrent network, taken apart. The Alignment Research Center trained a 16-neuron ReLU RNN to read
ten numbers one at a time and point to the position of the second-largest, published its weights, and asked whether
anyone could explain them. This repository holds the explanation, deduced from the weights, and a site that tells
it to a first-year student with figures computed live from those weights.

**The site:** https://skyway1111.github.io/RNN432/

## What is here
- `web/` is the site. It opens from disk with no build step: `web/index.html` is the story, `web/explorer/` a 3-D
  sandbox for clicking any neuron or weight. `python -m web.build` regenerates `web/data.js` from `results/`;
  `python -m web.check` is the site's gate (headless Chrome via Playwright).
- `az/` is the model's library: weights, batched forward pass, traces, the rare-event yardstick. `python -m az.check`
  runs its gate.
- `experiments/` holds one script per measurement, each opening with the question it answers; `results/` holds their
  committed outputs, which every number on the site is read from.
- `data/` holds the example model and the five sibling networks with their training checkpoints, as ARC released
  them.

## Credits
The network and its siblings are ARC's. The site's typefaces are Fraunces and Source Serif 4, under the SIL Open Font
License (`web/fonts/OFL.txt`).
