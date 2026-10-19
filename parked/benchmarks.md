# Benchmarks, parked

Canonical numbers of parked work, as they stood when it was parked ([README](README.md)): the RNN's accuracy control.
The active scoreboard: [benchmarks.md](../benchmarks.md).

## RNN accuracy control: σ = 0.002 suite (192 models, mean accuracy 0.8590, sd 0.0653)
`mse/samp` < 1 means the estimator beats sampling at equal ops.
Reproduce: `python -m az.bench NAME --budgets ...`; full rows per budget in `results/bench/NAME_sigma0.002.json` ([sampling](results/bench/sampling_sigma0.002.json), [adf](results/bench/adf_sigma0.002.json), [gauss_margin](results/bench/gauss_margin_sigma0.002.json), [sline](results/bench/sline_sigma0.002.json), [lattice](results/bench/lattice_sigma0.002.json), [sweep](results/bench/sweep_sigma0.002.json)).
Each estimator's idea and status: `research/angles/`.

| Estimator | Reads the model's structure | Ops | RMSE | Bias | Sampling RMSE, same ops | mse/samp |
|---|---|---|---|---|---|---|
| sampling | no | 9.8e4 | 0.0566 | +0.0074 | 0.0586 | 0.93 |
| sampling | no | 1.0e6 | 0.0172 | +0.0013 | 0.0184 | 0.88 |
| sampling | no | 1.0e7 | 0.0058 | +0.0002 | 0.0058 | 1.00 |
| adf | yes | 2.3e7 | 0.5455 | −0.5284 | 0.0039 | 20068 |
| gauss_margin, level (5, 6) | yes | 6.0e9 | 0.0054 | −0.0047 | 0.00024 | 514 |
| gauss_margin, level (7, 8) | yes | 1.5e10 | 0.0027 | +0.0023 | 0.00015 | 323 |
| gauss_margin, level (9, 10) | yes | 3.0e10 | 0.0023 | +0.0017 | 0.00011 | 455 |
| gauss_margin, level (12, 12) | yes | 5.7e10 | 0.0021 | +0.0016 | 0.00008 | 767 |
| **lattice**, budget 3e5 | no: stratified deterministic sampling | 2.7e5 | 0.0272 | +0.0030 | 0.0355 | **0.59** |
| **lattice**, budget 1e6 | no: stratified deterministic sampling | 8.1e5 | 0.0158 | +0.0009 | 0.0204 | **0.60** |
| **lattice**, budget 3e6 | no: stratified deterministic sampling | 2.2e6 | 0.0096 | +0.0026 | 0.0125 | **0.59** |
| **lattice**, budget 1e7 | no: stratified deterministic sampling | 8.6e6 | 0.0051 | −0.0020 | 0.0062 | **0.67** |
| **lattice**, budget 3e7 | no: stratified deterministic sampling | 2.6e7 | 0.0026 | +0.0012 | 0.0036 | **0.52** |
| **lattice**, budget 1e8 | no: stratified deterministic sampling | 9.1e7 | 0.0011 | +0.0002 | 0.0019 | **0.34** |
| **lattice**, budget 3e8 | no: stratified deterministic sampling | 2.9e8 | 0.00061 | +0.0000 | 0.0011 | **0.32** |
| **lattice**, budget 1e9 | no: stratified deterministic sampling | 1.0e9 | 0.00036 | −0.0002 | 0.00058 | **0.39** |
| **lattice**, budget 3e9 | no: stratified deterministic sampling | 3.0e9 | 0.00017 | +0.0000 | 0.00034 | **0.26** |
| **lattice**, budget 1e10 | no: stratified deterministic sampling | 1.0e10 | 0.00012 | +0.0000 | 0.00018 | **0.43** |
| **lattice**, budget 3e10 | no: stratified deterministic sampling | 3.0e10 | 0.00009 | −0.0000 | 0.00011 | **0.75** |
| sline, budget 3e7 | no: stratified deterministic sampling | 2.5e7 | 0.0031 | −0.0013 | 0.0037 | 0.74 |
| sline, budget 1e8 | no: stratified deterministic sampling | 7.4e7 | 0.0015 | −0.0001 | 0.0021 | 0.46 |
| sline, budget 1e9 | no: stratified deterministic sampling | 8.9e8 | 0.00045 | +0.0002 | 0.00061 | 0.54 |
| sline, budget 1e10 | no: stratified deterministic sampling | 9.7e9 | 0.00012 | +0.0000 | 0.00019 | 0.44 |
| sweep, budget 3e7 | no: stratified deterministic sampling | 2.4e7 | 0.0025 | +0.0001 | 0.0038 | 0.42 |
| sweep, budget 1e8 | no: stratified deterministic sampling | 9.5e7 | 0.0011 | +0.0004 | 0.0019 | 0.37 |
| sweep, budget 3e8 | no: stratified deterministic sampling | 2.9e8 | 0.00061 | −0.0002 | 0.0011 | 0.32 |
| sweep, budget 1e9 | no: stratified deterministic sampling | 9.5e8 | 0.00034 | −0.0001 | 0.00059 | 0.32 |
| sweep, budget 3e9 | no: stratified deterministic sampling | 3.0e9 | 0.00020 | −0.0001 | 0.00034 | 0.36 |
| sweep, budget 1e10 | no: stratified deterministic sampling | 1.0e10 | 0.00012 | −0.0000 | 0.00018 | 0.42 |
| constant 0.859 (reference, not an estimator) | — | 0 | 0.0653 | 0 | — | — |

Sampling's rows confirm the analytic baseline: the ratio should be 1 up to noise across 192 models. lattice, sline and sweep are stratified deterministic sampling: they read nothing from the weights except by running them, and the same point set serves any model of the task (parked/research/angles/family-lattice.md). Their 0.3–0.6 of random sampling's MSE is a constant factor on a common event, which is why accuracy is the control and not the yardstick. The sweep, the last input's exact integral at each lattice root for a quarter of a pass, ties the lattice within its luck here (0.32–0.42 against 0.26–0.52) after a 1.3× at 1e9 on the held-out suite. Their designs were settled on a held-out suite and budgets fixed before the run. Near and above 1e10 the RMSE meets the truth's own sampling error, about 8.5e-5, so those ratios are upper bounds.
