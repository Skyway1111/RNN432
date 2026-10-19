# Benchmarks

Canonical numbers; other docs point here. The RNN's estimator protocol: `parked/README.md`. The accuracy control's
scores: `parked/benchmarks.md`.

## The rare-event yardstick
Event at level c: the best other logit beats the label's by at least c·|x| (`az/rare.py`, terms in `research/notation.md`). The score is the ops an estimator needs for 10% relative RMSE over the 192 bench copies, against sampling's; its advantage is sampling's ops over its own. Any importance sampler's advantage grows with c, so progress is an edge over `ce`, the structure-blind control row, that grows with c, paired by seed salt; the advantage over sampling gives the scale. Reproduce: `python -m az.rare NAME [--suite heldout]`; full rows per budget in `results/bench/NAME_rare_SUITE.json` ([sampling, bench](results/bench/sampling_rare_bench.json), [sampling, held-out](results/bench/sampling_rare_heldout.json)).
Truth is float32 counts on 2^31 inputs per copy; the rarest copy at c = 8 has 3,544 hits, a relative error of 1.7%.

| Estimator | Reads the model's structure | c = 0 | c = 2 | c = 4 | c = 6 | c = 8 |
|---|---|---|---|---|---|---|
| truth, median probability | — | 1.3e-1 | 4.8e-3 | 3.5e-4 | 5.8e-5 | 8.8e-6 |
| sampling, analytic: ops to 10% | no | 2.3e6 | 7.3e7 | 1.0e9 | 8.4e9 | 4.8e10 |
| sampling, measured: ops to 10%; advantage | no | 2.2e6; 1.02 | 7.6e7; 0.97 | 1.2e9; 0.82 | 8.9e9; 0.95 | 4.2e10; 1.14 |
| **modes**, advantage: geometric mean over five realizations (range) | yes: the failure modes per base, as order-statistic gap proposals fitted on fresh noisy copies, each copy's orders learned in a first stage | 0.35 (0.32–0.38) | **1.64** (1.31–1.89) | **7.28** (5.99–9.21) | **34.3** (26.1–45.7) | **92.4** (74.5–107) |
| ce, the control: same, five realizations | no: Gaussian mixtures fitted to the same fresh-copy events, weights learned in a first stage | 0.22 (0.21–0.24) | 0.45 (0.39–0.48) | 1.40 (1.14–1.72) | 6.48 (4.94–7.45) | 17.2 (13.0–22.5) |

Sampling's measured advantage is 1 up to noise at every level, which confirms the harness. modes (`az/estimators/modes.py`) is importance sampling from a mixture over each base's failure modes, with exact angular densities and each copy's orders learned in a first stage ([mode-mixture](research/angles/mode-mixture.md)). Its advantage rises from 1.64 at c = 2 to 92 at c = 8, above 1 and rising in every realization. One realization's score is a draw, so its row is the geometric mean over seed salts 0 to 4, fixed before the bench run ([modes_score](results/experiments/modes_score.txt)); salt 0's per-budget rows: [held-out](results/bench/modes_rare_heldout.json), [bench](results/bench/modes_rare_bench.json). On the held-out suite, where the design was settled, it scores 1.68, 8.02, 27.8 and 57.2.
ce (`az/estimators/ce.py`) is modes with full-covariance Gaussians, the cross-entropy method's family, in place of gap proposals; its design was chosen on the held-out suite. Its advantage grows with c too, so growth alone does not show that the structure pays. Paired by realization, modes beats it by 3.7 at c = 2 and about 5 from c = 4 to 8 ([ce_score](results/experiments/ce_score.txt)). A realization's c = 8 score rests on a few copies' single heavy draws, so the expectation is scored too, from each copy's exact mismatch with each estimator's mixtures ([expected_score](results/experiments/expected_score.txt)): modes' expected edge on the bench suite is 4.25, 4.70, 4.45 and 6.97 at c = 2, 4, 6, 8, flat through c = 6 and higher at c = 8 only because ce fails on s2, which holds 70% of its expected error there. So no estimator meets the bar yet ([mode-mixture](research/angles/mode-mixture.md)).
