# Prior work

> ARC posed the challenge in Jan 2026; by Aug 2026 Hilton reported no real progress. Each source below, and what it established.
> Known so far: running-max neurons and comparators (ARC), an empirical linear fit for four comparators, and a phase-wheel readout (comments). On M_{4,3}, ARC's own estimators reached 3–5% RMSE against sampling's 0.8% at equal ops, so matching sampling on M_{16,10} would be new.

The origin: Hilton's question, [hilton-question.png](hilton-question.png). Local copies in `_ext/` (gitignored; re-fetch from the URLs below).

| Source | What it gives us |
|---|---|
| [ARC, AlgZoo post](https://www.alignment.org/blog/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters/), [code and weights](https://github.com/alignment-research-center/alg-zoo) | The challenge and its metric; neurons 2, 4 (running max), 6, 7, 1 (leave-one-out max comparators); handcrafted 22-neuron model at 99% |
| [ARC, Competing with sampling](https://www.alignment.org/blog/competing-with-sampling/) | The matching sampling principle; "mechanistic" means deductive; advice may be unverifiable; cumulant propagation solves random MLPs |
| Hilton, Heuristic explanations for 2nd argmax models (Dec 2024, [Drive](https://drive.google.com/file/d/1t2rXhDQqWrOXRwjqstbrBPWiCmYlMts_/view)) | M_{4,3} estimators, the noisy-weight MSE protocol (σ = 0.05, 256 models), op counting by multiply-adds |
| Robinson, Symmetric RNNs ([Drive](https://drive.google.com/file/d/1qH61_WshTrgYh0YLj5vkONzk8x00zImN/view)) | Weight symmetries reduce M_{4,3}-sized nets to absolute-value networks |
| Chen, RNNs for the 2nd argmax ([Drive](https://drive.google.com/file/d/1USCD1LPa6hXMTj7PAO6Jmc4dVdxemZY3/view)) | First analysis of this model: N2 running max, N4 = ReLU(x − max), N7 = ReLU(max − x), N8 ≈ ReLU(2nd max − x); the 4 negative-input neurons are "very entangled" |
| [Goldowsky-Dill, comment and notebook](https://gist.github.com/NixGD/2651be79bfd22bcb13d1f609f23439a4) | Recurrent input to N1, N6, N7, N8 ≈ a(δ)·M + b(δ)·S⁺, R² 0.99, fitted from samples; Σ_j W_hh[n,j]·w_ih[j] ≈ 0 cancels the input term |
| [Keane, comments and gist](https://gist.github.com/mrsirrisrm/d6850ff8647d1ed2f67cc92d5bce3ed0) | Two-impulse inputs reach 100%; canonical activation sequence after the 2nd impulse; phase-wheel readout; forward and reverse cases differ |
| [Hilton, IPAM talk](https://www.youtube.com/watch?v=N0hhsESjKEs) (transcript `_ext/talk.txt`) | Mechanistic ≠ human understanding; running inputs through the model is fine if what you conclude is deductive |
| [Cammarata, Neural Networks Are Already Clean](https://already-clean.goodfire.com/essay.md) | Neurons as control flow and routes; explanations that can lose; agents doing normal science at scale |
| [Hilton, Cammarata et al., Understanding RL Vision](https://distill.pub/2020/understanding-rl-vision/) | Model editing as quantitative validation of an interpretation |
