# Exact lines on the rare levels

> Integrate S exactly over (T, M) at each draw: inside the modes mixture the exact tie wedges cut the worst copies' variance 4–23×.
> Each line costs 6.6–8.6 passes against modes' 1.47 per point, so it pays only on s1/12 and s2/28; on short lines alone it nets 1.5–2× on s2 at c = 8.

**Status:** parked

## Idea
Along the line where only S moves between T and M the label stays put and the model is piecewise linear in S. Per piece each other logit's gap is a + b s, and the event a + b s ≥ c √(s² + ρ²) is one interval, since the left side minus the right is concave in s. The union over logits is exact, so P(event | the rest) is exact at every level at once (`level_mass` in `az/estimators/sline.py`). Inside an importance sampler the rest R comes from the proposal and the estimate is φ_9(R) line(R) / q_R(R), with q_R the mixture's law of R, S integrated in closed form per gap component.

## Measured
On nominal rests the per-line gain over sampling is 2.8–17.5 at c = 2–8 on five bases and 6.3, 56, 107, 208 on s2 ([tail_lines](../../../results/experiments/tail_lines.txt)). Inside the modes library mixture, by exact mismatch on six held-out copies' own events ([heavy_modes](../../../results/experiments/heavy_modes.txt), part 4):

| Copy | m at c = 6, plain → S-line | m at c = 8 | m × passes at c = 8: plain, S-line on every point, on short lines only |
|---|---|---|---|
| s2/15 | 837 → 287 | 3,730 → 849 | 5,466; 6,064; 3,555 |
| s2/28 | 2,335 → 623 | 14,968 → 2,399 | 21,933; 16,871; 11,114 |
| s1/12 | 34,671 → 1,502 | 6,953 → 830 | 10,189; 5,496; 10,765 |
| s1/16, s4/3, example/0 | 1.2–2.4× lower | 1.1–2.2× lower | 2.7–4.9× higher on every point; 1.05–1.18× higher on short lines |

The gain is largest where the variance is: the three-way and top-two ties of [tail.md](../../../research/findings/tail.md), What a library misses. It grows from c = 6 to c = 8 on s2. A short line, (M − T) < 0.03 |R|, is a tie wedge, cheap to track, and a rule on R alone keeps the estimate unbiased; but s1/12's gain comes from long lines.

## Where it could pay
Wherever the line is cheaper than its variance cut. That needs a tracker that stops at the event's ends, or lines kept to the copies whose tail is a tie, which the estimator would have to tell from the copy's weights.
