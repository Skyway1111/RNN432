# Screening by the reference route

> Run the model only where the reference route's linear margin predicts an event, weighted by 1/acceptance: the route misses the tail.
> The route per (p, q) is exact on 1–8% of inputs and correlates 0.22–0.47 with the margin; screening loses to sampling at c = 2 on all six bases and at c = 8 on five.

**Status:** dead

## Idea
Accept a candidate direction with probability a(u) from a surrogate that costs a tenth of a pass, run the model on the accepted ones, and weight each by 1/a. The estimate stays unbiased for any a > 0, so a surrogate that keeps the events and drops the rest would cut the cost per candidate tenfold. The surrogate: per (p, q), the most common gate pattern of the clean base among inputs with that order, whose logits are a linear map of the input ([votes.md](../../../research/findings/votes.md)).

## Why it died
Raw output: [tail_lines](../../../results/experiments/tail_lines.txt), part 2. The reference route reproduces the copy's margin exactly on 1.1–7.7% of inputs and correlates 0.22–0.47 with it. Accepting with a = 1 where the route's margin is within 2 of the level and 0.02 elsewhere gives a screening gain of 0.10–0.24 at c = 2 and 0.15–0.38 at c = 8 on five bases. On s3 at c = 8 it gives 12.6, because there the tail is one regime whose route is near the reference. The tail lives off the typical routes, at ties and in the all-negative regime ([tail.md](../../../research/findings/tail.md)), which is where a typical-route surrogate is wrong.
