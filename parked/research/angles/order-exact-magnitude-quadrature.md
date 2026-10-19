# Order-exact, magnitude-quadrature

> Accuracy = E over magnitude sets y of acc(y): orderings combinatorially, magnitudes by quadrature; acc(y) ≈ 1 off near-ties.
> acc(y) is low-dimensional, but its near-tie part depends on the order of the lower ranks, and one exact acc(y) costs 2.3e9 ops.

**Status:** parked

## What holds
- At typical magnitudes the failures are one family, 2nd max first and max last: 1,403 of 10! orderings ([orderings.md](../../../research/findings/orderings.md)).
- acc(y) is smooth on a (S−T, M−S) grid. Gaps and sign count explain 81–85% of its variance held out, against 16–28% of per-input error.

## Why it is parked
- The error sits at near-ties, where acc(y) depends on more than the gaps. With T fixed, redrawing y_4..y_10 moves acc(y) by sd 0.03–0.04 at S−T = 0.02. Plugging in the mean set biases it by up to 0.02.
- Near-tie failures do not reduce to a table of (p, q, r) families: 90% of errors spread over 62–351 of the 720 families, and a per-family majority vote misses acc(y) by up to 0.04.
- `exact_acc` costs 2.3e9 ops per quadrature node, 230 times the largest scoreboard budget, 1e7 ops ([benchmarks.md](../../../benchmarks.md)).

## What would revive it
A rule giving acc(y) at a near-tie from how the order of the lower ranks shifts the S-versus-T margin, at far fewer than 10! orderings.
