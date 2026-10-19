# Verification-based branch and bound

> Certify input boxes correct by bound propagation, split the rest: boxes straddling the 9-D boundary shrink like N^(-1/9).
> Linear boxes would be resolved exactly. Not run. Revive it only with cells aligned to the decision boundary, or with bounds tight enough to certify large near-tie regions.

**Status:** parked

## Cost argument
Box measures under N(0, I) are exact products of Φ differences. But the uncertified mass is (box size) × (boundary area), and in 9 effective dimensions halving it needs 2^9 times more boxes, while sampling needs 4 times more samples. Interval bounds through 10 recurrent steps with |λ(W_hh)| > 1 are also loose ([circuit.md](../../../research/findings/circuit.md)).
