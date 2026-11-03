# Y/Z/G Primitive Research Note

## Status
This is a corrected classical implementation and research plan. It does not claim quantum hardware, quantum advantage, or a breakthrough. "Bit" is historical project terminology; these are classical data structures.

## Z-bit
A Z-bit is a deterministic scalar sampled from a seeded NumPy generator, constrained to the real line outside [1, 2]. It exposes a logistic bias and a phase-like scalar |z| modulo 2π. The latter is a classical feature, not a physical quantum phase.

## Y-bit
A Y-bit normalizes a complex pair (alpha, beta), computes p = |beta|², and interpolates it with the Z-bit bias:
pY = lambda*p + (1-lambda)*bias(z,k).
This is a convex probability blend and can be treated as a tunable regularizer or prior. It is not a new measurement postulate.

An entangled subsystem is generally mixed and cannot be represented by a standalone pure-state amplitude pair. The experiment helper now extracts the correct local computational-basis marginal probabilities and encodes them as (sqrt(P0), sqrt(P1)). This preserves those probabilities but does not claim to preserve local coherence.

## G-graph
The prior propagation accumulated unnormalized path weights and applied tanh. In the tested 512-node graph, distinct keys saturated to identical outputs. The corrected version distributes each node's outgoing mass evenly, propagates a gamma fraction with 0 < gamma < 1, and normalizes even/odd mass by total mass. The two outputs sum to one.

## Reproducibility
- Graph and Z-bit use explicit NumPy seeds.
- Key-to-start-node mapping uses SHA-256 rather than Python's process-randomized hash().
- Tests cover normalization, determinism, invalid inputs, and Bell-state local marginals.
- Run pytest -q from the quantacap project directory.

## What would count as evidence of value?
Compare binary encoding, floating-point encoding, Y/Z without graph, and Y/Z plus graph on identical tasks. Match data, parameter count, memory, compute budget, and seeds. Report task accuracy/loss, wall time, peak memory, calibration, and variance across independent seeds. Include ablations for lambda, k, epsilon, and gamma. A useful result must replicate on held-out data and beat the strongest baseline by a meaningful margin with uncertainty intervals.

## Current conclusion
These changes repair identified correctness and numerical-stability issues. They do not yet demonstrate that Y/Z/G outperform ordinary classical representations. The next milestone is controlled benchmarking.
