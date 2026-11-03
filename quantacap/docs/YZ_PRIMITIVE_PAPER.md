# A Classical Hybrid Scalar-and-Graph Encoding: Correctness and Evaluation of Y/Z/G Primitives

**Research status:** preliminary engineering note; not peer reviewed.

## Abstract
We examine a small family of classical computational primitives called Z-bit, Y-bit, and G-graph. A Z-bit maps a deterministic seeded scalar to a logistic feature and a phase-like feature. A Y-bit combines a normalized complex amplitude pair with the Z-bit feature through convex interpolation. A directed acyclic graph propagates bounded mass and returns a normalized even/odd partition. Inspection identified two correctness problems: local amplitudes were estimated by summing complex amplitudes across an entangled subsystem, and graph path weights could grow until a hyperbolic tangent saturated, destroying distinctions among keys. We replace the former with local computational-basis marginals and the latter with mass-conserving propagation and normalized outputs. The design is deterministic and testable. No quantum advantage or superiority over conventional classical encodings is established.

## 1. Introduction
The goal is to explore compact structured numerical representations for an AI system. The scientific question is whether these representations provide measurable benefits over simpler classical alternatives under equal resource constraints.

## 2. Definitions
For a seeded scalar z outside [1,2], define s(z) = sign(z) log(1 + |z-1.5|), and b_k(z) = 1/(1+exp(k*s(z))). The phase-like feature is phi(z) = |z| mod 2π.

For a nonzero complex pair (alpha,beta), normalize by N = sqrt(|alpha|²+|beta|²). Let p = |beta/N|². The Y-bit output is pY = lambda*p + (1-lambda)*b_k(z), with 0 <= lambda <= 1. By convexity, pY lies in [0,1]. This is a classical interpolation rule.

For a DAG, initialize mass w_s=1. At each node i, distribute gamma*w_i/d_i to each of d_i outgoing neighbors, where 0 < gamma < 1. Let E and O be total mass on even and odd indices. Return etaE=E/(E+O), etaO=O/(E+O), so etaE+etaO=1.

## 3. Entangled-state correctness
A subsystem of an entangled pure state need not be pure. Summing complex amplitudes over the other subsystem is not a valid general method for obtaining a local state vector: it can create interference between amplitudes whose environmental states are orthogonal. The local computational-basis probability is instead obtained by summing squared magnitudes over all basis states with the selected local bit. The implementation uses (sqrt(P0),sqrt(P1)) as a canonical pair. This retains the marginal distribution but discards off-diagonal coherence; it is not a full reduced density matrix.

## 4. Corrections
1. Validate finite inputs and reject invalid graph parameters.
2. Bound logistic exponents to avoid overflow.
3. Use SHA-256 for stable key-to-node mapping across Python processes.
4. Normalize graph output by total propagated mass rather than applying tanh to unbounded path counts.
5. Compute local subsystem marginals from probabilities, not amplitude sums.

## 5. Test plan
Tests cover seeded determinism, forbidden-band exclusion, bounded scalar features, Y-bit normalization, graph normalization, invalid-input handling, and Bell-state marginal probabilities. These are correctness tests, not performance evidence.

## 6. Evaluation protocol
Compare binary, floating-point, Y/Z-only, and Y/Z/G variants on identical tasks and data splits. Match parameter count, memory, compute budget, and seeds; report wall-clock time, peak memory, accuracy/loss, calibration, and confidence intervals; include ablations. Select hyperparameters on validation data only. Replicate on held-out tasks and compare with strong conventional baselines.

## 7. Limitations
The design is classical. It does not implement quantum gates, quantum measurement, entanglement, or physical qubits. The phase-like feature is not a physical quantum phase. The Y-bit probability blend is a heuristic regularizer. G-graph compresses a potentially large graph history into two numbers and may discard useful information. The present tests do not demonstrate predictive advantage, compression advantage, or complexity improvement.

## 8. Conclusion
The corrected Y/Z/G implementation has clearer mathematical semantics and avoids two identified failure modes. Whether it is useful for AI computation remains open. Controlled benchmarks against matched classical baselines are required before claims of superiority.
