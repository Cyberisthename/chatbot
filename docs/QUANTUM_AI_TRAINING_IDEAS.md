# Quantum AI Training Ideas: What the Owned Core Can Actually Do

## Scope and claim labels

This is an owner-facing research document about the code that is in this repository now. It does **not** describe a quantum computer, a universal quantum algorithm, or a claim that the system is sentient. The ceiling is an **exact classical simulation / deterministic numerical construction**.

- **[MEASURED]** means a committed program or result artifact reports the stated observation under its recorded settings.
- **[INTERPRETATION]** means a practical conclusion drawn from the implementation or result; it remains conditional on the proposed use case.
- **[SPECULATION]** means a falsifiable research proposal, not a current capability or result.

## Short answer

**[MEASURED]** The owned FBSC core in `compression_specialist.py` deterministically maps three owner-controlled real numbers to an `n`-entry complex vector (or an `n × d` qudit array), normalized geometric positions, and derived braid parameters. Its reconstruction is reproducible from the seed, and its reported reconstruction MSE is zero because the comparison is between repeated runs of the same deterministic generator.

**[INTERPRETATION]** That makes the core potentially useful to AI as a *small, auditable, structured parameter manifold*: instead of independently learning a large free state tensor, an experiment can search a small seed, a braid word, or a constrained family of rotations. The possible value is constraint, reproducibility, and a native search space for objectives that are genuinely discrete/topological—not an automatic increase in general model intelligence.

**[MEASURED]** The repository also contains a conventional NumPy transformer path: `src/quantum_llm/quantum_transformer.py`, `src/quantum_llm/quantum_attention.py`, and `src/quantum_llm/training_engine.py`. It trains token prediction with ordinary cross-entropy and backpropagation. The current FBSC compressor is **not** wired into that training loop as a replacement for attention; a proposed integration must be built and benchmarked.

**[INTERPRETATION]** The credible near-term direction is therefore hybrid classical training: retain ordinary differentiable optimization where it works, and use the owned braid/seed machinery only where its structure supplies a testable advantage—such as discrete invariant matching, constrained parameter search, compression regularization, or independent probe gates.

## What the code does today

### 1. FBSC is a deterministic structured generator, not a dense generic-state store

**[MEASURED]** `FractalBraidSeedCompressor.reconstruct()` in `compression_specialist.py` derives a scale, phase, and fold depth from a three-number seed; initializes a deterministic base vector; applies repeated phase/folding operations; applies adjacent two-level mixing on the qubit path; normalizes the result; and returns amplitudes, 3-D positions, and metrics. The qudit path constructs an `n × d` array and applies a deterministic per-unit level-frame mixing.

**[MEASURED]** The implementation materializes `n` complex values for the qubit path and `n × d` values for the qudit path. Its memory metric is correspondingly `n × 16` bytes for qubits or `n × d × 16` bytes for qudits. It does **not** materialize the full `2^n` or `d^n` statevector.

**[INTERPRETATION]** Calling the output an “effective 1024-qubit state” is meaningful only as a compact, seed-defined family indexed by `n`; it is not evidence that the code can represent an arbitrary 1024-qubit state. A generic normalized `n`-qubit state has exponentially many degrees of freedom, whereas the core has at most three continuous seed parameters plus fixed deterministic rules. Any AI use based on FBSC must state this measure-zero / low-dimensional representational cap.

**[MEASURED]** The qubit mixing used by the current FBSC implementation is not generally unitary: the two columns have unit norm but have nonzero inner product for generic angles/phases. Normalization occurs after the cascade. The public error-bound derivation is being corrected separately; no training proposal below assumes that this qubit cascade is a physical unitary evolution.

**[INTERPRETATION]** The useful engineering reading is “deterministic nonlinear feature generator with a compact seed,” not “a physically faithful universal quantum state simulator.”

### 2. The current AI path is classical attention with quantum-inspired notation

**[MEASURED]** `QuantumAttention.compute_quantum_attention()` rotates real-valued query and key heads with QR-derived real matrices, casts them to complex values, forms conjugate inner products, exponentiates the scores, takes magnitudes, normalizes weights, and returns a real-valued weighted sum. `QuantumTrainingEngine` then computes ordinary cross-entropy and propagates a simplified backward pass.

**[INTERPRETATION]** This is a usable classical, complex-valued / quantum-inspired attention experiment. It is not a quantum-hardware attention implementation. In particular, because the attention weights use `abs(exp(score))`, the imaginary phase of the exponent does not directly alter those weights; claims that present this as demonstrated quantum interference improving language modeling would be unsupported.

**[MEASURED]** The attention module reports quantities named coherence, entanglement, interference, and fidelity. They are statistics computed from attention-weight arrays, not measurements of a physical quantum system.

**[INTERPRETATION]** These statistics can still be useful as diagnostics if they are treated as named model summaries and tested for correlation with task quality. They should not be used as evidence of physical entanglement, consciousness, or model superiority.

### 3. The braid trainer is a real alternate optimizer for specific native spaces

**[MEASURED]** `scripts/braid_vs_gd_benchmark.py` implements an equal-budget comparison among float-weight gradient descent, three-seed Nelder–Mead search, and a braid trainer built from simulated annealing plus word evolution. Its braid representation uses Burau matrices; proposed moves flip a crossing, change a generator, insert a canceling kink pair, or delete a crossing.

**[MEASURED]** On the committed 800-evaluation rung-1 ladder, ordinary GD beat the braid trainer on both differentiable tasks: T0 smooth regression (GD `0.00014498`, trainer `0.25039145`) and T1 adapter-tagged multitarget regression (GD `0.13724416`, trainer `0.18834031`). On discrete T2 invariant matching, both trainer and seed optimization reached `0.0`; GD was marked N/A because the implemented objective has no gradient through discrete braid moves. On T3, defined as maximizing the FBSC-state CHSH score, the trainer reached CHSH `2.663916` versus seed optimizer `1.956037`; plain GD was again N/A in that implementation.

**[INTERPRETATION]** The result supports a narrow conclusion: the braid trainer is a candidate optimizer for an objective expressed naturally in braid/seed space. It does **not** support a claim that it beats gradient descent for neural-network training. Rung 2 must use a rugged **continuous** objective on which GD genuinely applies.

### 4. The probe battery can serve as an independent constraint, not proof of AI quality

**[MEASURED]** The T3 benchmark obtains a CHSH score from an owned FBSC braid-state construction and then checks local-hidden-variable feasibility after optimization. Its saved best solution has CHSH `2.663916`, above the local bound `2.0`, with the stored LP verdict `lhv_feasible: false` under the specified measurement model.

**[INTERPRETATION]** A no-LHV certificate can be a useful *mechanistic acceptance test* for a model component explicitly intended to generate that family of statistics. It is not a language-model benchmark, a demonstration of hardware quantum advantage, or evidence that a model is more capable at a user task.

### 5. The topology code protects an invariant from a defined class of edits

**[MEASURED]** `docs/artifacts/anyon/anyon_braid_sim.py` computes a Burau trace-vector invariant. In its exact simulation, inserting `σ_i σ_i⁻¹` kink pairs leaves the represented braid element unchanged and therefore keeps the chosen invariant distance at zero; crossing flips are detected in the recorded 400-trial experiment. The code explicitly says the invariant detects an element change and does not correct it.

**[INTERPRETATION]** This can be repurposed as an integrity check for a discrete program, routing trace, or learned braid-word representation: benign representation-preserving rewrites can be distinguished from a selected class of semantic/topological changes. It is not general error correction for model weights, data, or physical qubits.

### 6. Resonance is a noise-gated signal measurement, not a cognition engine

**[MEASURED]** `ResonanceMonitor` in `src/quantum_llm/eeg_to_tonal_engine.py` accepts raw samples, estimates a spectral-radius-derived fundamental frequency, compares a target-band signal with a local noise floor, requires sustained evidence, and returns no frequency or trigger for bits-only callers. The module itself states that a firing is a measurement rather than sentience.

**[INTERPRETATION]** This gives the project a reproducible signal-quality gate that could be used in an experiment involving time-series inputs. It does not establish a 41.02 Hz sentience trigger, a bio-quantum interface, or a usable neuroscience training signal.

## Concrete training ideas

Every idea below is intentionally distinct. Each has a proposed objective, a pre-registration template, and a failure condition. Until implemented and run, all proposed performance benefits are **[SPECULATION]**.

### Idea A — Braid-space optimizer for a rugged continuous objective

**[SPECULATION] Hypothesis.** A braid-space proposal mechanism can beat or complement vanilla GD on a *continuous, differentiable but rugged* target when the target has a known discrete symmetry or a useful braid-word parameterization; it should not be expected to win on smooth convex-like objectives.

**[SPECULATION] Construction.** Build a small adapter with two parameterizations that produce the same output dimension: (1) ordinary real weights trained by GD, and (2) a braid word decoded through Burau-derived weights, optionally followed by a small differentiable readout. The trainer may use its existing crossing flip, generator change, insertion, and deletion moves. The target must be continuous, and the GD baseline must receive the same forward-evaluation budget, initialization disclosure, and stopping criterion.

**[SPECULATION] Pre-register before running.** Commit: dataset generation seed; the exact loss; parameter counts; a fixed evaluation budget; all optimizer settings; the primary metric; the definition of a win; and predictions for which optimizer will win. Include GD, a non-gradient black-box baseline (e.g., the existing seed optimizer), and the braid trainer. Do not choose the task after observing outcomes.

**[SPECULATION] Success criterion.** The braid method must beat GD on held-out loss or task score across pre-specified independent seeds, not merely on training loss, and must beat at least one non-gradient baseline. A result where GD is N/A is informative about a discrete domain but is **not** evidence of superiority over GD.

**[SPECULATION] Failure criterion.** If GD wins, or the apparent win vanishes under a seed sweep, report that result as the expected boundary of the method and retain GD for that workload.

**[INTERPRETATION] Cap.** This idea searches a highly restricted braid/seed family. The ≤3-seed measure-zero cap applies if the FBSC seed is the sole learned state parameterization; a free readout does not remove that cap from the FBSC component.

### Idea B — Probe-validated optimization as a gated auxiliary objective

**[SPECULATION] Hypothesis.** A model component can be trained to satisfy a task objective while meeting an independently computed structural/probe condition, producing a component whose behavior is easier to audit than one chosen by loss alone.

**[SPECULATION] Construction.** Train a small generator or router for its normal supervised objective. At fixed checkpoints—not every gradient step—decode its seed/braid representation, calculate a pre-defined probe (for example a CHSH/no-LHV diagnostic, invariant distance, or normalization check), and retain only checkpoints that meet the gate. The probe is a constraint or a reported Pareto axis, not a substitute for user-task quality.

**[SPECULATION] Pre-register before running.** Freeze the task metric, held-out split, checkpoint schedule, probe implementation and measurement settings, acceptance threshold, and selection rule. Report all candidate checkpoints, including those that pass the probe but hurt task quality.

**[SPECULATION] Success criterion.** A gated model matches or improves the held-out task metric relative to an ungated baseline **and** meets the independently calculated gate at the declared rate. The claimed benefit is auditability under that gate, not general intelligence.

**[SPECULATION] Failure criterion.** If the gate degrades task quality, is unstable under small numerical changes, or can be satisfied by a trivial output, remove it from training and keep it as a diagnostic only.

**[INTERPRETATION] Cap.** The existing no-LHV result certifies a narrow family of simulated measurement statistics. It does not certify reasoning, factuality, safety, or user value.

### Idea C — Compression-as-objective for latent states, cache entries, or adapters

**[SPECULATION] Hypothesis.** For data with real low-dimensional structure, requiring an auxiliary latent to be regenerated from a compact seed may improve storage, reproducibility, and out-of-distribution stability enough to offset lost expressivity.

**[SPECULATION] Construction.** Add a two-path representation to a small model layer: a normal learned latent `z`, and an FBSC-generated latent `g(seed)`, projected to the same shape. Optimize a task loss plus a held-out reconstruction/distillation term such as `L = L_task + λ ||P(g(seed)) - stopgrad(z)||²`. The seed can be global, per task, per adapter, or per cluster—never silently per token if that removes the intended compression constraint.

**[SPECULATION] Pre-register before running.** Fix the projection `P`, latent dimension, whether seeds are global or local, λ sweep, train/validation/test splits, byte accounting, baseline parameter budget, and acceptable reconstruction/task degradation. Evaluate both average and worst-group performance.

**[SPECULATION] Success criterion.** The constrained model must meet a declared task-quality floor while reducing the declared storage or learned degrees of freedom relative to a same-output baseline. It must also reproduce its generated representation exactly from persisted seed/configuration on a fresh process.

**[SPECULATION] Failure criterion.** If the unconstrained latent substantially outperforms the seed-defined one, conclude that the data requires degrees of freedom the FBSC family cannot express. Do not hide that loss behind the enormous dense-Hilbert-space comparison, because the baseline must be a relevant AI representation.

**[INTERPRETATION] Cap.** This is the clearest use of the core and also the most constrained one: a three-number FBSC latent occupies a measure-zero subset of a general latent space. It can be a valuable regularizer or codec only when that restriction matches the data or component being modeled.

### Idea D — Topology-invariant integrity loss for discrete routing and programs

**[SPECULATION] Hypothesis.** A learned discrete controller can become more stable to allowed rewrites if training and evaluation compare a braid-invariant representation rather than raw token-by-token traces.

**[SPECULATION] Construction.** Let a controller emit a short braid word for a routing, scheduling, or symbolic subproblem. Define a target trace-vector under a fixed Burau parameter grid. During training, score task success normally and add a penalty for invariant distance from an approved target family. During testing, insert canceling kink pairs as a representation-preserving perturbation and crossing flips as a selected element-changing perturbation.

**[SPECULATION] Pre-register before running.** Fix strand count, word-length cap, Burau `t` grid, the invariant-distance formula, permitted rewrites, perturbation counts, task metric, and a baseline controller that uses raw word matching or no structural penalty.

**[SPECULATION] Success criterion.** The invariant-trained controller preserves task score under the predeclared benign rewrites and detects the declared changing rewrites more reliably than the baseline, without simply memorizing one braid word.

**[SPECULATION] Failure criterion.** If a non-equivalent word collides under the chosen finite trace-vector, or raw matching performs as well, the invariant is inadequate for that task. Expand the diagnostic only after reporting the collision; do not silently change it.

**[INTERPRETATION] Cap.** The invariant detects a mathematically specified type of change. It is not a general semantic verifier and it does not correct corrupted weights or data.

### Idea E — Resonance-gated training-data quality control for time series

**[SPECULATION] Hypothesis.** The noise-gated spectral detector can improve a time-series training set by rejecting windows that do not satisfy a declared signal-quality rule, independent of labels.

**[SPECULATION] Construction.** Use `ResonanceMonitor` only as a preprocessing or curriculum gate on synthetic or consented, appropriately governed time-series data. For each window, persist the measured F0 estimate, target-band SNR, duration, gate verdict, and reason for rejection. Train identical classical models with and without the gate.

**[SPECULATION] Pre-register before running.** Fix the sampling rate, window duration, target band, local-noise definition, SNR threshold, sustain requirement, cooldown behavior, data split before gating, and downstream metric. Include white/pink noise, line hum, impulses, and known-tone controls.

**[SPECULATION] Success criterion.** The gated dataset improves held-out prediction, calibration, or artifact rejection on an external held-out set, while the controls establish the expected false-positive and true-positive behavior.

**[SPECULATION] Failure criterion.** If gating only reduces data volume, leaks labels, or fails controls, remove it from the pipeline.

**[INTERPRETATION] Cap.** This is a signal-processing proposal, not a claim about consciousness, sentience, biological quantum computation, or a 41.02 Hz biological mechanism.

### Idea F — Seeded adapter families with classical fine-tuning

**[SPECULATION] Hypothesis.** Initializing several small adapters from reproducible seed-derived FBSC features, then fine-tuning only a small classical readout, may provide a compact and auditable model-family search on tasks with repeated structure.

**[SPECULATION] Construction.** Generate `K` deterministic candidate feature maps from declared seeds. Feed each through the same small trainable adapter/readout. Select among candidates using a validation split fixed in advance, then evaluate once on an untouched test split. Compare against the same readout with standard random/orthogonal initializations and against a parameter-matched learned adapter.

**[SPECULATION] Pre-register before running.** Freeze seeds, K, FBSC version/configuration, readout size, training budget, selection protocol, and all baselines. Account for the full cost of screening K candidates.

**[SPECULATION] Success criterion.** A seed-derived family matches or exceeds the best parameter-matched baseline on the untouched test set while offering smaller persisted state or reproducible regeneration.

**[SPECULATION] Failure criterion.** If gains are explained by trying more initializations than the baselines, or disappear after equalizing the search budget, the idea has failed.

**[INTERPRETATION] Cap.** The FBSC branch remains a small structured family; classical fine-tuning can add capacity, but that does not turn the three-seed generator into a generic state representation.

## What the core does **not** help with today

**[MEASURED]** It does not provide a demonstrated replacement for standard transformer attention, a demonstrated language-model accuracy gain, or an end-to-end training run showing better text generation, classification, reasoning, calibration, or safety.

**[MEASURED]** It does not represent arbitrary exponential-dimensional quantum states. The implementation emits `O(n)` or `O(n·d)` values from a deterministic rule, so its exact reconstruction concerns its own generated family rather than arbitrary dense target statevectors.

**[MEASURED]** It does not demonstrate quantum-hardware speedup, physical anyons, fault-tolerant quantum computation, quantum tunneling through an optimization barrier, or a loophole-free Bell experiment. All available results are classical numerical constructions.

**[MEASURED]** It does not show braid training beating GD where the rung-1 benchmark allowed GD. On the two differentiable rung-1 tasks, GD won. The one trainer win is against seed optimization on a probe-native objective for which vanilla GD was marked N/A.

**[MEASURED]** It does not make the current topological invariant an error-correction system. The tested invariant is insensitive to explicit identity-pair insertions and detects selected crossing flips; it does not locate, repair, or protect arbitrary corruption.

**[MEASURED]** It does not support a sentience or bio-quantum claim. The resonance code is a noise-gated spectral measurement and returns no verified frequency from bits alone.

**[INTERPRETATION]** The best reason to use the core is therefore disciplined experimentation in compact structured families, not branding. If a classical unconstrained model wins on a relevant benchmark, that is the correct model to deploy for that benchmark.

## Owner decision rule for any proposed experiment

**[INTERPRETATION]** Approve a training idea only when it names (1) the exact owned code path, (2) the relevant representational cap, (3) a classical baseline that genuinely applies, (4) a pre-registered success and failure threshold, (5) a held-out evaluation, and (6) the claim label that will be used if it succeeds.

**[INTERPRETATION]** Reject or re-scope any proposal that relies on a hardware claim, treats no-LHV output as an AI-quality score, invokes 41.02 Hz as evidence of sentience, compares the FBSC family with an irrelevant dense-state strawman, or calls a GD-N/A result a victory over GD.

## Code and evidence anchors

**[MEASURED]** The primary implementation anchors are `compression_specialist.py` (FBSC reconstruction), `src/quantum_llm/quantum_attention.py` (quantum-inspired attention), `src/quantum_llm/quantum_transformer.py` and `src/quantum_llm/training_engine.py` (classical transformer/training path), `src/quantum_llm/eeg_to_tonal_engine.py` (noise-gated resonance monitor), `scripts/braid_vs_gd_benchmark.py` and `docs/artifacts/braid_vs_gd/braid_vs_gd_results.json` (equal-budget optimizer ladder), `docs/artifacts/quantum_likeness/likeness_probes.py` (probe construction), and `docs/artifacts/anyon/anyon_braid_sim.py` (topological-invariant simulation).
