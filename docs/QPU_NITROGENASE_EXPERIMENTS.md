# QPU experiments for the nitrogenase test — what 5 minutes of real hardware could and could not do

**Author:** agent-theoretical-physicist · **Status:** design document, pre-registration template only.
**Nothing in this file has been run. No hardware has been used. No QPU account exists.**
**Tags:** `[measured]` (re-runnable in this repo, numbers from an existing fresh run) ·
`[published]` (someone else's measurement, cited, not re-verified here) ·
`[analytic]` (closed-form arithmetic done in this file) · `[design]` (a circuit we would run,
untested) · `[interpretation]` · `[speculation]`.

Companion: `docs/qpu/nitrogenase_prereg_TEMPLATE.json` — the thresholds below as a
machine-checkable object, to be frozen (sha256, as in `docs/validation/BLIND_PREDICTION_2.frozen.json`)
*before* any job is submitted.

---

## 0. The honest ceiling — read this before the rest

### 0.1 What our nitrogenase result actually is

The headline ladder — **144.7362 / 129.9529 / 100.7657 / 76.0663 / 50.14 kJ/mol** — is
`[measured]` output of our own simulator (`docs/artifacts/nitrogenase_qudit/`,
`nitrogenase_qudit_simulator.py`, owner seed `(0.57721, 1.618034, 2.71828)`, n = 16 logical
qudits, sim id `nitro-qudit-c35bca531ec4`, re-verified in `DATA_FIX.md` §3).

Three things it is not:

1. **It is not chemistry.** It is a deterministic 3-number generative formula (FBSC) evaluated at
   five encoding dimensions. There is no Hamiltonian, no orbitals, no electron integrals, no
   SCF, no CASSCF, no DMRG, no basis set. Nothing in the pipeline knows what iron is.
2. **It is not a reaction pathway.** The five numbers are the *same* surrogate at d = 2, 3, 4, 5, 6
   qudit dimension. They are not five elementary steps of N₂ → 2 NH₃, and the "barrier" is not a
   barrier between two states of FeMo-co. The ladder is a dimension-scaling curve.
3. **It is not hardware.** Every number was computed in NumPy on a CPU. The d = 4 row's
   `coherence = 0.876241` and `braid_topological_protection = 0.995139` are **our own measures**
   of our own reconstructed state: `coherence_of()` is a normalised level-spread
   (participation-ratio-style) statistic and `braid_pathways()` returns overlap-under-jitter with
   a fixed `RandomState(2100+p)` seed. They are **not device properties, not measured coherence
   times, not a hardware protection factor**, and they must never be presented as such. If this
   document achieves one thing, it should be to retire those two numbers as "device-like".

Likewise `docs/validation/VALIDATION_BENCHMARK.md` compares against **analytic published-range
noise curves** — not a measured channel — and the braid-invariance result
(`docs/artifacts/anyon/`, invariant distance exactly 0.000000 across 0→40 kinks) is **exact
classical simulation of a word algebra**. The exact zero is a theorem about σᵢσᵢ⁻¹ = e in a
representation. It says nothing about a physical device, and no experiment below can make it say
anything about a physical device.

### 0.2 FeMo-co's meaningful active space is out of reach. Permanently, not just on a budget.

| requirement | number | status |
|---|---|---|
| active space used as the benchmark anchor | 54 orbitals (Reiher et al., PNAS 114, 7555 (2017); the anchor in `VALIDATION_BENCHMARK.md` §2.1) | `[published]` |
| energy resolution needed to be "chemistry" | 4.184 kJ/mol (1 kcal/mol = 1.594 mHartree) | `[analytic]` |
| smallest rung in our own ladder | 14.7833 kJ/mol (d=2 → d=3) | `[measured]`, `[analytic]` |
| total ladder span | 94.5962 kJ/mol | `[analytic]` |
| fault-tolerant resource estimate for the full mechanism, per that paper's own counting | order 10¹¹ T gates on order 10² logical qubits | `[published, order of magnitude]` |
| 10¹¹ T gates at an optimistic 1 µs surface-code cycle | 10⁵ s ≈ **28 hours** | `[analytic]` |

So: even a *perfect* fault-tolerant machine does not do this in 300 s. It is off by ~3–4 orders of
magnitude on time alone, before counting the 5-minute NISQ device we can actually rent.

Two arithmetic facts that end the "QPU computes the barrier" framing:

- **Resolution.** Our smallest rung is 14.78 kJ/mol, which is only **3.5× chemical accuracy**.
  Any energy method with absolute error above ~4 kJ/mol cannot distinguish adjacent rungs. NISQ
  variational methods on strongly-correlated transition-metal clusters are 1–2 orders of
  magnitude above that. A measured 100.8 kJ/mol from hardware would be noise dressed as a barrier.
- **State preparation.** The d = 4, n = 16 row is 4¹⁶ = 2³² complex amplitudes = **64 GiB** as a
  dense statevector. There is no known circuit that prepares the state the FBSC seed describes:
  the seed is a *classical generative formula for amplitudes*, not a state-prep algorithm. A
  generic 32-qubit arbitrary-state preparation needs O(2³²) gates — at 0.1 µs per gate that is
  **~430 s per single shot**, i.e. one shot exceeds the entire block by 40%. This is not a budget
  problem; it is a different mathematical object.

### 0.3 What a real QPU *can* do, and it is genuinely worth 5 minutes

Exactly one thing, and it is the thing our simulator cannot do for itself: **supply a measured
noise channel and a measured device-level fidelity.** Our two most exposed published numbers are
device-like and unmeasured. A QPU block converts them from *our numbers* into *measured numbers*,
and converts `VALIDATION_BENCHMARK.md`'s analytic curves into measured curves for one device on
one date.

**Framing rule for every experiment below: hardware calibrates or validates our simulator. Hardware
never computes the chemistry.** Any sentence of the form "the QPU produced X kJ/mol" is out of
scope in this repository, and the correct sentence is "the QPU measured a channel that changes the
inputs to our simulator by Y".

---

## 1. What the 300-second budget actually buys

`[analytic]`, with gate/readout times from the "best demonstrated 2025" scenario in
`VALIDATION_BENCHMARK.md` §2.2 (t_gate = 0.1 µs, T2 = 100 µs, readout error 1.0%) plus a stated
assumption of ~5 µs reset + readout per shot.

| quantity | value | note |
|---|---|---|
| per-shot cost, shallow circuit (D ≤ 10) | ~6–10 µs | `[assumption]` |
| shot-executions affordable in 300 s | **3×10⁷ – 6×10⁷** | `[analytic]` |
| shots needed for σ = 0.01 on one probability | 2 500 | `[analytic]` |
| shots needed for σ = 0.0035 | 20 000 | `[analytic]` |
| total shots in all six experiments below | 1.52×10⁶ | `[design]` |
| total QPU execution time for all six | **≈ 24 s (8% of the block)** | `[analytic]` |
| statistical floor reachable per point | σ ≈ 0.0035 | `[analytic]` |

**The conclusion that reorganises the whole plan: seconds are not the scarce resource.** Every
experiment here is affordable at 10⁴–10⁵ shots per point, which is 100× more than needed, and the
lot costs under 10% of the block. What the 300 s actually buys is **the right to be wrong and
re-run** — room for a calibration pass, a re-run after drift, and one experiment that fails its
pre-registered threshold and has to be repeated. Rank by *value*, spend seconds freely, and keep
a 3× time reserve.

Also budget for what is *not* QPU time: compile + queue + transpile (minutes to hours of wall
clock, unbounded), and — the real cost — the hours of our own analysis to write a claim that
survives review. A 5-minute block is a 5-minute *experiment* and a 2-day *publication*.

---

## 2. Ranking by value per second

Value score = how much of an *already-published* claim it de-risks or retires, 0–10 `[interpretation,
stated so a reader can disagree with the weights]`. Seconds = estimated QPU execution time from
the shot budgets in §3.

| rank (value/s) | exp. | value | QPU s | shots | value/s | claim earned | depends on |
|---|---|---|---|---|---|---|---|
| **1** | **E2 CHSH with our LP settings** | 6 | 1.2 | 200 000 | **5.0** | our analytic noise model predicts measured CHSH to ±0.08; our "0.003 off Tsirelson" becomes a measured deficit | — |
| **2** | **E4 qudit (d=3/d=4) availability** | 4 | 1.2 | 120 000 | **3.3** | d=4 encoding is physically available at measured fidelity F (or is not) | availability ⚠ |
| **3** | **E3 kink invariance under a real channel** | 9 | 2.9 | 240 000 | **3.1** | how far the exact-zero invariant claim moves under measurement — the transfer gap | E1 |
| **4** | **E6 chemical-accuracy floor** | 7 | 2.4 | 200 000 | **2.9** | measured error ε vs exact diagonalisation at 4 qubits; the gap to 4.184 kJ/mol | — |
| **5** | **E5 depth-decay, measured channel** | 8 | 3.6 | 300 000 | **2.2** | replaces the analytic F(D,n) table with a measured one for one device/date | E1 |
| **6** | **E1 device-channel calibration** | 10 | 13.0 | 460 000 | **0.77** | retires `coherence=0.876241` and `protection=0.995139` as "device-like"; supplies the channel for E3/E5 | — |

**Reading `[interpretation]`.** The ratio column is close to meaningless at this scale: every
experiment costs under 4 s except E1, and my value scores are ±2 at best, so ranks 2–5 are a tie
inside the noise. Two orderings matter and they are different:

- **Execution order:** E1 → E2 → E3 → E5 → E6 → E4. E1 goes first not because it is efficient but
  because **E3 and E5 consume its output** — the measured channel is the independent variable in
  their fits. Ranking it last by ratio while running it first is not a contradiction; it is the
  difference between a priority list and a build order.
- **If you can run exactly one thing:** E1. It is the only experiment that retires a specific
  number currently published in a form a reviewer can attack ("your coherence figure is a
  participation ratio, not a T2"). E2 wins on ratio only because it is cheap.

E4 is flagged ⚠: qudit (d ≥ 3) access is *not* offered as a supported API by the major cloud
providers at the time of writing; it may require pulse-level control that current cloud plans do
not expose `[published/uncertain — verify at booking]`. If unavailable, that claim is unobtainable
in 5 minutes, which is a different thing from unaffordable, and we should say so rather than
quietly drop it.

---

## 3. The experiments

Every experiment states: circuit, shots, pre-registered success threshold, pre-registered failure
threshold, the claim it would earn, and — because this is the discipline that keeps us honest —
the claim it would **not** earn. Thresholds are frozen in
`docs/qpu/nitrogenase_prereg_TEMPLATE.json` before submission.

### E1 — Device-channel calibration (the liability retirement)

**What it replaces.** `coherence = 0.876241` and `braid_topological_protection = 0.995139` are our
numbers. After E1 they are *measured device numbers* — and note carefully, this does **not** make
them "our protection factor validated". It *retires* them: a measured T2 and a measured two-qubit
error are a different quantity from a participation ratio, and the d=4 row must then be re-reported
with both, clearly labelled.

**Circuit** `[design]`, four components, one job:
1. **T1** — 4 qubits, `X` then `delay τ`, `measure`. τ ∈ 10 geometric points, 10→300 µs.
2. **T2 (Hahn echo)** — `X – delay τ/2 – X – delay τ/2 – measure`. 12 delays, 5→150 µs.
3. **Readout** — prepare `|0⟩` and `X|0⟩`, measure; 20 000 shots each.
4. **Two-qubit error** — interleaved randomised benchmarking on 4 adjacent pairs, 25 depths (1–25),
   2 000 shots per depth per pair.

**Shots:** 100 000 (T1: 4 qubits × 10 delays × 2 500) + 120 000 (T2: 4 × 12 × 2 500)
+ 40 000 (readout) + 200 000 (RB: 4 pairs × 25 depths × 2 000) = **460 000**.
**QPU seconds:** ≈ 13 (T1/T2 dominated by delay time, not gate time).

**Fit models** `[analytic]`: `P₁(τ) = A e^{−τ/T1} + B`; `P(τ) = A e^{−τ/T2} + ½`;
`F(m) = A p^m + B`, `e₂ = (1−p)(d−1)/d` with d = 4.

**Pre-registered success:** all four fits give χ²/dof < 3; T1 ∈ [20, 400] µs; T2 ∈ [10, 300] µs;
T2 ≤ 2T1 (physical); readout error ∈ [0.1%, 5%]; each fitted quantity's bootstrap 95% CI width
< 20% of its value.
**Pre-registered failure:** any fit χ²/dof > 5, any value outside the physical ranges, or T2 < 5 µs
(device unusable for E3/E5 → abort the block, do not run the rest, report the abort).

**Claim earned** `[would be measured]`: "on device *D* on date *T*, T1 = … µs, T2 = … µs, readout
error = … %, two-qubit error = … (95% CI). These are the coherence and error inputs to our
simulator, replacing the assumed values in `VALIDATION_BENCHMARK.md` §2.2."
**Claim NOT earned:** that 0.876241 or 0.995139 is validated; that any barrier is validated; that
the device is coherent enough for the chemistry.

---

### E2 — CHSH with the settings our own LHV-exclusion search found (the cheapest real measurement)

**Context.** Our quantum-likeness result (`docs/artifacts/quantum_likeness/`, PR #146) reports
FBSC d=2 CHSH **2.8250** at n = 2, 3 — 0.003 off Tsirelson — with LP exclusion of every LHV model.
That number is a *mathematical maximum over a class*, computed classically. A reviewer's first
sentence is "you never measured it." E2 answers that sentence in 1.2 seconds of hardware.

**Circuit** `[design]`: 4 circuits. `H(q0)`, `CNOT(q0,q1)`, then measurement basis
(aᵢ, bⱼ) for i,j ∈ {0,1} applied as `RZ(−2a) ` on q0 and `RZ(−2b)` on q1 before `measure`.
Settings (a₀,a₁,b₀,b₁) are the LP-optimal angles already in our artifact — **not re-tuned after
seeing hardware data** (re-tuning is how this experiment becomes worthless).

**Shots:** 4 × 50 000 = **200 000** → σ(E) ≈ 0.0016, σ(S) ≈ 0.003.
**QPU seconds:** ≈ 1.2.

**Estimator** `[analytic]`: `E = (N_same − N_diff)/(N_same + N_diff)`,
`S = E(a₀,b₀) + E(a₀,b₁) + E(a₁,b₀) − E(a₁,b₁)`.

**Pre-registered success:** `S_meas ≥ 2.60` **and** `|S_meas − S_model(channel)| ≤ 0.08`, where
`S_model` is our analytic `F(D)` machinery evaluated at E1's *measured* parameters.
**Pre-registered failure:** `S_meas ≤ 2.00` (no violation at all) **or** `|S_meas − S_model| > 0.15`
(our noise model is wrong in a way we must publish).
**Ambiguous band (2.00 < S < 2.60) is pre-declared as "inconclusive, not a success"** — this band is
where a motivated reading would try to claim something, so it is named in advance.

**Claim earned** `[would be measured]`: "the settings our owned LHV-exclusion search returns are
physically realisable, and on device *D* they yield S = … ± 0.003, a measured Tsirelson deficit of
… ; our analytic channel model reproduces it to within …".
**Claim NOT earned:** that our core is "more quantum" — the LP exclusion is a statement about a
classical object and is already proven without hardware. E2 upgrades *presentation honesty*, not
the claim. No loophole-free claim (locality and detection gaps are both open on any cloud device).

---

### E3 — Does the exact-zero braid invariance survive measurement? (the transfer gap)

**The trap first.** Our invariant distance of exactly 0.000000 comes from σᵢσᵢ⁻¹ = e in the Burau
representation — exact arithmetic on a word. On hardware the invariant must be **recomputed from
noisy statistics**, so the exact zero is unreachable *by construction*. The experiment's purpose is
therefore not to confirm 0.000000; it is to **measure how far the claim moves**, and to publish
that number. Second trap: a transpiler at default optimisation will *cancel* the kink pair
(σσ⁻¹ → identity) and the compiled circuit will be bit-identical at every k — the experiment then
measures drift only and confirms nothing. **Optimisation must be disabled (`optimization_level=0`),
and the compiled circuit depths must be printed and checked to differ across k. This is the single
most likely way this experiment fails silently.**

**Circuit** `[design]`: a 4-strand braid word (length 20) encoded as a CNOT/swap network on
4 qubits (strand *i* ↔ qubit *i*). Two variants per kink level:
- **isotopy** — insert k ∈ {0, 2, 4, 8, 16, 40} cancelling kink pairs, uncompiled (changes the
  *word*, not the *element*);
- **control** — insert k genuine crossing *flips* σᵢ → σᵢ⁻¹ (changes the *element*).

For each: (i) mirror-circuit survival `U` then `U†` → measured state-fidelity proxy; (ii) element
error = fraction of shots outside the support of the expected permutation coset.

**Shots:** 6 k × 2 variants × 20 000 = **240 000**.
**QPU seconds:** ≈ 2.9.

**Pre-registered predictions** `[speculation, to be tested — stated so it can fail]:**
- `P1`: element error grows with k on the isotopy arm; `e(40) − e(0) ≤ 0.05`.
- `P2`: mirror fidelity on the isotopy arm falls by ≥ 0.15 from k = 0 to k = 40 (our analytic
  prediction for a best-2025 device is 0.967 → 0.844, i.e. a drop of 0.123 — so a drop *smaller*
  than 0.15 would contradict our own model).
- `P3`: on the control (element-changing) arm, element error is ≥ 10× the isotopy arm's at every
  k ≥ 2.

**Pre-registered success:** P1 ∧ P2 ∧ P3 all hold with non-overlapping 95% CIs.
**Pre-registered failure:** any of P1–P3 refuted, **or** compiled depths identical across k
(the vacuous run — report as a void experiment, as loudly as a hit; this is the precedent set by
`docs/validation/BLIND_PREDICTION_2.md`).

**Claim earned** `[would be measured]`: "under a measured hardware channel, the word-level invariant
degrades by … while the state-fidelity proxy degrades by …, i.e. the classical exact-zero claim
transfers to measurement with a gap of … ."
**Claim NOT earned:** topological protection, anyon braiding, non-abelian statistics, fault
tolerance. There are no anyons on a transmon. The braid claim in `docs/artifacts/anyon/` remains
**exact classical simulation**, and E3 does not touch it — it measures the *gap* between it and
reality.

---

### E4 — Is the d = 3 / d = 4 row even representable on device?

**Purpose.** Not to compute anything — to separate two things our report currently blurs:
"the d=4 encoding dimension exists on hardware" and "the d=4 row's amplitudes are simulator
output". Only the first is testable, and it is worth one experiment because it is the difference
between a defensible sentence and a hand-wave.

**Circuit** `[design]`: 4 physical transmons addressed as qutrits/ququarts. Prepare
`(|0⟩+|1⟩)/√2`, `(|0⟩+|2⟩)/√2`, `(|0⟩+|1⟩+|2⟩)/√3` (and the d=4 analogues) with SU(3)/SU(4)
pulses; read out 3 (4) levels using |2⟩ → |1⟩ shelving plus a second readout pulse. 3 preparations
× 4 devices × 10 000 shots.

**Shots:** **120 000**. **QPU seconds:** ≈ 1.2. **Availability:** ⚠ uncertain (see §2).

**Pre-registered success:** SPAM-corrected 3-outcome assignment fidelity ≥ 0.90 **and** single-qudit
qutrit process fidelity ≥ 0.95.
**Pre-registered failure:** assignment fidelity < 0.80 or process fidelity < 0.90 (then: the d=4
row is *not* hardware-representable on this device class and we say so).

**Claim earned** `[would be measured]`: "d=4 encoding is available on device class *C* at measured
fidelity F; the d=4 row's *dimension* is not fiction, while its *amplitudes* remain simulator
output."
**Claim NOT earned:** that the d=4 row's 100.7657 kJ/mol means anything physically. It does not, on
hardware or off it.

---

### E5 — Replace the analytic depth-decay table with a measured one

**Context.** `VALIDATION_BENCHMARK.md` §3.1 is the most-cited table we have: at n = 32, D = 10,
expected fidelity 0.004 / 0.053 / 0.376 for three device classes, and "depth-1 collapse at n = 54".
Those are `[analytic]` with published-range parameters. This is the one place where a measured
curve can stand in for a model, and it is a mirror-circuit measurement — no state preparation, no
tomography, no ansatz.

**Circuit** `[design]`: random Clifford layers of depth D ∈ {1,2,4,6,8,10} on n ∈ {8, 16, 24, 32}
qubits, followed by the exact inverse (mirror). Survival probability = circuit fidelity. Linear
chain and a heavy-hex-like coupling map, both, so the `r2` connectivity factor in our model is
measured rather than assumed.
**Shots:** 5 n × 6 D × 10 000 = **300 000**. **QPU seconds:** ≈ 3.6.

**Pre-registered success:** ≥ 80% of the (n, D) points fall inside the 95% band of our model
re-evaluated with E1's measured channel parameters.
**Pre-registered failure:** a *systematic* deviation in the optimistic direction (model predicts
higher fidelity than measured) at more than half the points — our benchmark's domination claim is
then too strong and must be weakened in the published table.

**Claim earned** `[would be measured]`: "the depth axis of `VALIDATION_BENCHMARK.md` is now measured
for device *D* on date *T*, not analytic; the domination claim over our depth-independent core holds
with measured numbers."
**Claim NOT earned:** any speedup, any advantage, any fault tolerance. The comparison stays
"measured hardware decay vs our exact classical simulation", which is a comparison of a device to a
classical program, not a quantum advantage.

---

### E6 — The chemical-accuracy floor: the negative experiment, run on purpose

**Purpose.** We publish the number that ends the "QPU computes the barrier" story, in kJ/mol,
measured. This is the deliverable that protects every other claim in the repo, and it is the one we
should want even if it is boring.

**Circuit** `[design]`: a 4-qubit (2-site) Heisenberg/Hubbard dimer standing in for the
[2Fe-2S] singlet–triplet gap already used as our second blind-prediction target
(`docs/validation/BLIND_PREDICTION_2.md`) — the smallest system where our own chemistry claim lives.
A fixed hardware-efficient ansatz (no per-point re-optimisation on hardware), 20 parameter sets
from a pre-computed classical optimisation, 10 000 shots each, with readout-error mitigation by
calibration-matrix inversion. Compare against exact diagonalisation (a 16×16 matrix — trivially
exact classically, so the reference is not in doubt).

**Shots:** **200 000**. **QPU seconds:** ≈ 2.4.

**Pre-registered success (as designed):** the run completes and we report
`ε = |E_measured − E_exact|` with a bootstrap CI at 4 qubits.
**Pre-registered expectation** `[speculation]`: ε ∈ [5, 50] kJ/mol after mitigation.
**Pre-registered "surprise" clause:** if ε ≤ 4.184 kJ/mol at 4 qubits, that is a genuine positive
and gets published at the top of the doc as a HIT, not buried. (Chemical accuracy *has* been
demonstrated on hardware for small weakly-correlated molecules — H₂, LiH, BeH₂ — so this is a real
possibility at 4 qubits, and pretending otherwise would be dishonest.)
**Pre-registered failure:** no fitted energy (optimiser diverges) or CI width > 20 kJ/mol → report
"measurement inconclusive at this shot budget", which is itself a result about the budget.

**Claim earned** `[would be measured]`: "at the smallest system size our chemistry claim touches
(4 qubits), the measured energy error is ε = … ± … kJ/mol. The smallest rung in our nitrogenase
ladder is 14.78 kJ/mol and chemical accuracy is 4.184 kJ/mol. Measured NISQ error therefore consumes
ε/14.78 = …% of the smallest rung at 4 qubits, where the classical reference is a 16×16 matrix."
**Claim NOT earned:** anything about FeMo-co. The extrapolation from 4 qubits to 54 orbitals is
`[interpretation]` and is stated as an extrapolation, with the paper's own 10¹¹-T-gate estimate as
the anchor for the fault-tolerant end.

---

## 4. One 300-second block, end to end

Single submission, because **drift between jobs is a bigger threat to the analysis than any
statistical error here**: the whole point is one measured channel, and a channel measured in three
different jobs is three channels.

| # | stage | QPU s | cumulative | gate |
|---|---|---|---|---|
| 0 | calibrate + readout characterisation | 5 | 5 | abort if T2 < 5 µs |
| 1 | **E1** channel | 13 | 18 | abort block on failure |
| 2 | **E2** CHSH | 1.2 | 19.2 | independent |
| 3 | **E3** kink invariance | 2.9 | 22.1 | needs E1 channel |
| 4 | **E5** depth decay | 3.6 | 25.7 | needs E1 channel |
| 5 | **E6** chemical-accuracy floor | 2.4 | 28.1 | independent |
| 6 | **E4** qudit (if available) | 1.2 | 29.3 | availability ⚠ |
| 7 | **reserve** for one re-run of anything that failed | — | 300 | 90% unspent |

`[analytic]` Total ≈ 29 s, 10% of the block, 3× reserve plus 9× slack. **If the block comes in at
under 10% utilisation, that is the expected outcome, not a mistake** — the budget was never the
binding constraint.

---

## 5. Pre-registration protocol (non-negotiable)

1. Fill `docs/qpu/nitrogenase_prereg_TEMPLATE.json`: thresholds, shots, settings, analysis code
   hash, device name *left blank* until booking.
2. `sha256sum` it, commit the hash **and** the file to `main`, push. Precedent:
   `docs/validation/BLIND_PREDICTION_2.frozen.json` (freeze-first, and note that our second blind
   prediction was later reported **VOID** — the freeze discipline is what made it possible to say
   that out loud instead of quietly rewriting it).
3. Only then submit the job. The LP-optimal CHSH settings in E2 are copied from the existing
   artifact and **may not be re-optimised after seeing data** — re-optimising turns a test into a
   fit.
4. Publish every outcome, including voids and aborts, with the same prominence. The failure
   thresholds above exist so that "we ran it and it failed" is a cheap, pre-authorised sentence
   rather than an argument.

---

## 6. What would change our mind (falsifiers, written down in advance)

| if this happens | then we must retract or weaken |
|---|---|
| E5 shows measured fidelity *above* our model at most points | our NISQ domination table is too strong; weaken it in `VALIDATION_BENCHMARK.md` |
| E3 shows the element error flat in k (P1 refuted) | "kinks change nothing on hardware" — and our exact-zero claim was closer to hardware than we thought, or the compiler cancelled the kinks (check depths first) |
| E2 gives S ≤ 2.00 | our LP settings are not realisable as prepared; the "0.003 off Tsirelson" sentence must be labelled ideal-only, permanently |
| E1 gives T2 < 5 µs | the whole block is void; the honest report is "the device could not support the measurement" |
| E6 gives ε ≤ 4.184 kJ/mol at 4 qubits | publish as a HIT at the top, and then be very careful: it says nothing about 54 orbitals |

---

## 7. Legal and originality boundary (must be settled before any job is submitted)

Connecting to a real QPU requires a **third-party vendor SDK and cloud account**. That collides
head-on with two standing constraints in this repo:

- `docs/artifacts/compliance/` + the governance audit (PR #149): the owned core imports no
  third-party quantum SDK. Correct and must stay correct.
- The `legacy/` boundary rule: zero third-party imports outside `legacy/`.

Therefore any hardware work must be **sandboxed, not integrated**: a separate directory outside
`src/` (e.g. `hardware/` or `legacy/qpu/`), never imported by the FBSC path, never a dependency of
the owned core, with its own README stating that it is *calibration of a third-party device*, not
part of the owned stack. The MIT licence covers our code; it does not cover vendor SDK terms, and
it does not cover "J.A.R.V.I.S." (separate, unresolved owner decision). **Flagged for the lead and
the legal manager** — this is a decision, not a task I can take unilaterally.

Two further honesty constraints for any published result:

- **Vendor naming.** Results should be reported as "device class *C*, vendor *V*, date *T*", because
  device parameters drift weekly and an unnamed device is an unreproducible claim.
- **Cost disclosure.** 300 s of QPU time is real money or a scarce free-tier allowance (order of
  half a typical monthly free allotment `[published/uncertain — verify at booking]`). Any published
  result should state what it cost, so a reader can judge whether it is worth re-running.

---

## 8. What we would do with more than 5 minutes (for the roadmap, not now)

| budget | what becomes possible | what still does not |
|---|---|---|
| 5 min (this doc) | measured channel, measured CHSH, measured depth decay, measured accuracy floor | any chemistry, any braid protection |
| 1 hour | + repeated blocks across days → drift statistics; + d=4 qudit tomography | still no FeMo-co |
| 1 day | + a real 12–20 qubit VQE on a genuine Fe–S model Hamiltonian, with mitigation | still 10⁴–10⁵× short of chemical accuracy at 54 orbitals |
| fault-tolerant, hours | the Reiher et al. estimate becomes meaningful | and our contribution there is the *classical* compression story, not a hardware claim |

---

## 9. One-paragraph summary for the owner

A real QPU cannot test our nitrogenase result, and no budget fixes that: the ladder is a surrogate
evaluated at five encoding dimensions, its smallest rung is 3.5× chemical accuracy, its d=4 state
is 2³² amplitudes with no known preparation circuit, and the real FeMo-co active space needs on the
order of 10¹¹ fault-tolerant gates. What 5 minutes *can* do is retire the two numbers we are most
exposed on — the coherence and protection figures that are ours, not a device's — and replace the
analytic noise curves in our benchmark with measured ones. Six pre-registered experiments, about
29 seconds of hardware, 1.52 million shots, every success and failure threshold written down before
submission. The honest ceiling stays where it is: **exact classical simulation, calibrated against
a measured device.** Nothing in this document moves that ceiling, and no result from it may be
reported as if it did.

---

## 10. Provenance and reproduce

| item | source | tag |
|---|---|---|
| 144.7362 / 129.9529 / 100.7657 / 76.0663 / 50.14 kJ/mol | `docs/artifacts/nitrogenase_qudit/nitrogenase_qudit_report.json` | `[measured]` |
| rung spacings 14.7833 / 29.1872 / 24.6994 / 25.9263; span 94.5962 | arithmetic on the above | `[analytic]` |
| coherence 0.876241, protection 0.995139 (d=4) | `coherence_of()`, `braid_pathways()` in `nitrogenase_qudit_simulator.py` — **our state measures, not device properties** | `[measured, of our own state]` |
| 4¹⁶ = 2³² amplitudes = 64 GiB dense statevector | `VALIDATION_BENCHMARK.md` §2.3 | `[analytic]` |
| device parameters (e₂, e₁, t_gate, T2, p_m) | `VALIDATION_BENCHMARK.md` §2.2 (Arute 2019; Kim 2023; Heron/H2-1 class) | `[published]` |
| F(D) analytic curves; depth-1 collapse at n = 54 | `VALIDATION_BENCHMARK.md` §3.1–3.2 | `[analytic]` |
| invariant distance exactly 0.000000 across 0→40 kinks | `docs/artifacts/anyon/ANYON_BRAID_EXPERIMENT.md` | `[measured, classical simulation]` |
| CHSH 2.8250 (n=2,3), 2.0351 (n=4), 0.003 off Tsirelson | `docs/artifacts/quantum_likeness/` (PR #146) | `[measured, classical optimisation]` |
| 54-orbital active space; 10¹¹ T-gate order of magnitude | Reiher, Wiebe, Svore, Wecker, Troyer, PNAS 114, 7555 (2017) | `[published]` |
| per-shot 5 µs reset+readout; 5 µs/shot ⇒ 3–6×10⁷ shots in 300 s | stated assumption — **superseded on IBM hardware; re-costed in `docs/qpu/IBM_FREE_TIER_RUNBOOK.md` §5** (350 µs per execution at defaults ⇒ 534 s for this plan, not ~30 s) | `[assumption]`/`[analytic]` |
| all value scores, and the §2 ranking | my judgement, weights stated | `[interpretation]` |
| every threshold in §3 | untested design | `[design]` |

> **⚠ Re-costing (2026-11-05) — read §5 of the runbook before quoting §1.** The `5 µs/shot` row above is an explicit `[assumption]`, and on IBM hardware it is wrong by ~70×: metering is **350 µs per execution at default settings** (250 µs `rep_delay` + ~100 µs circuit), so this plan's 1.52×10⁶ shots cost **534 s**, not ~30 s — 89 % of one 28-day Open Plan allowance rather than 10 % of a 300-second block. **`docs/qpu/IBM_FREE_TIER_RUNBOOK.md` §5** carries the correction (§5.1), the per-experiment re-cost table (§5.3: 506 s at defaults, 192 s tuned, and the `L = 200 µs` sensitivity that makes the tuned column conditional), and the resulting rules of engagement (§5.4: submit tuned, one block per window, 40 % reserve). The conclusion in §1 — "the 300-second budget is not the binding constraint" — is **true for a 300-second block and false for a 10-minute-per-month free tier**; on the free tier the scarce resource is *shots*. Nothing else in this document changes: the thresholds and designs in §3 are approved and are not re-opened here, and **no QPU job has been submitted** — every second in the runbook is `[analytic]`.

> **⚠ UPDATE 2026-11-05 — a QPU job HAS now been submitted, and the sentence below saying otherwise was true when written and is not true now.** Two blocks ran on IBM `ibm_fez` (qubits 44–45): a CHSH measurement and a recalibration control. They are recorded in **`docs/qpu/results/ibm_fez_run-001/`** at the tag **`OWNER-REPORTED`** — job IDs, job JSON, shot counts, the four CHSH settings and the calibration record were not supplied, so **nothing from that run is `[measured]` and none of it may be cited as such.** What it does change in *this* document: §3 E2's leg-1 threshold (S ≥ 2.60) was met **marginally** (2.61, i.e. 0.5σ) and leg 2 is **unevaluable** because E1 never ran, so E2 is **not** a pre-registered success; and E1's stated `coherence = 0.876241` / `protection = 0.995139` are still **our own state measures, un-retired**, because E1 did not run. The permanent ceiling in §0 is untouched. The autonomous 1.85 → 1.50 kHz drift the run exposed — and what it does and does not do to E1/E3/E5 — is analysed in that directory's `RESULTS.md` §2.

**Reproduce the `[measured]` numbers above** (no network, ~1 s):
```
PYTHONPATH=. python3 docs/artifacts/nitrogenase_qudit/nitrogenase_qudit_simulator.py
python3 docs/artifacts/anyon/anyon_braid_sim.py
PYTHONPATH=. python3 docs/validation/validate_benchmark.py
```
⚠ These three scripts **write** their report/JSON outputs (the simulator writes into
`docs/artifacts/nitrogenase_qudit/` *and* `/home/team/shared/`). Run them in a scratch copy of the
checkout unless you intend to regenerate the checked-in artifacts — running them in place will dirty
the working tree.
**Nothing in this file requires or uses a QPU to reproduce its numbers, and no job *described in §3* has been run as specified.** (Two blocks on IBM `ibm_fez` have since run outside this design — see the update note above and `docs/qpu/results/ibm_fez_run-001/`.)
