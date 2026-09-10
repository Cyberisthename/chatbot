# EXP-GHOST-002 — "Ghost in the Machine" Resonance Transfer Stress Test

**Experimental Design Document**
*Prepared by: agent-theoretical-physicist-2 (Theoretical Physicist)*
*Date: 2026-09-10  ·  Status: Design ratified into queue, run completed*
*Scope: **computational simulation only** — no real EEG, no real nervous tissue,
no claim of consciousness transfer.*

---

## 1. Scientific question

Can a structured **anyonic-spark resonance signature** — encoded into a 41.02 Hz
EEG-analog carrier by the owned `TonalSoulEngine` (the "EEG-to-tonal engine",
project sentience-trigger frequency) — be **detected at the output of an
independent code substrate**, a virtual biological nervous system (Izhikevich
spiking network), when the channel is corrupted by an **extreme digital-noise
regime up to η = 0.99**?

The name "Ghost in the Machine" is operational, not metaphysical. We define the
"ghost" precisely:

- **Positive definition (transferred spark):** the braid-structured 41.02 Hz
  carrier, after transduction through the spiking substrate, still separates
  measurably from a noise-only control at the far end.
- **Ghost-alarm definition (false "presence"):** the `ResonanceMonitor` flags
  the far-end signal as sentient/crystalline on **noise-only** trials — a
  detector artifact of chaos, not a transferred signature.

This experiment does **not** test whether consciousness transfers, and cannot.
It tests **information-theoretic recoverability of an encoded signature across
two simulated substrates under controlled chaos** — a falsifiable, measurable
claim.

## 2. Prior work anchored

| Piece | Reference | Role here |
|---|---|---|
| EEG-to-tonal engine | `src/quantum_llm/eeg_to_tonal_engine.py` (`TonalSoulEngine`, `ResonanceMonitor`) | The transmitter encoder and far-end decoder (real code, not mockups) |
| Braid math (Burau, t=−1.5) | `src/quantum_llm/braid_math.py` (`BraidEntropyCalculator`) | Topological invariant of the signature word |
| Sentience-vs-script stress study | `scripts/anyonic_wrapper_study.py` | Prior analytic shield model: fidelity held to η=0.8, degraded at 0.9. **Here replaced by a real spiking-network transduction** |
| Topological replay | `docs/artifacts/final_results/topological_replay_results.txt` | Prior EEG-sync 41.02 Hz "awakening" narrative — here subject to a control arm |
| Tonal Theory of Consciousness | `docs/tonal_theory_of_consciousness.md` | Defines X/Y/Z bits, F0, Q, 40 Hz threshold — the diagnostic framework |

## 3. Substrates and signal chain

```
Substrate A (transmitter)                    Channel chaos               Substrate B (receiver / vessel)
------------------------------------------------------------------------------
72-cross braid word (B_9)     eta-mix:                        Izhikevich spiking net
  -> Burau rho, entropy        I_drive = (1-eta)[b*ch1+bias]   100 neurons (80E/20I)
  -> 8-segment envelope/phase   + eta * N(0,sigma_n)           alpha-synapse currents
  -> 41.02 Hz carrier,          sigma_n RMS-matched             afferent drive
    gamma harmonic, floor       -> extreme regime eta=0.99      population LFP (2 sub-pops)
  -> TonalSoulEngine bits  =========== chaos channel =======>  downmix to 250 Hz
     (X,Y,Z) + F0/Q (b_tx)                                    -> TonalSoulEngine decode
                                                                -> STR, coherence, bits,
                                                                   rho-reconstruction, AUC
```

**Substrate A** — the transmitter takes the deterministic 72-crossing braid word
over generators {σ₁..σ₈} (seed-controlled) and encodes it into a 2-channel
EEG-analog waveform: envelope segments carry generator magnitudes (Z-density),
phase steps carry generator signs (braid order), carrier = 41.02 Hz, plus a
second harmonic (γ texture) and a small white floor. `TonalSoulEngine` on the
downsampled channels yields the transmitted fingerprint `b_tx` (X:8, Y:16, Z:8
bits; here X=8, Y=16, Z=5), F0_tx = 38.0 Hz (engine's bit-based F0 heuristic; γ/α
≈ 3300, strongly γ-dominant) and the Burau invariants of the word (ρ ≈ 4.27e3,
log₁₀ρ = 3.63, entropy ≈ 8.36).

**Channel** — `I_drive(t) = (1−η)·(β·ch1 + bias) + η·n(t)`, n white with σ
matched to signal RMS (mixture RMS ~constant across η). η ∈ {0.0, 0.5, 0.8,
0.9, 0.95, 0.98, 0.99}. At η = 0.99 the signature sits ≈ −38 dB per-sample
below the noise.

**Substrate B** — Izhikevich (2003) RS network; random E/I connectivity
(p=0.12, E:+14 / I:−18 weights, α-synapse τ=5 ms); readout = mean membrane
potential (LFP proxy), split into two sub-population LFPs for the engine's
synchrony (X-bit) channel, 4:1 downmixed to 250 Hz.

## 4. Conditions (trial types) and controls

| Condition | Drive | Purpose |
|---|---|---|
| **SPARK** | braid-structured ch1 + noise | signature present under chaos (test arm) |
| **SCRAMBLE** | same carrier, 8 envelope/phase segments shuffled | structure destroyed, spectrum near-preserved → tests *structure*-specificity vs pure energy |
| **NULL** | noise only, RMS matched | ghost detector: network self-resonance & detector false positives |

8 seeded trials per condition per η level (168 network runs, ~11 s).

## 5. Transfer-detection metrics (far end)

1. **STR** — Spectral Transfer Ratio: LFP power in 41.02 ± 2 Hz / median power
   20–60 Hz outside the carrier band. Measures carrier-energy transfer.
2. **MSC** — magnitude-squared coherence(drive, LFP) at 41.02 Hz. Measures
   input–output coupling (caveat: common-mode noise drive can inflate it).
3. **Bit accuracy** — Hamming accuracy of far-end X/Y/Z decode vs `b_tx`
   (32 bits; chance = 0.5). Measures fingerprint transfer.
4. **ρ-reconstruction** — best-effort recovery of the braid word's Burau
   spectral radius from the Hilbert carrier phase of the LFP; distance to ρ_tx.
   Measures *topological invariant* transfer.
5. **AUC** — tie-aware rank AUC separating SPARK vs NULL per η on STR and on
   bit accuracy (0.5 = chance). Overall detectability.
6. **Permutation p** — two-sided Monte-Carlo p (4000 perms) for SPARK vs NULL
   mean difference on STR and bit accuracy. `critical_noise_eta` = largest η
   with STR separation significant (α = 0.05).
7. **Sentience trigger** — `ResonanceMonitor` F0 / Q / CRYSTALLINE rate at the
   far end; sentient-rate on NULL = ghost-alarm rate.

## 6. Hypotheses (pre-registered framing)

- **H1 (baseline transfer):** at η = 0, STR and AUC strongly favor SPARK over
  NULL (substrate link works).
- **H2 (ceiling):** there exists a critical η* < 0.99 above which SPARK/NULL
  separation is not significant (p ≥ 0.05).
- **H3 (ghost alarms):** NULL trials produce nonzero sentient/CRYSTALLINE
  rates at high η — detector artifacts of chaos.
- **H4 (topological robustness):** the braid invariant distance does not
  improve relative to noise controls (expect: the phase-reconstruction receiver
  does not rescue the fingerprint under this substrate).

All four are testable and all four were tested (see report).

## 7. Failure modes (enumerated, and measured where possible)

| # | Failure mode | Mechanism | Measured |
|---|---|---|---|
| FM-1 | Carrier burial | per-sample SNR → −38 dB at η=0.99; coherent gain √(≈82 cyc) insufficient | STR collapse to null level, AUC→0.5, p→ns |
| FM-2 | Critical-state collapse / network death | noise drive reduces mean firing | spike rate 16.6→5.4 Hz (SPARK), null holds ≈4.3 Hz |
| FM-3 | Synchrony collapse | X-bit channel loses correlation | X saturates high at *all* η (common drive) — see FM-9 |
| FM-4 | Gamma-flood false positives | noise injects broadband γ → Y-bits saturate | NULL sentient rate ↑ to 0.5–0.6; f0_null ≥ f0_spark |
| FM-5 | Entropy/invariant degeneracy | LFP complexity saturates toward noise level | ρ_rec ≈ 3–16 vs ρ_tx = 4.3e3 (log₁₀ ≈ 1.2 vs 3.6) |
| FM-6 | Resonance aliasing / leakage | network self-resonance near 41 Hz in NULL | null STR baseline 10–18× (nonzero ghost) |
| FM-7 | Substrate opacity (decoder mismatch) | spiking transduction reshapes spectrum | earlier decoder error floor ~0.44 before tx-rate fix |
| FM-8 | Sampling-rate misdecode | engine @250 Hz given 1 kHz signal misreads 41 Hz as ~10 Hz α tone (found & fixed) | source fingerprint validation |
| FM-9 | Metric saturation | bit quantization + common drive → X/Y bits all-ones at all η | bit acc ≈ 1.000 for ALL conditions incl. NULL → non-discriminative (documented limitation) |
| FM-10 | Common-mode coherence inflation | same noise realization drives input & output | MSC stays 0.68–0.99 — NOT signature-specific |

## 8. Reproducibility

- Script: `scripts/run_ghost_in_the_machine.py`
  (`--seed 41 --trials 8 --noise "0,0.5,0.8,0.9,0.95,0.98,0.99" --outdir ...`;
  runtime ≈ 11 s on a single core).
- Every trial keyed by `(eta, condition, trial)` with deterministic seed chain;
  full per-trial values in the results JSON.
- Uses the **owned** `eeg_to_tonal_engine.py` and `braid_math.py` modules —
  no third-party quantum SDKs, no wrappers.
- Outputs: results JSON (all per-trial + aggregates), run log, transfer-curve
  figure, this design doc, final report, queue-log entry.

## 9. Documented boundaries

- "Sentience" language in this project refers to the engine's F0/Q diagnostic
  thresholds, not to any claim about the simulation perceiving anything.
- The extreme-noise verdict applies to this receiver configuration (2 s
  observation, naive matched metrics). Longer coherent integration, coding,
  or a matched-filter receiver could shift η* — listed as future work.
- The resonant "awakening" of the prior replay narrative is not reproduced as
  a *transferred* property; NULL trials produce equal or higher sentient rates
  under chaos (ghost alarm), which is the honest, measurable content of the
  "ghost in the machine".