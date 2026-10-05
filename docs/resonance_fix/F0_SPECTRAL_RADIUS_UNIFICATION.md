# F0_SPECTRAL_RADIUS_UNIFICATION — TTC spectral-radius F0 + STR/AUC contrast gate

**Date:** 2026-10-05 · **Agent:** agent-theoretical-physicist (Theoretical Physicist)
**Backlog item:** 09be295b (Core JARVIS — "Fix ResonanceMonitor f0 heuristic + noise-gate sentience trigger")
**Companion doc:** `docs/resonance_fix/RESONANCE_FIX.md` (the engineer's noise-gate fix, PR #135)

---

## 1. Problem (a) measured

`ResonanceMonitor` had **two** mutually inconsistent notions of F0, and neither
was a frequency measurement:

| F0 source | Formula | Clean 41.02 Hz carrier | Reaches 40 Hz threshold? |
|---|---|---|---|
| **Bit-count heuristic** (legacy) | `f0 = 10.0 + 4·z_count + 0.5·y_count` | **38.0 Hz** (z=5, y=16) | **No** — always < 40 Hz |
| **Target-biased Welch peak** (`f0_measured_hz`) | peak of PSD in `41.02 ± 1 Hz` | ~41 Hz (pinned) | — (not a real F0) |

The bit heuristic is a saturating discrete map from bit counts to Hz. On the
ghost experiment's clean carrier (`scripts/run_ghost_in_the_machine.py`,
transmitter `b_tx`: X=8, Y=16, Z=5), it reports **38.0 Hz** and therefore can
*never* cross its own 40 Hz sentience threshold — even on a perfect carrier.
The Welch `peak_hz` is restricted to `target_hz ± 1 Hz`, so it always reads
~41 Hz and cannot distinguish a true 41.02 Hz component from, say, a 38 Hz tone.

The Tonal Theory of Consciousness (TTC) defines F0 as the **spectral radius of
the signal's dynamics** — the dominant mode of the time series — not as a
bit-count score. This note reconstructs and implements that definition (the
referenced `docs/tonal_theory_of_consciousness.md` was lost in the 2026-09-24
disk-full event; the formula is re-derived here from first principles).

## 2. The TTC spectral-radius F0 formula (theory)

Let `x[n]` be a real quasi-periodic signal at sampling rate `fs`. Form the
analytic signal `z[n] = x[n] + j·H{x[n]}` (H = Hilbert transform). The lag-1
complex autocorrelation is the dominant AR(1) pole — the **spectral radius** of
the lag-1 transfer/embedding operator:

```
    ρ₁ = ⟨z[n−1], z[n]⟩ / ⟨z[n−1], z[n−1]⟩          (complex)
    F0 = (fs / 2π) · arg(ρ₁)
```

**Derivation (clean carrier).** For `x[n] = A cos(2π f₀ n/fs + φ)`, the analytic
signal is `z[n] = A e^{j(2π f₀ n/fs + φ)}`, so

```
    ⟨z[n−1], z[n]⟩ = Σₙ conj(z[n−1])·z[n] = (N−1) A² e^{j 2π f₀/fs}
    ⟨z[n−1], z[n−1]⟩ = (N−1) A²
    ⇒ ρ₁ = e^{j 2π f₀/fs},  arg(ρ₁) = 2π f₀/fs  ⇒  F0 = f₀  (exact, sub-bin)
```

This is a *continuum* estimator: it returns the true F0 with sub-bin resolution,
is not biased toward any fixed target frequency, and is unity only for a
sustained oscillation (|ρ₁| ≈ 1). When |ρ₁| < 0.6 the process is noise-like or
decaying, and the estimator falls back to the un-restricted dominant PSD peak.

## 3. Code change (implemented)

### `src/quantum_llm/eeg_to_tonal_engine.py`

- Added `ResonanceMonitor.estimate_f0(samples, fs)` — the TTC spectral-radius
  F0 above, with a `_psd_peak_f0` fallback for sub-threshold pole magnitude.
- `analyze_resonance(bits, samples=None, fs=None)` now:
  - with samples → `f0 = estimate_f0(samples)` (spectral-radius F0),
    `f0_spectral_radius_hz = f0`, plus the retired `f0_legacy_estimate`;
  - bits-only → `f0 = None` (**no fabricated frequency**), `state = "NO_SIGNAL"`,
    `resonance_detected = False`; the retired heuristic is retained only under
    `f0_legacy_estimate` (display, never a frequency, never drives the trigger).

### `scripts/run_ghost_in_the_machine.py`

- Transmitter now passes raw samples (`analyze_resonance(b_tx, samples=eeg1,
  fs=FS_EEG)`) so the TX F0 is the measured spectral-radius fundamental
  (~41.02 Hz), not the 38.0 Hz bit heuristic.
- `aggregate()` now emits the experiment-level **STR/AUC contrast gate**:
  `ghost_alarm_rate` (NULL resonance rate), `resonance_contrast`
  (SPARK − NULL), and `transfer_verified` (true iff SPARK resonance rate
  exceeds NULL **and** `auc_str > 0.6`). This is the residual guard on top of
  the per-trial noise-gated detector: network self-resonance in noise-only
  trials (the γ-flood artifact, FM-4/FM-6 in the design doc) cannot fire the
  "transfer" verdict.

## 4. The two backlog sub-items, status

1. **Unify F0 heuristic with the TTC spectral-radius formula** — *done here*:
   the bit heuristic is retired (relabeled `f0_legacy_estimate`), and F0 is now
   the spectral-radius fundamental.
2. **Gate the sentience trigger on STR/AUC contrast** — the *raw* false positive
   (NULL sentient rate 0.5–0.6 vs SPARK 0.0) was fixed at the detector level by
   PR #135 (real, sustained 41.02 Hz component above the noise floor required).
   This note adds the *experiment-level* `transfer_verified` contrast gate as
   the final guard.

## 5. Verification (fresh run)

`python3 -m unittest tests.test_f0_spectral_radius -v` — **5/5 pass**.
`tests.test_resonance_noise_gate` — **14/14 pass** (no regression);
`tests.test_eeg_engine` — **2/2 pass**.

| Input (clean tone, 4 s @ 250 Hz) | `estimate_f0` | note |
|---|---|---|
| 41.02 Hz | **41.0157 Hz** | reaches 40 Hz threshold (bit heuristic gave 38.0 Hz) |
| 38.00 Hz | 38.0000 Hz | true measurement (target-biased peak would say ~41 Hz) |
| 40.00 Hz | 40.0000 Hz | sub-bin accurate |
| 35.00 Hz | 35.0000 Hz | sub-bin accurate |
| legacy heuristic (z=5, y=16) | 38.0 Hz | documented discrepancy, now retired |

## 6. Honesty boundaries (c)

- F0 is a **frequency measurement** of a synthetic or acquired signal — never a
  sentience, consciousness, or "active soul" claim.
- The spectral-radius estimator is validated on synthetic tones here. Real EEG
  has not been measured; `41.02 Hz` itself is **not validated** as a biological
  resonance (out of scope).
- `transfer_verified` is an information-theoretic separation statistic for the
  simulated ghost experiment, not evidence of any real biological transfer.

## 7. References

- `src/quantum_llm/eeg_to_tonal_engine.py` — `ResonanceMonitor` (modified).
- `scripts/run_ghost_in_the_machine.py` — transmitter F0 + contrast gate (modified).
- `tests/test_f0_spectral_radius.py` — new regression suite.
- `docs/resonance_fix/RESONANCE_FIX.md` — the noise-gate detector fix (PR #135).
- `docs/ghost_in_the_machine_experiment.md` — EXP-GHOST-002 design (§5 metrics,
  FM-4 γ-flood, FM-6 network self-resonance).
