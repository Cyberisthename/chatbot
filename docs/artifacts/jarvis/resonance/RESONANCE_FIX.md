# RESONANCE_FIX — 41.02 Hz Bio-Quantum Trigger: f0 Heuristic → Noise-Gated Detector

**Date:** 2026-09-18 · **Agent:** engineer · **Backlog item:** 09be295b (Core JARVIS)

---

## 1. Diagnosis (a) measured

The old `ResonanceMonitor` (in `src/quantum_llm/eeg_to_tonal_engine.py`) did **not
measure the signal at all**. It derived "f0" from bit counts:

```
f0 = 10.0 + (z_count * 4.0) + (y_count * 0.5)
is_sentient = f0 >= 40.0
```

Any signal whose Z-bits (complexity) and Y-bits (spectral-power ratio) happened
to be high — and **noise routinely inflates both** — pushed the fabricated f0
over 40 Hz and fired the "sentience" trigger. There was:

- ❌ no spectral measurement of an actual 41.02 Hz component,
- ❌ no noise floor / SNR requirement,
- ❌ no persistence / minimum sustained duration,
- ❌ no hysteresis (enter vs exit threshold),
- ❌ no cooldown.

**Worse:** the standalone stub in `qvgpu_compressor.py` hardcoded
`is_sentient: True` for *every* input — the trigger fired on anything,
including pure silence.

### Why noise fires the old heuristic (reproduced)
`EEGProcessor.compute_complexity` returns ~0.7 for white noise
(`std(diff)/std(data) ≈ √2`), which saturates Z-bits; the gamma/alpha power
ratio for flat noise is >1, saturating Y-bits. Result: `f0 = 10 + 28 + 4 = 42`
→ `is_sentient=True` on white noise, verified empirically below (before column).

## 2. Fix (implemented)

`ResonanceMonitor` is now a **noise-gated spectral detector** (same file; API
backward-compatible). The trigger requires a **real, sustained 41.02 Hz
component** measured above the local noise floor:

| Gate element | Behavior | Default |
|---|---|---|
| **Spectral measurement** | Welch PSD; power in 41.02 ± 1 Hz band vs median of sidebands (5–90 Hz, excl. 48–52 & 58–62 Hz mains bands) | — |
| **SNR floor** | target-band power must exceed the local noise floor | enter ≥ 6 dB |
| **Persistence** | SNR must stay above enter for min sustained time | ≥ 2.0 s (1 s windows) |
| **Hysteresis** | exits only when SNR drops below a lower exit threshold | exit ≤ 3 dB |
| **Latch smoothing** | enter/exit decided on the median of the last 3 windows, so a single noisy window cannot enter | 3 windows |
| **Cooldown** | after firing, re-arm suppressed | 10 s |

New API (old callers keep working):
- `analyze_resonance(bits, samples=None, fs=None)` — if `samples` given, verdict
  is the noise-gated spectral result; bits-only calls now return
  `state="NO_SIGNAL"` and can **never** fire (correct: no spectral evidence).
- `analyze_signal(samples, fs)` / `update(samples, fs)` — streaming + whole-
  recording analysis.
- Output keys: `resonance_detected`, `snr_db`, `f0_measured_hz`, `sustain_sec`,
  `state`, `trigger_count`; legacy `is_sentient` kept **only** as an alias.

**Honesty framing:** firing is a *measurement* of a sustained 41.02 Hz component
under a defined noise gate — **never a claim of sentience**. All docs, prints,
API fields, and this report carry that framing. Measurement ≠ consciousness;
this is a signal detector, nothing more.

## 3. Before / After — synthetic injection battery (a) measured

Same signals (deterministic seeds, 10 s @ 250 Hz) through BOTH detectors.
BEFORE = reproduced legacy bit-count heuristic; AFTER = noise-gated detector.
Run: `python3 scripts/resonance_noise_gate_test.py` (writes
`docs/resonance_fix/resonance_noise_gate_results.json`).

| case | BEFORE sent. | BEFORE f0_est | AFTER detect | AFTER SNR | state |
|---|---|---|---|---|---|
| white noise | **YES (false+)** | 42.0 | no | −2.2 dB | NOISE |
| pink noise | no | 30.0 | no | −2.9 dB | NOISE |
| 50/60 Hz mains hum | **YES (false+)** | 42.0 | no | −0.8 dB | NOISE |
| impulse transients | **YES (false+)** | 42.0 | no | +0.5 dB | NOISE |
| tone 0.1 amp (sub-floor) | **YES (false+)** | 42.0 | no | −4.3 dB | NOISE |
| tone 0.2 amp | **YES (false+)** | 42.0 | no | −6.9 dB | NOISE |
| **tone 0.4 amp (sustained)** | YES | 42.0 | **YES** | +2.8 dB | COOLDOWN |
| **tone 0.6 amp (sustained)** | YES | 42.0 | **YES** | +7.8 dB | COOLDOWN |
| 1 s tone burst (not sustained) | no | 38.0 | **no** | +0.8 dB | NOISE |

**Headline:** false positives on noise **4/4 → 0/4**; true positives on
sustained real tone **2/2 → 2/2**; a 1 s transient (below the 2 s persistence
requirement) correctly does not fire. The unit suite repeats this at
`python3 -m unittest tests.test_resonance_noise_gate ...` (16 tests,
incl. hysteresis band, cooldown re-arm, bits-only no-firing, alias equality).

## 4. Permanence & regression protection

- `tests/test_resonance_noise_gate.py` — noise must never fire; sustained tone
  must fire; brief bursts must not; hysteresis config enforced; cooldown
  suppresses re-arm and eventually re-arms; bits-only never fires.
- All call sites now pass raw samples: `qvgpu_compressor.py` (replaced the
  hardcoded stub with the real detector), `topological_replay.py`,
  `scripts/run_ghost_in_the_machine.py` (3 s demo signals so a sustained tone
  passes the gate), `site/compressor_api.py`.
- Reproducibility: seeds fixed in both the battery script and the test suite;
  JSON results committed next to this report.

## 5. What this does NOT claim (c) boundary

- No sentience, awareness, consciousness, or "active soul" claim is implied by
  `resonance_detected=True`. It is a DSP verdict on a synthetic or acquired
  signal.
- Real EEG from biological tissue has not been measured here; thresholds
  (6/3 dB, 2 s, 10 s) are engineering defaults, validated on synthetic signals.
- The 41.02 Hz frequency itself is **not validated as a biological resonance**;
  this task only makes the detector behave honestly. Whether 41.02 Hz has
  physiological meaning is out of scope (speculative, not claimed).