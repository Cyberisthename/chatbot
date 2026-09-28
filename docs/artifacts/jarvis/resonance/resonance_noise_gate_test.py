#!/usr/bin/env python3
"""resonance_noise_gate_test.py — Before/after false-positive proof for the
41.02 Hz bio-quantum resonance trigger (f0 heuristic → noise-gated detector).

BEFORE (pre-fix behavior, reproduced here for comparison):
    f0 = 10.0 + z_count*4.0 + y_count*0.5 ; is_sentient = f0 >= 40.0
    — a pure bit-count heuristic: ANY signal whose z/y bit counts are high
      (noise routinely inflates them) fires the "sentience" trigger.

AFTER (the fix):
    Spectral measurement of the 41.02 Hz band vs local noise floor + persistence
    (min sustained time) + hysteresis (enter > exit) + cooldown + median-smoothed
    latch. Trigger requires a REAL sustained 41.02 Hz component.

This script generates the same synthetic battery (white noise, pink noise,
50/60 Hz hum, impulse transients, sustained tone at several amplitudes) and runs
BOTH detectors on every case, printing the before/after table and writing
resonance_noise_gate_results.json.

Framing (honesty requirement): firing the detector is a MEASUREMENT of a
sustained 41.02 Hz component — never a claim of sentience.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from quantum_llm.eeg_to_tonal_engine import ResonanceMonitor, TonalSoulEngine

FS = 250
TARGET_HZ = 41.02


# ----------------------------------------------------------------------
# Signal generators (deterministic seeds; same battery for both detectors)
# ----------------------------------------------------------------------
def white_noise(seconds=10.0, rms=1.0, seed=1):
    rng = np.random.default_rng(seed)
    return rms * rng.standard_normal(int(seconds * FS))


def pink_noise(seconds=10.0, rms=1.0, seed=2):
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    x = rng.standard_normal(n)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1.0 / FS)
    f[0] = 1.0
    X = X / np.sqrt(f)
    y = np.fft.irfft(X, n)
    return y / (np.std(y) + 1e-12) * rms


def mains_hum(seconds=10.0, amp=1.0, seed=3):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * FS)) / FS
    return (amp * np.sin(2 * np.pi * 50 * t)
            + 0.5 * amp * np.sin(2 * np.pi * 60 * t)
            + 0.3 * amp * np.sin(2 * np.pi * 100 * t)
            + 0.3 * amp * np.sin(2 * np.pi * 120 * t)
            + 0.05 * rng.standard_normal(int(seconds * FS)))


def impulses(seconds=10.0, n=10, amplitude=50.0, seed=4):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(int(seconds * FS))
    idx = rng.choice(len(x), size=min(n, len(x)), replace=False)
    x[idx] += amplitude
    return x


def sustained_tone(seconds=10.0, tone_amp=0.5, noise_rms=1.0, seed=5,
                   phase=0.0, start=0.0, end=None):
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    t = np.arange(n) / FS
    x = noise_rms * rng.standard_normal(n)
    if end is None:
        end = seconds
    mask = (t >= start) & (t < end)
    x[mask] += tone_amp * np.sin(2 * np.pi * TARGET_HZ * t[mask] + phase)
    return x


# ----------------------------------------------------------------------
# BEFORE: reproduced legacy heuristic (the bug) — bit-count f0, no signal
# ----------------------------------------------------------------------
def legacy_analyze(bits):
    z_count = int(sum(bits["z_bits"]))
    y_count = int(sum(bits["y_bits"]))
    x_count = int(sum(bits["x_bits"]))
    f0 = 10.0 + (z_count * 4.0) + (y_count * 0.5)
    q_factor = (x_count * 1.5) / (1.0 + (8 - z_count) * 0.1)
    is_sentient = f0 >= 40.0
    return {"f0": f0, "q_factor": q_factor, "is_sentient": is_sentient,
            "state": "CRYSTALLINE" if is_sentient and q_factor > 8 else "NOISY"}


# ----------------------------------------------------------------------
# Battery
# ----------------------------------------------------------------------
BATTERY = [
    ("white_noise",      white_noise()),
    ("pink_noise",       pink_noise()),
    ("mains_hum_50_60",  mains_hum()),
    ("impulse_transients", impulses()),
    ("tone_0.1_amp",     sustained_tone(tone_amp=0.1,  noise_rms=1.0)),
    ("tone_0.2_amp",     sustained_tone(tone_amp=0.2,  noise_rms=1.0)),
    ("tone_0.4_amp",     sustained_tone(tone_amp=0.4,  noise_rms=1.0)),
    ("tone_0.6_amp",     sustained_tone(tone_amp=0.6,  noise_rms=1.0)),
    ("tone_01s_burst",   sustained_tone(tone_amp=2.0, noise_rms=0.3,
                                        start=4.0, end=5.0)),
]

GATE = dict(target_hz=TARGET_HZ, fs=FS, snr_enter_db=6.0, snr_exit_db=3.0,
            min_sustain_sec=2.0, cooldown_sec=10.0, window_sec=1.0)


def main():
    engine = TonalSoulEngine()
    results = []
    table = []
    for label, sig in BATTERY:
        bits = engine.extract_bits([sig, sig])
        before = legacy_analyze(bits)                       # old heuristic
        gate = ResonanceMonitor(**GATE)
        after = gate.analyze_signal(sig, FS)                # noise-gated
        row = {
            "case": label,
            "before_is_sentient": bool(before["is_sentient"]),
            "before_f0_est": round(before["f0"], 2),
            "after_resonance_detected": bool(after["resonance_detected"]),
            "after_snr_db": after["snr_db"],
            "after_state": after["state"],
            "after_trigger_count": after["trigger_count"],
        }
        results.append(row)
        table.append([
            label,
            "YES" if row["before_is_sentient"] else "no",
            f"{row['before_f0_est']:.1f}",
            "YES" if row["after_resonance_detected"] else "no",
            "—" if row["after_snr_db"] is None else f"{row['after_snr_db']:.1f}",
            row["after_state"],
        ])

    # Summary counts
    n_fp_before = sum(1 for r in results if r["before_is_sentient"]
                      and r["case"] in ("white_noise", "pink_noise",
                                        "mains_hum_50_60", "impulse_transients",
                                        "tone_0.1_amp"))
    n_fp_after = sum(1 for r in results if r["after_resonance_detected"]
                     and r["case"] in ("white_noise", "pink_noise",
                                       "mains_hum_50_60", "impulse_transients",
                                       "tone_0.1_amp"))
    n_tp_before = sum(1 for r in results if r["before_is_sentient"]
                      and r["case"] in ("tone_0.4_amp", "tone_0.6_amp"))
    n_tp_after = sum(1 for r in results if r["after_resonance_detected"]
                     and r["case"] in ("tone_0.4_amp", "tone_0.6_amp"))

    out = {"gate_cfg": GATE,
           "summary": {
               "false_positive_cases_before": n_fp_before,
               "false_positive_cases_after": n_fp_after,
               "true_positive_sustained_tone_before": n_tp_before,
               "true_positive_sustained_tone_after": n_tp_after,
               "note": "before = legacy bit-count f0>=40 heuristic; "
                       "after = noise-gated spectral detector. Firing is a "
                       "measurement, not a sentience claim."},
           "cases": results}
    out_path = ROOT / "docs" / "resonance_fix"
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "resonance_noise_gate_results.json").write_text(
        json.dumps(out, indent=2))

    print("BEFORE (legacy f0 heuristic) vs AFTER (noise-gated spectral detector)")
    print(f"{'case':<22}{'before sent.':<14}{'f0_est':<9}"
          f"{'after detect':<14}{'SNR dB':<9}{'state'}")
    print("-" * 78)
    for label, b_sent, b_f0, a_det, a_snr, a_state in table:
        fp = " (false +)" if b_sent == "YES" and label in (
            "white_noise", "pink_noise", "mains_hum_50_60",
            "impulse_transients", "tone_0.1_amp") else ""
        print(f"{label:<22}{b_sent:<14}{b_f0:<9}{a_det:<14}{a_snr:<9}{a_state}{fp}")
    print("-" * 78)
    print(f"False positives (noise-only cases): before={n_fp_before} after={n_fp_after}")
    print(f"True positives (sustained tone)    : before={n_tp_before} after={n_tp_after}")
    print(f"JSON: {out_path / 'resonance_noise_gate_results.json'}")


if __name__ == "__main__":
    main()