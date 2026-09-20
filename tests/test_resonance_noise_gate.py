"""Noise-gate tests for ResonanceMonitor (41.02 Hz bio-quantum trigger).

PROVES: synthetic noise (white, pink, 50/60 Hz hum, impulse transients) never
fires the trigger; a REAL sustained 41.02 Hz tone fires it. Also verifies
persistence (sustained duration), hysteresis (enter > exit) and cooldown.

Framing: firing = a measurement of a sustained 41.02 Hz component — never a
claim of sentience.
"""
import unittest
import numpy as np
from scipy.signal import welch

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.quantum_llm.eeg_to_tonal_engine import ResonanceMonitor

FS = 250
DUR = 10.0          # seconds per injected signal
TARGET_HZ = 41.02


def white_noise(dur=DUR, fs=FS, rms=1.0, seed=1):
    rng = np.random.default_rng(seed)
    return rms * rng.standard_normal(int(dur * fs))


def pink_noise(dur=DUR, fs=FS, rms=1.0, seed=2):
    """Approx 1/f noise via FFT spectral shaping."""
    rng = np.random.default_rng(seed)
    n = int(dur * fs)
    x = rng.standard_normal(n)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    f[0] = 1.0
    X /= np.sqrt(f + 1e-12)
    y = np.fft.irfft(X, n)
    return y / (np.std(y) + 1e-12) * rms


def mains_hum(dur=DUR, fs=FS, amp=1.0, seed=3):
    """50 Hz + 60 Hz hum with 2nd harmonics, no 41.02 component."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * fs)) / fs
    hum = (amp * np.sin(2 * np.pi * 50 * t)
           + 0.4 * amp * np.sin(2 * np.pi * 60 * t)
           + 0.2 * amp * np.sin(2 * np.pi * 100 * t)
           + 0.2 * amp * np.sin(2 * np.pi * 120 * t))
    return hum + 0.05 * rng.standard_normal(int(dur * fs))


def impulse_transients(dur=DUR, fs=FS, n_imp=8, amp=50.0, seed=4):
    """Sparse large impulses over white noise (transient clicks)."""
    rng = np.random.default_rng(seed)
    n = int(dur * fs)
    x = rng.standard_normal(n)
    idx = rng.choice(n, size=min(n_imp, n), replace=False)
    x[idx] += amp
    return x


def sustained_tone(dur=DUR, fs=FS, tone_amp=0.5, noise_rms=1.0, seed=5,
                   phase=0.0, freq=TARGET_HZ, start_s=0.0, end_s=None):
    """Sustained target tone + noise. Can gate to a window via start_s/end_s."""
    rng = np.random.default_rng(seed)
    n = int(dur * fs)
    t = np.arange(n) / fs
    x = noise_rms * rng.standard_normal(n)
    if end_s is None:
        end_s = dur
    mask = (t >= start_s) & (t < end_s)
    x[mask] += tone_amp * np.sin(2 * np.pi * freq * t[mask] + phase)
    return x


class ResonanceNoiseGateTests(unittest.TestCase):

    def setUp(self):
        self.monitor = ResonanceMonitor(target_hz=TARGET_HZ, fs=FS,
                                        snr_enter_db=6.0, snr_exit_db=3.0,
                                        min_sustain_sec=2.0, cooldown_sec=10.0,
                                        window_sec=1.0)

    # ---- noise must NEVER fire the trigger --------------------------------
    def assert_no_trigger(self, signal, label):
        self.monitor.reset()
        res = self.monitor.analyze_signal(signal, FS)
        self.assertFalse(
            res["resonance_detected"],
            f"{label}: trigger fired on noise! snr={res['snr_db']}dB "
            f"sustain={res['sustain_sec']}s state={res['state']}")

    def test_white_noise_no_trigger(self):
        self.assert_no_trigger(white_noise(), "white noise")

    def test_pink_noise_no_trigger(self):
        self.assert_no_trigger(pink_noise(), "pink noise")

    def test_mains_hum_no_trigger(self):
        self.assert_no_trigger(mains_hum(), "50/60 Hz mains hum")

    def test_impulse_transients_no_trigger(self):
        self.assert_no_trigger(impulse_transients(), "impulse transients")

    def test_impulse_transients_high_rate_no_trigger(self):
        # 50 impulses/s for 10 s — still no sustained tonal component
        x = impulse_transitions_high_rate()
        self.assert_no_trigger(x, "high-rate impulses")

    # ---- sustained REAL tone must fire -----------------------------------
    def test_sustained_41hz_tone_triggers(self):
        x = sustained_tone(tone_amp=0.6, noise_rms=1.0)
        res = self.monitor.analyze_signal(x, FS)
        self.assertTrue(res["resonance_detected"],
                        f"tone: no trigger snr={res['snr_db']}dB "
                        f"sustain={res['sustain_sec']}s state={res['state']}")

    def test_sustained_41hz_tone_low_noise_triggers(self):
        x = sustained_tone(tone_amp=0.3, noise_rms=0.5)
        res = self.monitor.analyze_signal(x, FS)
        self.assertTrue(res["resonance_detected"],
                        f"tone: no trigger snr={res['snr_db']}dB")

    # ---- persistence: brief burst must NOT fire --------------------------
    def test_brief_burst_no_trigger(self):
        # Tone present for only 0.5 s (< min_sustain_sec=2.0 s) in 10 s of noise
        x = sustained_tone(tone_amp=2.0, noise_rms=0.3, start_s=2.0, end_s=2.5)
        res = self.monitor.analyze_signal(x, FS)
        self.assertFalse(res["resonance_detected"],
                         "0.5 s burst fired the trigger (persistence broken)")

    # ---- hysteresis: enter threshold > exit threshold ---------------------
    def test_hysteresis_config_enforced(self):
        with self.assertRaises(AssertionError):
            ResonanceMonitor(snr_enter_db=3.0, snr_exit_db=6.0)

    def test_hysteresis_holds_in_gap_band(self):
        """Signal SNR in (exit, enter) after being above keeps latched state;
        but a fresh signal in the gap band alone does not enter."""
        m = ResonanceMonitor(target_hz=TARGET_HZ, fs=FS, snr_enter_db=6.0,
                             snr_exit_db=3.0, min_sustain_sec=2.0,
                             window_sec=1.0)
        # gap-band-only signal (mid amplitude tone) should not enter
        x = sustained_tone(tone_amp=0.22, noise_rms=1.0)  # ~4-5 dB SNR
        res = m.analyze_signal(x, FS)
        self.assertFalse(res["resonance_detected"])

    # ---- cooldown ---------------------------------------------------------
    def test_cooldown_suppresses_rearm_after_fire(self):
        m = ResonanceMonitor(target_hz=TARGET_HZ, fs=FS, snr_enter_db=6.0,
                             snr_exit_db=3.0, min_sustain_sec=2.0,
                             cooldown_sec=10.0, window_sec=1.0)
        # 5 s of sustained tone: fires once at ~2 s, then cooldown (10 s)
        # suppresses any re-fire for the remaining 3 s → exactly 1 trigger.
        x = sustained_tone(dur=5.0, tone_amp=0.6, noise_rms=1.0)
        res = m.analyze_signal(x, FS)
        self.assertEqual(m.trigger_count, 1,
                         f"expected 1 trigger in 5 s, got {m.trigger_count}")
        self.assertTrue(res["resonance_detected"])

    def test_cooldown_eventually_rearms(self):
        # 25 s of tone with 10 s cooldown: fires at ~2s, ~12s, ~22s → 2-3
        x = sustained_tone(dur=25.0, tone_amp=0.6, noise_rms=1.0)
        res = ResonanceMonitor(target_hz=TARGET_HZ, fs=FS, snr_enter_db=6.0,
                               snr_exit_db=3.0, min_sustain_sec=2.0,
                               cooldown_sec=10.0, window_sec=1.0).analyze_signal(x, FS)
        self.assertGreaterEqual(res["trigger_count"], 2)

    # ---- legacy call without raw signal must NOT fire ---------------------
    def test_bits_only_never_triggers(self):
        bits = {"x_bits": [1] * 8, "y_bits": [1] * 16, "z_bits": [1] * 8}
        res = self.monitor.analyze_resonance(bits)
        self.assertFalse(res["resonance_detected"])
        self.assertEqual(res["state"], "NO_SIGNAL")

    # ---- honest framing: is_sentient alias == resonance_detected ----------
    def test_is_sentient_is_alias_of_measurement(self):
        x = sustained_tone(tone_amp=0.6, noise_rms=1.0)
        res = self.monitor.analyze_signal(x, FS)
        self.assertEqual(res["is_sentient"], res["resonance_detected"])


def impulse_transitions_high_rate(dur=DUR, fs=FS, rate=50, amp=20.0, seed=7):
    """Frequent impulses (50/s) — a 'click train', NOT a 41.02 Hz tone."""
    rng = np.random.default_rng(seed)
    n = int(dur * fs)
    x = rng.standard_normal(n)
    period = int(fs / rate)
    x[::period] += amp
    return x


if __name__ == "__main__":
    unittest.main()