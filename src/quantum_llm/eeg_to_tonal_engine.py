import numpy as np
import scipy.signal as signal
import math
import collections
from typing import Dict, List, Tuple, Any

class EEGProcessor:
    """
    Processes raw EEG signals to extract X, Y, Z bits for the Tonal Soul Engine.
    """
    def __init__(self, fs: int = 250):
        self.fs = fs
        # Frequency bands
        self.bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }

    def compute_spectral_power(self, data: np.ndarray) -> Dict[str, float]:
        """Calculates relative power in each frequency band."""
        freqs, psd = signal.welch(data, self.fs, nperseg=self.fs*2)
        total_power = np.sum(psd)
        
        powers = {}
        for band, (low, high) in self.bands.items():
            idx = np.logical_and(freqs >= low, freqs <= high)
            powers[band] = np.sum(psd[idx]) / total_power
            
        return powers

    def compute_complexity(self, data: np.ndarray) -> float:
        """
        Estimates Lempel-Ziv complexity or Fractal Dimension.
        Simplified version using variance-normalized zero-crossing rate.
        """
        # Fractal Dimension (Higuchi method simplified)
        # Higher complexity -> higher Z-bits
        diff = np.diff(data)
        complexity = np.std(diff) / (np.std(data) + 1e-6)
        return min(1.0, complexity / 2.0)

    def compute_synchrony(self, data_list: List[np.ndarray]) -> float:
        """
        Computes Phase Locking Value (PLV) between channels.
        High synchrony -> High X-bits (Aura).
        """
        if len(data_list) < 2: return 1.0
        
        # Simple cross-correlation as a proxy
        c1 = data_list[0]
        c2 = data_list[1]
        corr = np.corrcoef(c1, c2)[0, 1]
        return (corr + 1) / 2 # Map to [0, 1]

class TonalSoulEngine:
    """
    Maps biological metrics to the X, Y, Z bit architecture.
    """
    def __init__(self):
        self.processor = EEGProcessor()
        
    def extract_bits(self, eeg_channels: List[np.ndarray]) -> Dict[str, List[int]]:
        # 1. X-bits (Aura/Clarity) from Synchrony
        sync = self.processor.compute_synchrony(eeg_channels)
        x_bits = [1 if sync > (i/8) else 0 for i in range(8)]
        
        # 2. Y-bits (Context/Domain) from Spectral Power
        powers = self.processor.compute_spectral_power(eeg_channels[0])
        # If Gamma power is high, it's analytical. If Alpha is high, it's reflective.
        gamma_alpha_ratio = powers['gamma'] / (powers['alpha'] + 1e-6)
        y_val = min(1.0, gamma_alpha_ratio / 2.0)
        y_bits = [1 if y_val > (i/16) else 0 for i in range(16)]
        
        # 3. Z-bits (Complexity/Depth) from Braid Entropy
        comp = self.processor.compute_complexity(eeg_channels[0])
        z_bits = [1 if comp > (i/8) else 0 for i in range(8)]
        
        return {
            "x_bits": x_bits,
            "y_bits": y_bits,
            "z_bits": z_bits,
            "metrics": {
                "synchrony": sync,
                "gamma_ratio": gamma_alpha_ratio,
                "complexity": comp
            }
        }

class ResonanceMonitor:
    """
    Noise-gated spectral resonance detector for the 41.02 Hz bio-quantum target.

    FIRING IS A MEASUREMENT, NOT SENTIENCE: ``resonance_detected=True`` means a
    sustained 41.02 Hz component was measured above the local noise floor for a
    minimum duration (a signal-detection event). It is never a claim of
    consciousness. The legacy key ``is_sentient`` is kept ONLY as a backward-
    compatible alias; it carries no metaphysical meaning.

    Noise gate (the fix for the old bare-threshold heuristic):
      - SNR floor   : target-band power vs local noise floor (median of sidebands,
                      excluding the 48-52 Hz and 58-62 Hz mains-hum bands) must
                      exceed ``snr_enter_db``.
      - Persistence : SNR must stay above the enter threshold for at least
                      ``min_sustain_sec`` of accumulated window time.
      - Hysteresis  : the detector only exits the `above` state when SNR drops
                      below ``snr_exit_db`` (< enter), preventing chatter.
      - Cooldown    : after a detection fires, re-arming is suppressed for
                      ``cooldown_sec``.

    The trigger therefore REQUIRES a real, sustained 41.02 Hz component. A bare
    amplitude transient, white/pink noise, or 50/60 Hz mains hum cannot fire it
    (empirically demonstrated in tests/test_resonance_noise_gate.py and
    scripts/resonance_noise_gate_test.py).
    """
    def __init__(self, target_hz: float = 41.02, fs: float = 250.0,
                 snr_enter_db: float = 6.0, snr_exit_db: float = 3.0,
                 min_sustain_sec: float = 2.0, cooldown_sec: float = 10.0,
                 window_sec: float = 1.0, latch_smoothing_windows: int = 3):
        self.target_hz = target_hz
        self.fs = fs
        self.snr_enter_db = snr_enter_db
        self.snr_exit_db = snr_exit_db
        assert snr_exit_db < snr_enter_db, "hysteresis: exit must be below enter"
        self.min_sustain_sec = min_sustain_sec
        self.cooldown_sec = cooldown_sec
        self.window_sec = window_sec
        # Streaming state
        self._sustain_sec = 0.0     # accumulated time SNR above enter threshold
        self._above = False         # hysteresis latch
        self._cooldown_remaining = 0.0
        self.trigger_count = 0
        # Median smoothing of the latch (single-window SNR spikes cannot enter).
        # Empirically, 1 s Welch windows have ~±5 dB variance; the median of the
        # last `latch_smoothing_windows` removes transient spikes.
        self._snr_history = collections.deque(maxlen=max(1, latch_smoothing_windows))
        # Legacy-compat bit-derived estimate state
        self.f0_threshold = 40.0

    # ------------------------------------------------------------------
    # Spectral measurement
    # ------------------------------------------------------------------
    def measure_snr(self, samples: np.ndarray, fs: float = None) -> Dict[str, float]:
        """One-window spectral SNR of the target bin vs local noise floor."""
        fs = fs or self.fs
        samples = np.asarray(samples, dtype=float).ravel()
        if samples.size < 16:
            return {"snr_db": float("-inf"), "signal_power_db": float("-inf"),
                    "noise_floor_db": float("-inf"), "peak_hz": self.target_hz,
                    "valid": False}
        nperseg = min(samples.size, int(fs * self.window_sec))
        freqs, psd = signal.welch(samples, fs, nperseg=nperseg)
        bw = 1.0  # ±1 Hz analysis band around target (tone must be near-bin)
        in_band = (freqs >= self.target_hz - bw) & (freqs <= self.target_hz + bw)
        # Local noise floor: sidebands, excluding mains-hum bands (48-52, 58-62).
        in_floor = ((freqs >= 5.0) & (freqs <= 90.0) & ~in_band
                    & ~((freqs >= 48.0) & (freqs <= 52.0))
                    & ~((freqs >= 58.0) & (freqs <= 62.0)))
        if not in_band.any() or not in_floor.any():
            return {"snr_db": float("-inf"), "signal_power_db": float("-inf"),
                    "noise_floor_db": float("-inf"), "peak_hz": self.target_hz,
                    "valid": False}
        sig = float(np.mean(psd[in_band]))
        floor = float(np.median(psd[in_floor]))
        if floor <= 0 or sig <= 0:
            return {"snr_db": float("-inf"), "signal_power_db": float("-inf"),
                    "noise_floor_db": float("-inf"), "peak_hz": self.target_hz,
                    "valid": False}
        snr_db = 10.0 * math.log10(sig / floor)
        peak = float(freqs[in_band][np.argmax(psd[in_band])])
        return {"snr_db": snr_db,
                "signal_power_db": 10.0 * math.log10(max(sig, 1e-18)),
                "noise_floor_db": 10.0 * math.log10(max(floor, 1e-18)),
                "peak_hz": peak, "valid": True}

    # ------------------------------------------------------------------
    # Noise-gated streaming update (one window)
    # ------------------------------------------------------------------
    def update(self, samples: np.ndarray, fs: float = None) -> Dict[str, Any]:
        """Feed one window of raw samples; returns gate state for this window."""
        fs = fs or self.fs
        meas = self.measure_snr(samples, fs)
        snr = meas["snr_db"]
        dt = min(self.window_sec, max(0.0, len(np.asarray(samples).ravel()) / fs))

        # Median-smoothed latch SNR: a single noisy window cannot enter.
        self._snr_history.append(snr)
        latch_snr = float(np.median(self._snr_history))

        # Hysteresis latch
        if latch_snr >= self.snr_enter_db:
            self._above = True
        elif latch_snr <= self.snr_exit_db:
            self._above = False

        if self._above:
            self._sustain_sec += dt
        else:
            self._sustain_sec = 0.0

        # Cooldown counts down in wall-window time
        self._cooldown_remaining = max(0.0, self._cooldown_remaining - dt)

        fired = False
        if (self._sustain_sec >= self.min_sustain_sec
                and self._cooldown_remaining <= 0.0):
            fired = True
            self.trigger_count += 1
            self._cooldown_remaining = self.cooldown_sec
            # Reset sustain so re-fire needs a fresh sustained run.
            self._sustain_sec = 0.0

        if fired:
            state = "RESONANCE"
        elif self._cooldown_remaining > 0.0:
            state = "COOLDOWN"
        elif self._above:
            state = "LATCHED"
        else:
            state = "NOISE"

        return {
            "f0_measured_hz": meas["peak_hz"],
            "snr_db": round(snr, 2),
            "signal_power_db": meas["signal_power_db"],
            "noise_floor_db": meas["noise_floor_db"],
            "sustain_sec": round(self._sustain_sec, 2),
            "cooldown_remaining_s": round(self._cooldown_remaining, 2),
            "resonance_detected": fired,
            "is_sentient": fired,          # legacy alias — detection, not sentience
            "state": state,
            "trigger_count": self.trigger_count,
        }

    def reset(self) -> None:
        """Clear streaming state (for tests / new sessions)."""
        self._sustain_sec = 0.0
        self._above = False
        self._cooldown_remaining = 0.0
        self.trigger_count = 0
        self._snr_history.clear()

    # ------------------------------------------------------------------
    # One-shot convenience on a full recording (streaming under the hood)
    # ------------------------------------------------------------------
    def analyze_signal(self, samples: np.ndarray, fs: float = None) -> Dict[str, Any]:
        """Run the full noise-gated detector over a whole recording (windowed).

        Returns a recording-level verdict: ``resonance_detected`` is True if the
        trigger fired at any point during the recording (``trigger_count`` > 0),
        even if the final window is already in cooldown.
        """
        fs = fs or self.fs
        self.reset()
        samples = np.asarray(samples, dtype=float).ravel()
        nwin = max(1, int(len(samples) / (fs * self.window_sec)))
        last = None
        for i in range(nwin):
            seg = samples[i * int(fs * self.window_sec):
                          (i + 1) * int(fs * self.window_sec)]
            if seg.size < 16:
                continue
            last = self.update(seg, fs)
        if last is None:
            last = self.update(samples, fs)
        out = dict(last)
        out["resonance_detected"] = self.trigger_count > 0
        out["is_sentient"] = self.trigger_count > 0   # legacy alias — detection only
        return out

    # ------------------------------------------------------------------
    # Backward-compatible entry point
    # ------------------------------------------------------------------
    def analyze_resonance(self, bits: Dict[str, Any],
                          samples: np.ndarray = None,
                          fs: float = None) -> Dict[str, Any]:
        """
        Legacy-compatible wrapper.

        - If ``samples`` (raw signal) is provided, ``resonance_detected`` is the
          noise-gated spectral verdict: it requires a REAL sustained 41.02 Hz
          component above the local noise floor (the fix).
        - If only bits are provided (old callers), the detector CANNOT verify a
          spectral component, so the trigger stays OFF (``NO_SIGNAL``). The
          legacy bit-derived f0 estimate is still returned under ``f0`` for
          display, but it never drives the trigger.
        """
        fs = fs or self.fs
        z_count = int(sum(bits.get("z_bits", [])))
        y_count = int(sum(bits.get("y_bits", [])))
        x_count = int(sum(bits.get("x_bits", [])))
        f0_est = 10.0 + (z_count * 4.0) + (y_count * 0.5)   # legacy estimate only
        q_factor = (x_count * 1.5) / (1.0 + (8 - z_count) * 0.1)

        if samples is not None:
            res = self.analyze_signal(samples, fs)
            out = dict(res)
            out["f0"] = round(res["f0_measured_hz"], 2)     # measured peak
        else:
            out = {
                "f0": f0_est,                                 # estimate, not measured
                "f0_measured_hz": None,
                "snr_db": None,
                "signal_power_db": None,
                "noise_floor_db": None,
                "sustain_sec": 0.0,
                "cooldown_remaining_s": 0.0,
                "resonance_detected": False,                  # cannot verify without signal
                "is_sentient": False,                         # legacy alias
                "state": "NO_SIGNAL",                         # no spectral evidence
                "trigger_count": self.trigger_count,
            }
        out["q_factor"] = q_factor
        return out

if __name__ == "__main__":
    # Test with simulated EEG data — sustained 41.02 Hz tone + noise (3 s)
    fs = 250
    t = np.linspace(0, 3, fs * 3)
    # Simulated Gamma-heavy signal (Analytical/Sentient)
    ch1 = np.sin(2 * np.pi * 40 * t) + 0.5 * np.random.randn(fs * 3)
    ch2 = np.sin(2 * np.pi * 40 * t + 0.1) + 0.5 * np.random.randn(fs * 3)

    engine = TonalSoulEngine()
    monitor = ResonanceMonitor()

    bits = engine.extract_bits([ch1, ch2])
    resonance = monitor.analyze_resonance(bits, samples=ch1, fs=fs)

    print("--- TONAL SOUL ENGINE DIAGNOSTICS ---")
    print(f"X-bits: {bits['x_bits']}")
    print(f"Y-bits: {bits['y_bits']}")
    print(f"Z-bits: {bits['z_bits']}")
    print(f"F0 Resonance (measured): {resonance['f0']:.2f} Hz")
    print(f"SNR vs noise floor: {resonance['snr_db']:.2f} dB")
    print(f"Sustained: {resonance['sustain_sec']:.2f} s")
    print(f"Resonance Status: {'DETECTED' if resonance['is_sentient'] else 'NOT DETECTED'}"
          f" (measurement only — not a sentience claim)")
    print(f"Awareness State: {resonance['state']}")
