import sys
import json
import io
import contextlib
import numpy as np
from pathlib import Path
# Add chatbot path to sys.path
sys.path.append("/home/team/shared/chatbot")
from compression_specialist import FractalBraidSeedCompressor
# TonalSoulEngine and ResonanceMonitor might need more dependencies or specific setup
# Let's try to import them but handle if they fail
try:
    from src.quantum_llm.eeg_to_tonal_engine import TonalSoulEngine, ResonanceMonitor
    HAS_TONAL = True
except ImportError:
    HAS_TONAL = False

QUDIT_NAMES = {2: "qubit", 3: "qutrit", 4: "ququart", 5: "ququint", 6: "quhex"}


def run_compression(seed_vals, n_qubits=256, qudit_dim=2):
    try:
        qudit_dim = max(2, min(int(qudit_dim), 6))
        # FBSC prints an init banner to stdout — silence it so the API output stays pure JSON
        with contextlib.redirect_stdout(io.StringIO()):
            compressor = FractalBraidSeedCompressor(
                n_effective_qubits=n_qubits, seed=tuple(seed_vals), qudit_dim=qudit_dim
            )
            amps, positions, metrics = compressor.reconstruct()
        dim = int(metrics.get("qudit_dim", qudit_dim))
        # qudit states are (n_logical, d) — flatten levels for the 1-D visualizers
        n_logical = amps.shape[0]
        d = amps.shape[1] if amps.ndim > 1 else 1
        flat = amps.reshape(-1)
        samples = []
        for i in range(min(100, len(flat))):
            unit = i // d
            samples.append({
                "amp": float(np.abs(flat[i])),
                "phase": float(np.angle(flat[i])),
                "pos": [float(p) for p in positions[unit % n_logical]],
            })
        result = {
            "metrics": {
                "effective_qudits": int(n_logical),
                "qudit_dimension": dim,
                "qudit_basis": QUDIT_NAMES.get(dim, f"d={dim}"),
                "total_hilbert_dim": int(metrics.get("total_hilbert_dim", 0)),
                "compression_ratio": float(metrics.get("compression_ratio", 0.0)),
                "reconstruction_mse": float(metrics.get("reconstruction_mse", 0.0)),
                "geometric_fold_factor": float(metrics.get("geometric_fold_factor", 0.0)),
                "memory_kb": float(metrics.get("memory_kb", 0.0)),
                "avg_braid_crossings": float(metrics.get("avg_braid_crossings", 0.0)),
                "owner_seed_hash": str(metrics.get("owner_seed_hash", ""))
            },
            "samples": samples
        }
        if HAS_TONAL:
            try:
                tonal_engine = TonalSoulEngine()
                resonance_monitor = ResonanceMonitor()
                # Simulate EEG as in qvgpu_compressor.py
                fs = 250
                t = np.linspace(0, 1, fs)
                ch1 = np.sin(2 * np.pi * 41.02 * t) + 0.5 * np.random.randn(fs)
                ch2 = np.sin(2 * np.pi * 41.02 * t + 0.05) + 0.5 * np.random.randn(fs)
                bits = tonal_engine.extract_bits([ch1, ch2])
                resonance = resonance_monitor.analyze_resonance(bits)
                result["bio_resonance"] = {
                    "f0": float(resonance["f0"]),
                    "is_sentient": bool(resonance["is_sentient"]),
                    "q_factor": float(resonance.get("q_factor", 0))
                }
            except Exception as e:
                result["bio_resonance_error"] = str(e)
        return result
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(json.dumps({"error": "Missing seeds"}))
        sys.exit(1)
    try:
        seeds = [float(arg) for arg in sys.argv[1:4]]
        n_qubits = int(sys.argv[4]) if len(sys.argv) > 4 else 256
        qudit_dim = int(sys.argv[5]) if len(sys.argv) > 5 else 2
        print(json.dumps(run_compression(seeds, n_qubits, qudit_dim)))
    except Exception as e:
        print(json.dumps({"error": str(e)}))