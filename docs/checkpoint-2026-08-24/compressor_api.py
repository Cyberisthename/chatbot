import sys
import json
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

def run_compression(seed_vals, n_qubits=256):
    try:
        compressor = FractalBraidSeedCompressor(n_effective_qubits=n_qubits, seed=tuple(seed_vals))
        amps, positions, metrics = compressor.reconstruct()
        
        result = {
            "metrics": {
                "effective_qubits": int(metrics["effective_qubits"]),
                "compression_ratio": float(metrics["compression_ratio"]),
                "reconstruction_mse": float(metrics["reconstruction_mse"]),
                "geometric_fold_factor": float(metrics["geometric_fold_factor"]),
                "memory_kb": float(metrics["memory_kb"]),
                "avg_braid_crossings": float(metrics["avg_braid_crossings"]),
                "owner_seed_hash": str(metrics["owner_seed_hash"])
            },
            "samples": [
                {"amp": float(np.abs(amps[i])), "phase": float(np.angle(amps[i])), "pos": [float(p) for p in positions[i]]}
                for i in range(min(100, len(amps)))
            ]
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
        print(json.dumps(run_compression(seeds, n_qubits)))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
