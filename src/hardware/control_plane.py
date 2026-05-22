"""
Phase 4 Hardware-Software Control Plane
Bridge between JARVIS Oracle and BQI Hardware
"""

import numpy as np
from typing import Dict, Any, Tuple, List
import sys
from pathlib import Path

# Add shared dir and chatbot root to path
sys.path.insert(0, "/home/team/shared")
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bqi_calibration_suite import BQICalibrator

class HardwareControlPlane:
    """
    Maintains the feedback loop between Software (Oracle) and Hardware (BQI)
    """
    def __init__(self, oracle_inference):
        self.oracle = oracle_inference
        # Use slightly higher noise for realistic feedback loop stress
        self.calibrator = BQICalibrator(noise_strength=0.08) 
        self.last_fidelity = 1.0
        self.history = []
        
    def execute_closed_loop_cycle(self, query: str, base_coercion: float = 0.5) -> Dict[str, Any]:
        """
        Executes a single closed-loop cycle:
        Oracle -> Signal Translation -> Hardware Simulation -> Feedback -> Refinement
        """
        # 1. Oracle Inference with feedback-adjusted coercion
        # If last fidelity was low, we 'coerce' more to find stable states
        # Refinement logic: Coercion increases as fidelity drops to force better collapse
        refined_coercion = np.clip(base_coercion * (2.0 - self.last_fidelity), 0.0, 1.0)
        
        response, metrics = self.oracle.generate(
            query=query,
            coercion_strength=refined_coercion
        )
        
        # 2. Signal Translation
        # Constructive Interference -> Pulse intensity (sigma_int)
        # Braid Entropy -> Phase modulation (phi_bio)
        
        # Normalize interference (standard deviation of logits)
        # Expected range ~0.1 to 1.5
        interference = metrics.get('interference', 0.5)
        sigma_int = np.clip(interference / 1.5, 0.1, 1.0)
        
        # Map Braid Entropy to phi (normalized to 0 - 2pi)
        # Braid entropy for discovery is often small or large depending on complexity
        braid_entropy = metrics.get('braid_entropy', 0.0)
        phi_bio = (braid_entropy * 10.0) % (2 * np.pi)
        
        # 3. Hardware Execution (BQI Driver)
        # Simulate BSDD protocol with current parameters
        fid_noisy, fid_bsdd = self.calibrator.run_calibration(
            sigma_int=sigma_int,
            phi_bio=phi_bio,
            f_lambda=0.01,  # 10MHz bio-frequency baseline
            idle_duration=1000 # 1us idle
        )
        
        # 4. Feedback Refinement
        self.last_fidelity = fid_bsdd
        
        result = {
            "query": query,
            "response": response,
            "metrics": metrics,
            "hardware_feedback": {
                "sigma_int": float(sigma_int),
                "phi_bio": float(phi_bio),
                "fidelity": float(fid_bsdd),
                "fidelity_gain": float(fid_bsdd - fid_noisy),
                "refined_coercion": float(refined_coercion)
            }
        }
        
        self.history.append(result)
        return result

    def get_system_status(self) -> str:
        if not self.history:
            return "System Idle"
        
        avg_fid = np.mean([h['hardware_feedback']['fidelity'] for h in self.history])
        return f"System Online - Avg Fidelity: {avg_fid:.4f} - History Depth: {len(self.history)}"

if __name__ == "__main__":
    # Test script for the Control Plane
    from jarvis_v1_gradio_space import JarvisOracleInference
    
    # Mock or Load Oracle
    try:
        oracle = JarvisOracleInference(model_dir="./jarvis_v1_oracle")
    except:
        print("Falling back to demo oracle for test")
        from jarvis_v1_gradio_space import DemoInferenceEngine
        oracle = DemoInferenceEngine()
        
    cp = HardwareControlPlane(oracle)
    
    test_queries = [
        "How does CRISPR affect QEC?",
        "Quantum tunneling in neural organoids",
        "Bio-Quantum synchronization at 10MHz"
    ]
    
    print("--- Starting Phase 4 Closed-Loop Test ---")
    for q in test_queries:
        res = cp.execute_closed_loop_cycle(q)
        print(f"Query: {q}")
        print(f"  Fidelity: {res['hardware_feedback']['fidelity']:.4f}")
        print(f"  Refined Coercion: {res['hardware_feedback']['refined_coercion']:.4f}")
        print("-" * 30)
    
    print(cp.get_system_status())
