#!/usr/bin/env python3
"""
Ollama Lab Integration - Connect Ollama LLM to PhaseDetector

This script bridges the Ollama API with your actual PhaseDetector code,
allowing the LLM to design experiments and interpret results.
"""
import json
import re
import requests
from typing import Dict, Any, Optional
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from jarvis5090x import (
        Jarvis5090X,
        AdapterDevice,
        DeviceKind,
        OperationKind,
        PhaseDetector,
    )
    PHASE_DETECTOR_AVAILABLE = True
except ImportError:
    PHASE_DETECTOR_AVAILABLE = False
    print("Warning: PhaseDetector not available. Running in demo mode.")


class OllamaLabAssistant:
    """
    AI assistant that can both chat about the lab AND run actual experiments.
    """
    
    def __init__(
        self,
        model_name: str = "ben-lab",
        ollama_host: str = "http://localhost:11434",
        enable_execution: bool = True
    ):
        self.model_name = model_name
        self.ollama_host = ollama_host
        self.enable_execution = enable_execution
        
        # Initialize PhaseDetector if available and enabled
        self.detector = None
        if PHASE_DETECTOR_AVAILABLE and enable_execution:
            self._setup_detector()
    
    def _setup_detector(self):
        """Initialize PhaseDetector with virtual device."""
        devices = [
            AdapterDevice(
                id="quantum_sim",
                label="Quantum Simulator",
                kind=DeviceKind.VIRTUAL,
                perf_score=50.0,
                max_concurrency=8,
                capabilities={OperationKind.QUANTUM},
            )
        ]
        orchestrator = Jarvis5090X(devices)
        self.detector = PhaseDetector(orchestrator)
        print("✅ PhaseDetector initialized and ready")
    
    def chat(self, prompt: str) -> str:
        """Send prompt to Ollama and get response."""
        try:
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            return f"Error communicating with Ollama: {e}"
    
    def parse_experiment_request(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Parse natural language into experiment parameters.
        
        Looks for patterns like:
        - "Run ising phase with depth 12"
        - "Test SPT with bias 0.7"
        - "Measure TRI for pseudorandom"
        """
        params = {}
        
        # Extract phase type
        phase_patterns = [
            (r'\bising\b', 'ising_symmetry_breaking'),
            (r'\bspt\b', 'spt_cluster'),
            (r'\btrivial\b', 'trivial_product'),
            (r'\bpseudorandom\b', 'pseudorandom'),
        ]
        for pattern, phase_type in phase_patterns:
            if re.search(pattern, text.lower()):
                params['phase_type'] = phase_type
                break
        
        # Extract depth
        depth_match = re.search(r'\bdepth[=\s]+(\d+)', text.lower())
        if depth_match:
            params['depth'] = int(depth_match.group(1))
        
        # Extract bias
        bias_match = re.search(r'\bbias[=\s]+(0\.\d+)', text.lower())
        if bias_match:
            params['bias'] = float(bias_match.group(1))
        
        # Extract system size
        size_match = re.search(r'\bsize[=\s]+(\d+)', text.lower())
        if size_match:
            params['system_size'] = int(size_match.group(1))
        
        # Check if this looks like an experiment request
        if 'phase_type' in params or any(
            keyword in text.lower()
            for keyword in ['run', 'execute', 'test', 'experiment', 'measure']
        ):
            # Fill in defaults
            params.setdefault('phase_type', 'ising_symmetry_breaking')
            params.setdefault('depth', 8)
            params.setdefault('system_size', 32)
            params.setdefault('seed', 42)
            return params
        
        return None
    
    def run_experiment(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute experiment with PhaseDetector."""
        if not self.detector:
            return {
                'error': 'PhaseDetector not available',
                'params': params
            }
        
        try:
            result = self.detector.run_phase_experiment(**params)
            return {
                'success': True,
                'experiment_id': result['experiment_id'],
                'phase_type': result['phase_type'],
                'feature_vector': result['feature_vector'],
                'summary': result['summary'],
                'params': params
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'params': params
            }
    
    def measure_tri(self, phase_type: str, depth: int = 12, bias: float = 0.7) -> Dict[str, Any]:
        """Measure Time-Reversal Instability for a phase."""
        if not self.detector:
            return {'error': 'PhaseDetector not available'}
        
        # Forward experiment
        result_fwd = self.detector.run_phase_experiment(
            phase_type=phase_type,
            system_size=32,
            depth=depth,
            seed=42,
            bias=bias
        )
        
        # Reverse experiment
        result_rev = self.detector.run_phase_experiment(
            phase_type=phase_type,
            system_size=32,
            depth=depth,
            seed=42,
            bias=1.0 - bias
        )
        
        # Compute TRI (L2 distance)
        import numpy as np
        fv1 = np.array(result_fwd['feature_vector'])
        fv2 = np.array(result_rev['feature_vector'])
        tri = float(np.linalg.norm(fv1 - fv2))
        
        return {
            'phase_type': phase_type,
            'bias': bias,
            'depth': depth,
            'tri': tri,
            'forward_features': result_fwd['feature_vector'],
            'reverse_features': result_rev['feature_vector']
        }
    
    def interactive_session(self):
        """Run interactive lab assistant session."""
        print("=" * 60)
        print("BEN-LAB Interactive Assistant")
        print("=" * 60)
        print("Type 'exit' to quit")
        print("Type 'run <experiment>' to execute (e.g., 'run ising depth 12')")
        print("Type 'tri <phase>' to measure TRI")
        print("Or just ask questions!\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == 'exit':
                    print("👋 Goodbye!")
                    break
                
                # Check for TRI measurement request
                if user_input.lower().startswith('tri '):
                    phase = user_input[4:].strip()
                    # Map simple names
                    phase_map = {
                        'ising': 'ising_symmetry_breaking',
                        'spt': 'spt_cluster',
                        'trivial': 'trivial_product',
                        'pseudorandom': 'pseudorandom'
                    }
                    phase_type = phase_map.get(phase.lower(), phase)
                    
                    print(f"\n🔬 Measuring TRI for {phase_type}...")
                    result = self.measure_tri(phase_type)
                    
                    if 'error' in result:
                        print(f"Error: {result['error']}")
                    else:
                        print(f"TRI = {result['tri']:.6f}")
                        print(f"Bias: {result['bias']}, Depth: {result['depth']}")
                    
                    continue
                
                # Check for experiment execution request
                exp_params = self.parse_experiment_request(user_input)
                
                if exp_params and self.enable_execution:
                    # Ask LLM to design the experiment first
                    design_prompt = f"Design an experiment for: {user_input}\n\nProvide concrete parameters."
                    llm_design = self.chat(design_prompt)
                    print(f"\n🤖 BEN-LAB: {llm_design}\n")
                    
                    # Execute
                    print(f"🔬 Executing: {exp_params}")
                    result = self.run_experiment(exp_params)
                    
                    if result.get('success'):
                        print(f"✅ Experiment complete!")
                        print(f"ID: {result['experiment_id']}")
                        print(f"Summary: {result['summary']}")
                        
                        # Ask LLM to interpret
                        interpret_prompt = f"""
I ran this experiment:
Phase: {result['phase_type']}
Params: {result['params']}
Results: {result['summary']}

Interpret these results.
"""
                        interpretation = self.chat(interpret_prompt)
                        print(f"\n🤖 BEN-LAB: {interpretation}")
                    else:
                        print(f"❌ Error: {result.get('error')}")
                    
                    continue
                
                # Regular chat
                response = self.chat(user_input)
                print(f"\n🤖 BEN-LAB: {response}\n")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    """Run the interactive lab assistant."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ollama Lab Integration")
    parser.add_argument(
        '--model',
        default='ben-lab',
        help='Ollama model name (default: ben-lab)'
    )
    parser.add_argument(
        '--host',
        default='http://localhost:11434',
        help='Ollama host (default: http://localhost:11434)'
    )
    parser.add_argument(
        '--no-execution',
        action='store_true',
        help='Disable actual experiment execution (chat only)'
    )
    
    args = parser.parse_args()
    
    assistant = OllamaLabAssistant(
        model_name=args.model,
        ollama_host=args.host,
        enable_execution=not args.no_execution
    )
    
    assistant.interactive_session()


if __name__ == "__main__":
    main()
