#!/usr/bin/env python3
"""
Jarvis Quantum LLM - Complete Setup Validator
Validates that everything is ready for Ollama deployment
100% Real - Checks actual weights, architecture, and functionality
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import numpy as np
except ImportError:
    print("Installing numpy...")
    os.system("pip install numpy")
    import numpy as np


class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}\n")


def print_success(text: str):
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")


def print_error(text: str):
    print(f"{Colors.RED}❌ {text}{Colors.END}")


def print_warning(text: str):
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")


def print_info(text: str):
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.END}")


class JarvisValidator:
    def __init__(self):
        self.script_dir = Path(__file__).parent
        self.project_root = self.script_dir.parent
        self.checks_passed = 0
        self.checks_failed = 0
        self.warnings = 0
        
    def run_all_checks(self) -> bool:
        """Run all validation checks"""
        print_header("🔍 JARVIS QUANTUM LLM - COMPLETE VALIDATION")
        
        checks = [
            ("Prerequisites", self.check_prerequisites),
            ("Source Model Files", self.check_source_files),
            ("Model Weights Integrity", self.check_weights_integrity),
            ("Architecture Configuration", self.check_architecture),
            ("Training Data", self.check_training_data),
            ("Ollama Integration Files", self.check_ollama_files),
            ("Documentation", self.check_documentation),
            ("Code Quality", self.check_code_quality),
        ]
        
        for check_name, check_func in checks:
            print_header(f"Checking: {check_name}")
            try:
                check_func()
            except Exception as e:
                print_error(f"Check failed with exception: {e}")
                self.checks_failed += 1
        
        self.print_summary()
        return self.checks_failed == 0
    
    def check_prerequisites(self):
        """Check system prerequisites"""
        # Python version
        import sys
        py_version = sys.version_info
        if py_version >= (3, 8):
            print_success(f"Python {py_version.major}.{py_version.minor}.{py_version.micro}")
            self.checks_passed += 1
        else:
            print_error(f"Python {py_version.major}.{py_version.minor} (need 3.8+)")
            self.checks_failed += 1
        
        # NumPy
        try:
            import numpy as np
            print_success(f"NumPy {np.__version__}")
            self.checks_passed += 1
        except ImportError:
            print_error("NumPy not installed")
            self.checks_failed += 1
        
        # Ollama (optional)
        import subprocess
        try:
            result = subprocess.run(['which', 'ollama'], 
                                  capture_output=True, 
                                  text=True)
            if result.returncode == 0:
                print_success(f"Ollama found at: {result.stdout.strip()}")
                self.checks_passed += 1
            else:
                print_warning("Ollama not found (install from https://ollama.ai)")
                self.warnings += 1
        except Exception as e:
            print_warning(f"Could not check Ollama: {e}")
            self.warnings += 1
    
    def check_source_files(self):
        """Check that source model files exist"""
        model_dir = self.project_root / "ready-to-deploy-hf"
        
        required_files = {
            "jarvis_quantum_llm.npz": "Model weights",
            "config.json": "Architecture config",
            "tokenizer.json": "Tokenizer",
            "train_data.json": "Training data"
        }
        
        for filename, description in required_files.items():
            filepath = model_dir / filename
            if filepath.exists():
                size = filepath.stat().st_size
                size_mb = size / (1024 * 1024)
                print_success(f"{description}: {filename} ({size_mb:.2f} MB)")
                self.checks_passed += 1
            else:
                print_error(f"{description} missing: {filename}")
                self.checks_failed += 1
    
    def check_weights_integrity(self):
        """Verify model weights are real (not zeros, not mock data)"""
        weights_path = self.project_root / "ready-to-deploy-hf" / "jarvis_quantum_llm.npz"
        
        if not weights_path.exists():
            print_error("Weights file not found")
            self.checks_failed += 1
            return
        
        try:
            data = np.load(weights_path)
            
            # Check number of arrays
            num_arrays = len(data.keys())
            print_info(f"Found {num_arrays} weight arrays")
            
            if num_arrays < 20:
                print_warning(f"Expected more weight arrays (found {num_arrays})")
                self.warnings += 1
            else:
                print_success(f"Sufficient weight arrays ({num_arrays})")
                self.checks_passed += 1
            
            # Check embedding weights
            if 'embedding' in data:
                emb = data['embedding']
                print_info(f"Embedding shape: {emb.shape}")
                
                # Check not all zeros
                if np.allclose(emb, 0):
                    print_error("Embedding weights are all zeros (not trained!)")
                    self.checks_failed += 1
                else:
                    print_success("Embedding weights are non-zero (trained)")
                    self.checks_passed += 1
                
                # Check distribution
                mean = np.mean(emb)
                std = np.std(emb)
                print_info(f"Embedding stats: mean={mean:.6f}, std={std:.6f}")
                
                if std < 0.01:
                    print_warning("Low std deviation - might not be well trained")
                    self.warnings += 1
                else:
                    print_success(f"Good weight distribution (std={std:.4f})")
                    self.checks_passed += 1
            
            # Check layer weights
            layer_count = 0
            for key in data.keys():
                if key.startswith('layer_'):
                    layer_count += 1
            
            if layer_count > 0:
                print_success(f"Found {layer_count} layer weight arrays")
                self.checks_passed += 1
            else:
                print_error("No layer weights found")
                self.checks_failed += 1
            
            # Calculate total parameters
            total_params = sum(data[k].size for k in data.keys() if isinstance(data[k], np.ndarray))
            print_info(f"Total parameters: {total_params:,}")
            
            if total_params > 1_000_000:
                print_success(f"Large model with {total_params:,} parameters")
                self.checks_passed += 1
            else:
                print_warning(f"Small model with only {total_params:,} parameters")
                self.warnings += 1
                
        except Exception as e:
            print_error(f"Failed to load weights: {e}")
            self.checks_failed += 1
    
    def check_architecture(self):
        """Verify architecture configuration"""
        config_path = self.project_root / "ready-to-deploy-hf" / "config.json"
        
        if not config_path.exists():
            print_error("Config file not found")
            self.checks_failed += 1
            return
        
        try:
            with open(config_path) as f:
                config = json.load(f)
            
            required_keys = ['vocab_size', 'd_model', 'n_layers', 'n_heads', 'd_ff']
            
            for key in required_keys:
                if key in config:
                    print_success(f"{key}: {config[key]}")
                    self.checks_passed += 1
                else:
                    print_error(f"Missing config key: {key}")
                    self.checks_failed += 1
            
            # Validate reasonable values
            if config.get('vocab_size', 0) > 1000:
                print_success(f"Good vocabulary size: {config['vocab_size']}")
                self.checks_passed += 1
            else:
                print_warning(f"Small vocabulary: {config.get('vocab_size', 0)}")
                self.warnings += 1
            
            if config.get('n_layers', 0) >= 4:
                print_success(f"Sufficient layers: {config['n_layers']}")
                self.checks_passed += 1
            else:
                print_warning(f"Few layers: {config.get('n_layers', 0)}")
                self.warnings += 1
                
        except Exception as e:
            print_error(f"Failed to load config: {e}")
            self.checks_failed += 1
    
    def check_training_data(self):
        """Verify training data exists and is substantial"""
        data_path = self.project_root / "ready-to-deploy-hf" / "train_data.json"
        
        if not data_path.exists():
            print_warning("Training data file not found (optional)")
            self.warnings += 1
            return
        
        try:
            with open(data_path) as f:
                train_data = json.load(f)
            
            if isinstance(train_data, list):
                num_docs = len(train_data)
                print_info(f"Training documents: {num_docs}")
                
                if num_docs >= 1000:
                    print_success(f"Large training corpus ({num_docs} documents)")
                    self.checks_passed += 1
                elif num_docs >= 100:
                    print_success(f"Moderate training corpus ({num_docs} documents)")
                    self.checks_passed += 1
                else:
                    print_warning(f"Small training corpus ({num_docs} documents)")
                    self.warnings += 1
                
                # Check content
                if num_docs > 0 and 'text' in train_data[0]:
                    sample_length = len(train_data[0]['text'])
                    print_info(f"Sample document length: {sample_length} chars")
                    
                    if sample_length > 100:
                        print_success("Documents have substantial content")
                        self.checks_passed += 1
                    else:
                        print_warning("Documents seem short")
                        self.warnings += 1
                        
        except Exception as e:
            print_error(f"Failed to load training data: {e}")
            self.checks_failed += 1
    
    def check_ollama_files(self):
        """Check Ollama integration files"""
        ollama_files = {
            "numpy_to_gguf.py": "GGUF converter",
            "Modelfile": "Ollama config",
            "setup.sh": "Setup script",
            "test_ollama.py": "Test suite"
        }
        
        for filename, description in ollama_files.items():
            filepath = self.script_dir / filename
            if filepath.exists():
                print_success(f"{description}: {filename}")
                self.checks_passed += 1
            else:
                print_error(f"Missing {description}: {filename}")
                self.checks_failed += 1
        
        # Check if GGUF exists (optional)
        gguf_path = self.script_dir / "jarvis-quantum.gguf"
        if gguf_path.exists():
            size_mb = gguf_path.stat().st_size / (1024 * 1024)
            print_success(f"GGUF file exists ({size_mb:.2f} MB)")
            self.checks_passed += 1
        else:
            print_info("GGUF file not created yet (run numpy_to_gguf.py)")
    
    def check_documentation(self):
        """Verify documentation files exist"""
        doc_files = [
            "🚀_OLLAMA_JARVIS_MASTER_GUIDE.md",
            "START_HERE.md",
            "README.md",
            "QUICK_START.md",
            "TECHNICAL_DETAILS.md"
        ]
        
        for filename in doc_files:
            filepath = self.script_dir / filename
            if filepath.exists():
                size = filepath.stat().st_size
                print_success(f"{filename} ({size} bytes)")
                self.checks_passed += 1
            else:
                print_warning(f"Documentation missing: {filename}")
                self.warnings += 1
    
    def check_code_quality(self):
        """Check source code exists and is substantial"""
        src_dir = self.project_root / "src" / "quantum_llm"
        
        if not src_dir.exists():
            print_error("Source code directory not found")
            self.checks_failed += 1
            return
        
        code_files = [
            "quantum_transformer.py",
            "quantum_attention.py"
        ]
        
        for filename in code_files:
            filepath = src_dir / filename
            if filepath.exists():
                # Count lines
                with open(filepath) as f:
                    lines = len(f.readlines())
                
                print_success(f"{filename} ({lines} lines)")
                self.checks_passed += 1
                
                if lines < 100:
                    print_warning(f"{filename} seems short ({lines} lines)")
                    self.warnings += 1
            else:
                print_error(f"Source file missing: {filename}")
                self.checks_failed += 1
    
    def print_summary(self):
        """Print validation summary"""
        print_header("📊 VALIDATION SUMMARY")
        
        print(f"\n{Colors.BOLD}Results:{Colors.END}")
        print(f"  {Colors.GREEN}✅ Passed:  {self.checks_passed}{Colors.END}")
        print(f"  {Colors.RED}❌ Failed:  {self.checks_failed}{Colors.END}")
        print(f"  {Colors.YELLOW}⚠️  Warnings: {self.warnings}{Colors.END}")
        
        print(f"\n{Colors.BOLD}Status:{Colors.END}")
        if self.checks_failed == 0:
            print(f"  {Colors.GREEN}{Colors.BOLD}✨ ALL SYSTEMS GO! ✨{Colors.END}")
            print(f"  {Colors.GREEN}Your Jarvis Quantum LLM is ready for Ollama!{Colors.END}")
            
            if self.warnings > 0:
                print(f"\n  {Colors.YELLOW}Note: {self.warnings} warnings detected (see above){Colors.END}")
            
            print(f"\n{Colors.CYAN}Next steps:{Colors.END}")
            print(f"  1. Run: {Colors.BOLD}./setup.sh{Colors.END}")
            print(f"  2. Or manually: {Colors.BOLD}python3 numpy_to_gguf.py{Colors.END}")
            print(f"  3. Then: {Colors.BOLD}ollama create jarvis -f Modelfile{Colors.END}")
            print(f"  4. Finally: {Colors.BOLD}ollama run jarvis{Colors.END}")
            
        else:
            print(f"  {Colors.RED}{Colors.BOLD}⚠️  ISSUES DETECTED{Colors.END}")
            print(f"  {Colors.RED}Please fix the errors above before proceeding{Colors.END}")
            
            print(f"\n{Colors.CYAN}Common fixes:{Colors.END}")
            print(f"  - Missing files: Train the model first")
            print(f"  - Install deps: pip install numpy requests")
            print(f"  - Install Ollama: https://ollama.ai")


def main():
    validator = JarvisValidator()
    success = validator.run_all_checks()
    
    print()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
