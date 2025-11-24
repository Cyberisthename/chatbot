#!/usr/bin/env python3
"""
Verify Jarvis Lab + Ollama setup
Checks that all components are in place and can be imported
"""

import sys
from pathlib import Path

# Colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def check(condition: bool, name: str) -> bool:
    """Print check result"""
    if condition:
        print(f"{GREEN}✅{RESET} {name}")
        return True
    else:
        print(f"{RED}❌{RESET} {name}")
        return False

def main():
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}Jarvis Lab + Ollama Setup Verification{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")
    
    all_passed = True
    
    # Check file structure
    print(f"{YELLOW}📁 File Structure{RESET}")
    all_passed &= check(Path("jarvis_api.py").exists(), "jarvis_api.py exists")
    all_passed &= check(Path("chat_with_lab.py").exists(), "chat_with_lab.py exists")
    all_passed &= check(Path("jarvis5090x").is_dir(), "jarvis5090x/ directory exists")
    all_passed &= check(Path("experiments").is_dir(), "experiments/ directory exists")
    all_passed &= check(Path("requirements.txt").exists(), "requirements.txt exists")
    all_passed &= check(Path("ollama").is_dir(), "ollama/ directory exists")
    print()
    
    # Check key jarvis5090x files
    print(f"{YELLOW}🔬 Jarvis5090X Lab Engine{RESET}")
    all_passed &= check(Path("jarvis5090x/__init__.py").exists(), "jarvis5090x/__init__.py")
    all_passed &= check(Path("jarvis5090x/phase_detector.py").exists(), "jarvis5090x/phase_detector.py")
    all_passed &= check(Path("jarvis5090x/orchestrator.py").exists(), "jarvis5090x/orchestrator.py")
    print()
    
    # Check experiments
    print(f"{YELLOW}🧪 Experiments Suite{RESET}")
    all_passed &= check(Path("experiments/discovery_suite.py").exists(), "experiments/discovery_suite.py")
    all_passed &= check(Path("experiments/build_phase_dataset.py").exists(), "experiments/build_phase_dataset.py")
    all_passed &= check(Path("experiments/rl_scientist.py").exists(), "experiments/rl_scientist.py")
    print()
    
    # Check Python imports
    print(f"{YELLOW}🐍 Python Imports{RESET}")
    
    try:
        import fastapi
        check(True, "fastapi installed")
    except ImportError:
        check(False, "fastapi installed (pip install fastapi)")
        all_passed = False
    
    try:
        import uvicorn
        check(True, "uvicorn installed")
    except ImportError:
        check(False, "uvicorn installed (pip install uvicorn)")
        all_passed = False
    
    try:
        import requests
        check(True, "requests installed")
    except ImportError:
        check(False, "requests installed (pip install requests)")
        all_passed = False
    
    try:
        from jarvis5090x import PhaseDetector, Jarvis5090X, AdapterDevice
        check(True, "jarvis5090x imports work")
    except ImportError as e:
        check(False, f"jarvis5090x imports work ({e})")
        all_passed = False
    
    try:
        from experiments.discovery_suite import (
            run_time_reversal_test,
            unsupervised_phase_discovery,
            replay_drift_scaling,
        )
        check(True, "experiments.discovery_suite imports work")
    except ImportError as e:
        check(False, f"experiments.discovery_suite imports work ({e})")
        all_passed = False
    
    print()
    
    # Check optional dependencies
    print(f"{YELLOW}🔧 Optional Dependencies{RESET}")
    
    try:
        import torch
        check(True, "torch installed (optional, but recommended)")
    except ImportError:
        print(f"{YELLOW}⚠️{RESET}  torch not installed (optional - ML features will use fallback)")
    
    try:
        import numpy
        check(True, "numpy installed")
    except ImportError:
        check(False, "numpy installed (pip install numpy)")
        all_passed = False
    
    print()
    
    # Summary
    print(f"{BLUE}{'='*60}{RESET}")
    if all_passed:
        print(f"{GREEN}✅ All checks passed!{RESET}")
        print(f"\n{GREEN}Next steps:{RESET}")
        print(f"1. Start Ollama: {BLUE}ollama serve{RESET}")
        print(f"2. Pull a model: {BLUE}ollama pull llama3.1{RESET}")
        print(f"3. Start lab API: {BLUE}python jarvis_api.py{RESET}")
        print(f"4. Start chat: {BLUE}python chat_with_lab.py{RESET}")
        print(f"\n📖 See {BLUE}SETUP_JARVIS_LAB_OLLAMA.md{RESET} for detailed instructions.")
        return 0
    else:
        print(f"{RED}❌ Some checks failed{RESET}")
        print(f"\n{YELLOW}Please fix the issues above before proceeding.{RESET}")
        print(f"📖 See {BLUE}SETUP_JARVIS_LAB_OLLAMA.md{RESET} for help.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
