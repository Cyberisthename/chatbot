#!/usr/bin/env python3
"""
Test script to verify Ben Lab LoRA pipeline setup.

Checks:
- Python packages required for training
- Jarvis API accessibility
- Directory structure
- File permissions
"""
import sys
from pathlib import Path


def check_python_packages():
    """Check if required Python packages are installed."""
    print("Checking Python packages...")
    
    required = {
        "requests": "pip install requests",
        "transformers": "pip install transformers>=4.40",
        "datasets": "pip install datasets",
        "peft": "pip install peft",
        "torch": "pip install torch",
        "accelerate": "pip install accelerate",
    }
    
    missing = []
    for package, install_cmd in required.items():
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - Install with: {install_cmd}")
            missing.append(package)
    
    if missing:
        print()
        print("Install all missing packages:")
        print("  pip install requests transformers>=4.40 datasets peft torch accelerate")
        return False
    
    return True


def check_jarvis_api():
    """Check if Jarvis API is running."""
    print("\nChecking Jarvis API...")
    
    try:
        import requests
        response = requests.get("http://127.0.0.1:8000/health", timeout=5)
        if response.status_code == 200:
            print("  ✓ Jarvis API is running at http://127.0.0.1:8000")
            return True
        else:
            print(f"  ✗ Jarvis API returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"  ✗ Jarvis API not reachable: {e}")
        print("  Start it with: python jarvis_api.py")
        return False


def check_scripts():
    """Check if required scripts exist and are executable."""
    print("\nChecking scripts...")
    
    scripts = [
        "generate_lab_training_data.py",
        "finetune_ben_lab.py",
        "train_and_install.sh",
        "jarvis_api.py",
    ]
    
    all_good = True
    for script in scripts:
        path = Path(script)
        if path.exists():
            if path.suffix == ".sh":
                if path.stat().st_mode & 0o111:
                    print(f"  ✓ {script} (executable)")
                else:
                    print(f"  ⚠ {script} exists but not executable - run: chmod +x {script}")
            else:
                print(f"  ✓ {script}")
        else:
            print(f"  ✗ {script} not found")
            all_good = False
    
    return all_good


def check_directories():
    """Check directory structure."""
    print("\nChecking directories...")
    
    data_dir = Path("data")
    if data_dir.exists():
        print(f"  ✓ data/ directory exists")
    else:
        print(f"  ℹ data/ directory will be created by generate_lab_training_data.py")
    
    return True


def check_modelfile():
    """Check if Modelfile exists."""
    print("\nChecking Modelfile...")
    
    modelfile = Path("Modelfile")
    if modelfile.exists():
        print("  ✓ Modelfile exists")
        return True
    else:
        print("  ℹ Modelfile will be created by train_and_install.sh")
        return True


def check_ollama():
    """Check if Ollama is installed."""
    print("\nChecking Ollama...")
    
    import subprocess
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print(f"  ✓ Ollama installed: {result.stdout.strip()}")
            return True
        else:
            print("  ✗ Ollama command failed")
            return False
    except FileNotFoundError:
        print("  ⚠ Ollama not found - Install from: https://ollama.ai/download")
        print("  (Optional: needed only for final model deployment)")
        return None
    except Exception as e:
        print(f"  ⚠ Could not check Ollama: {e}")
        return None


def main():
    print("=" * 60)
    print("Ben Lab LoRA Pipeline Setup Check")
    print("=" * 60)
    print()
    
    checks = {
        "Python packages": check_python_packages(),
        "Scripts": check_scripts(),
        "Directories": check_directories(),
        "Modelfile": check_modelfile(),
        "Jarvis API": check_jarvis_api(),
    }
    
    ollama_status = check_ollama()
    if ollama_status is not None:
        checks["Ollama"] = ollama_status
    
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    
    for check_name, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:10} {check_name}")
    
    print()
    
    all_critical_passed = all([
        checks["Python packages"],
        checks["Scripts"],
    ])
    
    if all_critical_passed:
        print("✅ Core setup complete!")
        print()
        if not checks.get("Jarvis API", False):
            print("⚠ Note: Start Jarvis API before generating training data:")
            print("  python jarvis_api.py")
            print()
        print("Next steps:")
        print("  1. python jarvis_api.py              # Start API (terminal 1)")
        print("  2. python generate_lab_training_data.py  # Generate data (terminal 2)")
        print("  3. python finetune_ben_lab.py       # Fine-tune model")
        print("  4. ./train_and_install.sh           # Or use one-shot script")
        print()
        print("Docs: BEN_LAB_LORA_OLLAMA.md")
    else:
        print("❌ Setup incomplete. Fix the issues above and try again.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
