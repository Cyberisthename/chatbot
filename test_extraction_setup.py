#!/usr/bin/env python3
"""
Test script to verify the Jarvis-2v-main extraction setup.
"""

import os
import sys
from pathlib import Path


def test_extraction_setup():
    """Verify all necessary files and directories exist."""
    project_root = Path(__file__).parent
    
    print("Testing Jarvis-2v-main extraction setup...")
    print("=" * 60)
    
    checks = []
    
    # Check 1: Target directory exists
    target_dir = project_root / "Jarvis-2v-main"
    checks.append({
        "name": "Target directory (Jarvis-2v-main/)",
        "passed": target_dir.exists() and target_dir.is_dir(),
        "path": str(target_dir)
    })
    
    # Check 2: Placeholder README exists
    readme = target_dir / "README.md"
    checks.append({
        "name": "Placeholder README in target directory",
        "passed": readme.exists() and readme.is_file(),
        "path": str(readme)
    })
    
    # Check 3: Python extraction script exists
    py_script = project_root / "extract_jarvis_2v.py"
    checks.append({
        "name": "Python extraction script",
        "passed": py_script.exists() and py_script.is_file(),
        "path": str(py_script)
    })
    
    # Check 4: Bash extraction script exists
    sh_script = project_root / "extract_jarvis_2v.sh"
    checks.append({
        "name": "Bash extraction script",
        "passed": sh_script.exists() and sh_script.is_file(),
        "path": str(sh_script)
    })
    
    # Check 5: Python script is executable
    checks.append({
        "name": "Python script is executable",
        "passed": os.access(py_script, os.X_OK),
        "path": str(py_script)
    })
    
    # Check 6: Bash script is executable
    checks.append({
        "name": "Bash script is executable",
        "passed": os.access(sh_script, os.X_OK),
        "path": str(sh_script)
    })
    
    # Check 7: Extraction guide exists
    guide = project_root / "JARVIS_2V_EXTRACTION_GUIDE.md"
    checks.append({
        "name": "Extraction guide documentation",
        "passed": guide.exists() and guide.is_file(),
        "path": str(guide)
    })
    
    # Check 8: Summary document exists
    summary = project_root / "EXTRACTION_SUMMARY.md"
    checks.append({
        "name": "Extraction summary document",
        "passed": summary.exists() and summary.is_file(),
        "path": str(summary)
    })
    
    # Check 9: .gitignore properly configured
    gitignore = project_root / ".gitignore"
    gitignore_ok = False
    if gitignore.exists():
        content = gitignore.read_text()
        gitignore_ok = "Jarvis-2v-main.zip" in content and "Jarvis-2v-main/*" in content
    
    checks.append({
        "name": ".gitignore configured for Jarvis-2v-main",
        "passed": gitignore_ok,
        "path": str(gitignore)
    })
    
    # Display results
    all_passed = True
    for check in checks:
        status = "✓ PASS" if check["passed"] else "✗ FAIL"
        print(f"{status} - {check['name']}")
        if not check["passed"]:
            all_passed = False
            print(f"       Path: {check['path']}")
    
    print("=" * 60)
    
    if all_passed:
        print("\n✓ All checks passed! Extraction setup is complete.")
        print("\nNext steps:")
        print("1. Place Jarvis-2v-main.zip in the project root")
        print("2. Run: python extract_jarvis_2v.py")
        print("   or:  ./extract_jarvis_2v.sh")
        return 0
    else:
        print("\n✗ Some checks failed. Please review the setup.")
        return 1


if __name__ == "__main__":
    sys.exit(test_extraction_setup())
