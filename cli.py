#!/usr/bin/env python3
"""
Top-level CLI dispatcher for quantum experiments
"""
import sys
import runpy


def main():
    if len(sys.argv) < 2:
        print("Usage: python cli.py <command>")
        print("\nAvailable commands:")
        print("  adapter-double-slit    - Run digital double-slit interference experiment")
        print("  atom-from-constants    - Solve atom from Schrödinger equation")
        print("  solve-atom             - Alias for atom-from-constants")
        print("  atom-3d-discovery      - 3D atom solver with progressive resolution (physics-only)")
        print("  atom-3d-v2             - Full-resolution split-operator atom solver")
        sys.exit(1)
    
    command = sys.argv[1]
    
    # Remove the first two arguments (script name and command) so subcommands can parse their own args
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    
    if command == "adapter-double-slit":
        runpy.run_module("experiments.adapter_double_slit", run_name="__main__")
    elif command in ["atom-from-constants", "solve-atom"]:
        runpy.run_path("quantacap/src/quantacap/experiments/solve_atom_from_constants.py", run_name="__main__")
    elif command == "atom-3d-discovery":
        runpy.run_module("experiments.solve_atom_3d_discovery", run_name="__main__")
    elif command == "atom-3d-v2":
        runpy.run_module("experiments.solve_atom_3d_v2", run_name="__main__")
    else:
        print(f"Unknown command: {command}")
        print("\nAvailable commands:")
        print("  adapter-double-slit    - Run digital double-slit interference experiment")
        print("  atom-from-constants    - Solve atom from Schrödinger equation")
        print("  solve-atom             - Alias for atom-from-constants")
        print("  atom-3d-discovery      - 3D atom solver with progressive resolution (physics-only)")
        print("  atom-3d-v2             - Full-resolution split-operator atom solver")
        sys.exit(1)


if __name__ == "__main__":
    main()
