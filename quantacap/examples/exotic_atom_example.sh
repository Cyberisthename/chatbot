#!/bin/bash
# Example script for running the exotic atom Floquet experiment
#
# This experiment simulates a Floquet-driven long-range Hamiltonian on N qubits,
# logs entanglement and energy, and converts per-qubit probabilities into a
# synthetic "atom" density visualization saved as PNG and GIF.

cd "$(dirname "$0")/../.."

# Set up PYTHONPATH to include quantacap source
export PYTHONPATH="${PWD}/quantacap/src:${PWD}/quantacap:${PYTHONPATH}"

echo "Running exotic atom Floquet experiment..."
echo ""

# Example 1: Basic run with default parameters
echo "Example 1: Basic run (N=8, 80 steps)"
python -m quantacap.cli exotic-atom

echo ""
echo "Example 2: Larger system with more steps"
python -m quantacap.cli exotic-atom --N 8 --steps 120 --alpha 1.3

echo ""
echo "Example 3: Different drive parameters, no GIF"
python -m quantacap.cli exotic-atom \
  --N 7 \
  --steps 100 \
  --drive-amp 1.5 \
  --drive-freq 3.0 \
  --alpha 1.2 \
  --no-gif

echo ""
echo "Example 4: Custom couplings and output path"
python -m quantacap.cli exotic-atom \
  --N 6 \
  --steps 60 \
  --J-nn 1.5 \
  --J-lr 0.8 \
  --alpha 1.8 \
  --out artifacts/custom_exotic_atom.json

echo ""
echo "All examples completed!"
echo "Check artifacts/ directory for results:"
echo "  - exotic_atom_floquet.json (experiment data)"
echo "  - exotic_atom_density.png (final density visualization)"
echo "  - exotic_atom_evolution.gif (time evolution animation)"
