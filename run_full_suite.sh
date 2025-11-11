#!/usr/bin/env bash
#
# Quick runner for the full Atom3D simulation suite
#

set -e

export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

echo "============================================="
echo "  ATOM3D COMPLETE SIMULATION SUITE"
echo "============================================="
echo ""
echo "Running full pipeline:"
echo "  1. Hydrogen ground state (multi-resolution)"
echo "  2. Hydrogen excited states (2s, 2p)"
echo "  3. Helium Hartree model"
echo "  4. Field perturbations (Stark, Zeeman)"
echo "  5. Tomography reconstruction"
echo "  6. Dashboard generation"
echo ""
echo "Note: This requires numpy, scipy, matplotlib, imageio, scikit-image"
echo "      Install with: pip install numpy scipy matplotlib imageio scikit-image"
echo ""

python3 run_atom3d_suite.py --full "$@"
