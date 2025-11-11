#!/bin/bash
# Full demonstration of the 3D Atom Suite

echo "=== 3D Atom Modeling Suite Demo ==="
echo ""

# Ensure we're in venv
PYTHON=.venv/bin/python

echo "1. Hydrogen ground state (1s)"
$PYTHON -m atomsim.cli hyd-ground --N 64 --L 10 --steps 400 --dt 0.005 --out artifacts/atom3d/demo_h1s
echo ""

echo "2. Hydrogen 2p excited state"
$PYTHON -m atomsim.cli hyd-excited --N 64 --L 12 --steps 400 --dt 0.005 --nlm 2,1,0 --out artifacts/atom3d/demo_h2p
echo ""

echo "3. Helium ground state (mean-field Hartree)"
$PYTHON -m atomsim.cli he-ground --N 56 --L 9 --steps 400 --dt 0.005 --mix 0.5 --out artifacts/atom3d/demo_he
echo ""

echo "4. Stark field perturbation on 2p"
$PYTHON -m atomsim.cli hyd-field --mode stark --in artifacts/atom3d/demo_h2p --Ez 0.02 --steps 250 --out artifacts/atom3d/demo_h2p_stark
echo ""

echo "5. Fine structure corrections (1s)"
$PYTHON -m atomsim.cli hyd-fstructure --in artifacts/atom3d/demo_h1s --nlm 1,0,0 --Z 1 --out artifacts/atom3d/demo_h1s_fs
echo ""

echo "6. Synthetic tomography on 1s density"
$PYTHON -m atomsim.cli hyd-tomo --in artifacts/atom3d/demo_h1s --angles 120 --noise 0.01 --out artifacts/atom3d/demo_h1s_tomo
echo ""

echo "=== Demo complete! ==="
echo "Check artifacts/atom3d/demo_* directories for results"
