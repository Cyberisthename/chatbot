#!/usr/bin/env bash
# hardware/check_boundary.sh — CI gate for the hardware/ sandbox.
#
# The claim this script defends, in the words of the originality audit:
#   "The owned core imports no third-party quantum SDK, enforced by
#    hardware/check_boundary.sh in CI; device calibration code lives in hardware/,
#    is a leaf, and is not imported by the FBSC path."
#
# Exit 0 = boundary intact (prints "BOUNDARY OK").
# Exit 1 = boundary crossed; every violation is printed above the failure line.
#
# Run from anywhere: it locates the repo root from its own path.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || { echo "FAIL: cannot cd to repo root"; exit 1; }

EXCLUDES=(--exclude-dir=hardware --exclude-dir=legacy --exclude-dir=node_modules --exclude-dir=.git --exclude-dir=.venv --exclude-dir=venv)
SDK_PATTERN='qiskit|qiskit_ibm_runtime|qiskit_aer'
fail=0

echo "== hardware boundary check =="
echo "root: $ROOT"
echo

# ---------------------------------------------------------------------------
# 0. The gate must have teeth.
# A grep that can never match would pass forever. Prove the pattern still
# matches a string we know is there, inside the one directory that is allowed
# to contain it. If this fails, the gate is broken, not the repo.
# ---------------------------------------------------------------------------
if grep -q -E "$SDK_PATTERN" hardware/requirements.txt 2>/dev/null; then
  echo "[0/3] gate has teeth: pattern '$SDK_PATTERN' matches hardware/requirements.txt (allowed)"
else
  echo "[0/3] FAIL: the SDK pattern matches nothing in hardware/requirements.txt."
  echo "        Either the requirements file moved or the pattern rotted."
  echo "        A boundary check that cannot fire is not a boundary check."
  fail=1
fi
echo

# ---------------------------------------------------------------------------
# 1. No third-party quantum SDK outside hardware/ (and legacy/, which has its
#    own rule). Any Python file anywhere else that mentions qiskit is a breach.
# ---------------------------------------------------------------------------
echo "[1/3] no third-party quantum SDK outside hardware/ (must be empty)"
sdk_hits="$(grep -rn "${EXCLUDES[@]}" --include='*.py' -E "$SDK_PATTERN" . 2>/dev/null || true)"
if [ -n "$sdk_hits" ]; then
  echo "$sdk_hits"
  echo "  ^ FAIL: vendor SDK referenced outside hardware/."
  fail=1
else
  echo "      none"
fi
echo

# ---------------------------------------------------------------------------
# 2. hardware/ is a leaf: nothing outside it may import it.
# ---------------------------------------------------------------------------
echo "[2/3] nothing outside hardware/ imports hardware (must be empty)"
imp_hits="$(grep -rn "${EXCLUDES[@]}" --include='*.py' \
  -E '(^|[^A-Za-z0-9_.])(from[[:space:]]+hardware([.[:space:]]|$)|import[[:space:]]+hardware([.[:space:]]|$))' . 2>/dev/null || true)"
if [ -n "$imp_hits" ]; then
  echo "$imp_hits"
  echo "  ^ FAIL: the FBSC path (or anything else) imports hardware/."
  fail=1
else
  echo "      none"
fi
echo

# ---------------------------------------------------------------------------
# 3. The SDK appears in exactly one requirements file: hardware/requirements.txt.
#    A root or HF-ready requirements file listing qiskit would install a vendor
#    SDK into the owned-core environment.
# ---------------------------------------------------------------------------
echo "[3/3] no qiskit in owned-core requirement files (must be empty)"
req_hits="$(grep -n -E "$SDK_PATTERN" requirements.txt requirements-*.txt \
  jarvis_quantum_ai_hf_ready/requirements.txt 2>/dev/null || true)"
if [ -n "$req_hits" ]; then
  echo "$req_hits"
  echo "  ^ FAIL: a vendor SDK is declared as an owned-core dependency."
  fail=1
else
  echo "      none"
fi
echo

if [ "$fail" -ne 0 ]; then
  echo "BOUNDARY CROSSED — see hardware/README.md. The owned-core claim is false until this passes."
  exit 1
fi

echo "BOUNDARY OK — owned core free of third-party quantum SDKs; hardware/ is a leaf."
exit 0
