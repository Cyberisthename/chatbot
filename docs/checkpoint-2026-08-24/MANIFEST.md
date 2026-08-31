# Checkpoint 2026-08-24 Manifest

## Committed to this snapshot
- `docs/checkpoint-2026-08-24/` — all key source files, theoretical reports, nitrogenase simulation, CHECKPOINT_SUMMARY.md
- `NITROGENASE_QUANTUM_SIMULATION_REPORT.md` and `nitrogenase_quantum_simulator.py` (via reports/)
- All theoretical framework documents from the team (physicist, scientific engineer)
- Updated FBSC compressor core (`compression_specialist.py`, `qvgpu_compressor.py`, `compressor_api.py`)
- Full interactive Quantum Compressor Lab (React + API route from port 3000 site)
- Owner seed and core state definitions

## Large artifacts referenced but NOT committed (see .gitignore + manifest)
- `jarvis_qvgpu_trained.npz` (~96 MB) — base trained state
- `qvgpu_compressed_state.npz` — current FBSC reconstruction
- `gguf` model file
- All adapter .json files in jarvis_v1_oracle/adapters/
- Visual and audio artifacts (can be regenerated from scripts)

These are available in the working directory and on the live port 3000 demo. The checkpoint focuses on source code, mathematics, and documentation so the owner has a clean, reviewable GitHub snapshot.

**Hash of this checkpoint**: Generated from owner seed (0.57721, 1.618034, 2.71828) + current git commit.

This fulfills the owner's request to "Checkpoint everything to GitHub".
