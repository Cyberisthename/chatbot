# Ownership Hardening Changelog

Date: 2026-05-12  
Scope: Remove non-owned wrapper/demo assets and disable demo/mock execution paths.

## What changed

### 1) Removed non-owned wrapper assets and docs
- Removed `ollama-jarvis-setup/` (third-party wrapper setup pack).
- Removed `docs/ollama/` documentation bundle tied to wrapper-based deployment.

### 2) Removed demo/mock execution paths
- Removed demo runtime artifacts:
  - `demo_multiverse/`
- Removed demo script bundle:
  - `demos/`
- Removed standalone demo entrypoints:
  - `scripts/demo_egf.py`
  - `scripts/demo_hypothesis_engine.py`
  - `scripts/demo_quantum_folding.py`
  - `scripts/run_protein_folding_demo.py`
  - `scripts/test_gradio_demo.py`
- Removed wrapper helper script:
  - `scripts/ollama_commands.sh`
- Removed demo adapters:
  - `quantacap/.adapters/*.demo*.json`

### 3) Preserved required research dataset without demo path coupling
- Migrated:
  - `demos/modern_research_observations.json`
  -> `research_data/modern_research_observations.json`

- Updated code references:
  - `src/research/crispr_falsification_sim.py`
  - `tests/test_hypothesis_engine.py`

### 4) Added automated ownership guard
- Added `scripts/ownership_guard_check.py`.
- Guard fails if forbidden demo/mock/wrapper paths reappear.

### 5) Updated user-facing wording in README
- Replaced demo framing with research-interface framing.

## Validation performed
- Guard check:
  - `python3 scripts/ownership_guard_check.py` -> **PASS**
- CRISPR simulation functional check after data-path migration:
  - `python3 scripts/simulate_crispr_falsification_experiment.py`
  - Output sanity: `n_trials=24`, `mean_relative_reduction=0.34955230317363223`, `support_fraction=1.0`

## Notes
- This pass removes demo/wrapper surfaces from the codebase to align with owner requirements for original, non-wrapper operation paths.
