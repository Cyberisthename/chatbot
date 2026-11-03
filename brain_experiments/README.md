# Brain experiments (snapshot 2026-11-03)

Research harnesses copied from the local flybrain worktree for review.
Classical / quantum-*inspired* methods only — no real qubits, no training weights, no connectome dumps.

## What's here
- `engine_v3/` — sparse OscRNN + STP / dual-tau / Dale / homeostasis ablation
- `dtm/` — Neuro-Vines (DTM) + MPS + SKF vs SET/RigL / PSO controls
- `sbqa/` — Simulated Bio-Quantum Annealing (WKB acceptance + Griffa mask)
- `regions/` — callosum, oscillation gating, human_wholebrain, and earlier region tests
- `hypothalamus/` — research notes for upcoming homeostat / drive / broadcast tests
- `jarvis_chat_rag/` — no-training JARVIS upgrades (CAD, pack router, `/define`)
- `jarvis_yzg_router/` — Y/Z/G Cognitive Router benchmark (separate from main model)
- `connectomes/human/*.md` — notes only (no scan data)

## Exclusions
No `.pt` checkpoints, no Institutional Books, no `knowledge/` dumps, no secrets,
no file >5 MB, no in-progress `ablation_runs.jsonl` / callosum eval dumps.
Tiny Griffa group npz (~31 KB) is included under `dtm/data/` and `sbqa/data/` (see DATA_LICENSE.md).

## Honest status (as of 2026-11-03 AM ET)
- CAD + router + `/define` helped JARVIS retrieval/gen hybrids; soft-exclusion / kNN / soup did not.
- engine_v3 STP early signal looks promising; full ablation still running.
- callosum_scaled: incomplete; sparse bridge beating real wiring so far at small N.
- SBQA / DTM Kaggle full multi-seed runs pending verdict.
- Y/Z/G router: experimental; no proof yet it beats the existing pack router.
