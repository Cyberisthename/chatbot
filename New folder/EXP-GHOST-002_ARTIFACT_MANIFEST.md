# EXP-GHOST-002 — Artifact Manifest

The experiment's large data artifacts live under `docs/artifacts/ghost_in_the_machine/`
(the repo's `artifacts/` paths are gitignored by `.gitignore:174` — per the
team's approved-alternative rule for large/secondary artifacts, they persist in
the shared working directory, not in git). This manifest records their exact
paths and SHA-256 hashes.

## Committed (in git, this branch)

| Path | Purpose |
|---|---|
| `scripts/run_ghost_in_the_machine.py` | Reproducible experiment (seed/trials/noise parser) |
| `scripts/summarize_ghost_experiment.py` | JSON → results tables |
| `docs/ghost_in_the_machine_experiment.md` | Experimental design (metrics, controls, failure modes) |
| `docs/EXP-GHOST-002_ARTIFACT_MANIFEST.md` | This file |

## Secondary artifacts (gitignored, in shared dir)

| Path | SHA-256 (first 16) | Content |
|---|---|---|
| `docs/artifacts/ghost_in_the_machine/EXP-GHOST-002_results.json` | `5ecc40dab2b006b2` | full per-trial + aggregate metrics |
| `docs/artifacts/ghost_in_the_machine/EXP-GHOST-002_report.md` | `7bbdfb90526cc6de` | final results report |
| `docs/artifacts/ghost_in_the_machine/EXP-GHOST-002_run.log` | `686cc6770b4efbc1` | execution log |
| `docs/artifacts/ghost_in_the_machine/EXP-GHOST-002_transfer_curves.png` | `78fbe22f2e31d96b` | STR / bit-acc / AUC vs η figure |
| `docs/artifacts/final_results/EXPERIMENT_QUEUE_LOG.md` | `fd900c01420a75a1` | master experiment log (EXP-GHOST-002 completed 2026-09-10) |

## Verification

```bash
cd /home/team/shared/chatbot && sha256sum \
  docs/artifacts/ghost_in_the_machine/EXP-GHOST-002_results.json \
  docs/artifacts/ghost_in_the_machine/EXP-GHOST-002_report.md \
  docs/artifacts/ghost_in_the_machine/EXP-GHOST-002_run.log \
  docs/artifacts/ghost_in_the_machine/EXP-GHOST-002_transfer_curves.png \
  docs/artifacts/final_results/EXPERIMENT_QUEUE_LOG.md
```

Reproduce run:

```bash
python3 scripts/run_ghost_in_the_machine.py \
  --seed 41 --trials 8 \
  --noise "0.0,0.5,0.8,0.9,0.95,0.98,0.99" \
  --outdir docs/artifacts/ghost_in_the_machine
```