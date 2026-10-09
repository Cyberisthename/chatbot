# Nitrogenase seed_file_written hygiene — VERDICT: SAFE

Task 7a8d1b21-02b9-41b6-b86d-ff384ddbd0e1. Checked 2026-09-30 on chatboat origin/main (PR #147 merged).

## Files checked (specific paths)
1. `jarvis_quantum_ai_hf_ready/src/api/rnd_battery.py` — `_handle_nitrogenase` (lines ~507-540), `_handle_fbsc` (~270-289), `_handle_lhv_qudit`
2. `docs/artifacts/nitrogenase_qudit/nitrogenase_qudit_simulator.py` — `simulate_dimension` (seed write at lines 231-232), `main()` (report writers 357-360, 425-429)
3. `compression_specialist.py` — `save_compressed` (line ~325+; writes raw `owner_seed` to JSON)
4. `docs/artifacts/quantum_likeness/qudit_lhv_probe.py` + `likeness_probes.py` — all functions invoked by API (grep for open/write/save: empty)

## What is written and where
- `simulate_dimension` writes ONE file: `save_compressed(f"{OUT_DIR}/owner_quantum_seed_d{d}.json")` where OUT_DIR is a **module global read at call time**.
- The handler monkeypatches `nq.OUT_DIR = scratch` (under `/var/tmp`, from `_SCRATCH_ROOT` env default `/var/tmp`) BEFORE the call and restores `old_out` AFTER. It also `os.chdir(scratch)` for the duration.
- Report writers with **hardcoded** `/home/team/shared/nitrogenase_qudit_report.json` and `/home/team/shared/NITROGENASE_QUDIT_REPORT.md` live in `main()` — **never called by the API**.
- fbsc handler: calls `reconstruct()` only — 0 file writes (verified).
- lhv_qudit handler: calls compute-only functions — 0 file writes (verified).

## Fresh-run proof (handler-equivalent, numpy/scipy venv)
- seed file landed: `/var/tmp/rnd-hygiene/nq_hygiene_test/owner_quantum_seed_d4.json` (scratch, contains raw seed as expected but OUTSIDE repo)
- `OUT_DIR` restored after run → `/home/team/shared/nitrogenase_artifacts`
- scratch dir contents: only `owner_quantum_seed_d4.json`
- `find <repo> -name '*seed*' -newer <marker>` → **(none)** — no new seed files in repo tree
- raw seed values (`0.57721|1.618034|2.71828`) grep across repo JSON/DB → only untracked `New folder/` (see note)

## Verdict: SAFE (scratch-only)
All seed-related file writes from the nitrogenase/fbsc/lhv_qudit endpoints land ONLY under `/var/tmp` scratch. The directory is derived from the server-side `_SCRATCH_ROOT` (per-request unique `nq_<ms>` subdir), NOT from server cwd; OUT_DIR default is hardcoded but always overridden+restored. Response never includes the seed file path or raw seed — it reports `seed_file_written: bool` + `seed_digest` only.

## Pre-existing observation (NOT from this server, flagged for lead)
The shared checkout contains **untracked** `New folder/owner_quantum_seed_d{2,3,4}.json` + nitrogenase reports (timestamps 13:51, predate PR #147; NOT in origin/main per `git ls-tree`). They contain the PUBLIC source-code default seed constant `[0.57721, 1.618034, 2.71828]` (same constant as `OWNER_SEED` in committed source), not an operational secret. Recommend: delete or add to .gitignore; no code change needed in rnd_battery.py.
