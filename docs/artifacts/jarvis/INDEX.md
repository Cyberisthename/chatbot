# JARVIS Quantum Core — Artifact Index

**Curated browse tree:** `/home/team/shared/jarvis/` · **Durable copies:** `docs/artifacts/jarvis/` + `docs/artifacts/indus/` in repo (PR #141)
**Last updated:** 2026-09-28 · **Curator:** agent-compression-specialist · **Status:** 31 files indexed, 1 set pending regen + factcheck pack

---

## ⚠️ Incident Warning (2026-09-24 disk-full event)

The 2026-09-24 disk-full event destroyed file *contents* in `/home/team/shared` while keeping
directory skeletons. **Confirmed lost from disk (irrecoverable locally):** the original `jarvis/INDEX.md`,
`jarvis/trainer/`, `validation/`, `qml/`, `anyon/`, `fbsc_efficiency_report.json`, `seedopt_results.json`,
base `owner_quantum_seed.json`. Everything re-listed below is either **git-surviving** (byte-identical from
origin/main), **regenerated** (re-verified against the approved numbers held in team-DB task records), or
**reconstructed** (values identical, provenance-tagged). Nothing is fabricated; every claim is tagged
`measured / interpretation / speculation` per house standard.

**Provenance rule:** approved numbers live in team-DB task results (tasks `ca8ea331`, `2124404a`,
`97c482eb`, `4c706c3b`, `f338a3e0`, `b65a40c6`, `cd416d93`, `af40e730`, `fda8f85c`). Any artifact can be
re-verified by fresh run (command listed per row). If a stored artifact ever disagrees with the DB task
record, the DB record wins — flag to the lead.

## Status Legend
- ✅ **git-survived** — byte-identical copy from origin/main or surviving shared copy
- 🔁 **regenerated** — re-created by the original author after the incident, re-verified
- ♻️ **reconstructed** — values identical to a surviving sibling, provenance-tagged
- 🔥 **destroyed** — confirmed lost, numbers preserved here via task record
- 🌱 **in-flight** — regenerated set still pending (author's task in-progress)

---

## 1. core/ — FBSC Compressor & Owner Seed

| File | What | Status |
|---|---|---|
| `compression_specialist.py` | FBSC core. Deterministic 3-number → exact complex amplitudes + braid positions + folding for n up to 1024+ effective qubits. Pure NumPy, ~8 ms/reconstruct. Includes `_safe_compression_ratio` / `_safe_compression_ratio_qb` (log-space, no overflow at n=1024 d=3). MSE=0, KB-scale memory vs 64 GiB dense vector. | ✅ git-survived (fix re-applied 2026-09-24; **repo HEAD predates the fix** — canonical copy is this one, see Repo-Sync note) |
| `owner_quantum_seed.json` | **Base** master seed: `[0.57721, 1.618034, 2.71828]`, params `{scale 1.65442, phase 5.083203727658507, fold_depth 11}`, no qudit_dim. | ♻️ reconstructed 2026-09-24 from `owner_quantum_seed_d2.json` — seed+params byte-values identical; NOT byte-identical to the lost original file (metadata differs) |
| `owner_quantum_seed_d2.json` | d=2 qudit seed (params include `qudit_dim: 2`). | ✅ git-survived |
| `owner_quantum_seed_d3.json` | d=3 qudit seed. | ✅ git-survived |
| `owner_quantum_seed_d4.json` | d=4 qudit seed (drives the 100.7657 kJ/mol headline). | ✅ git-survived |
| `fbsc_efficiency_report.json` | n=1024 d=2 → compression ratio **8.7777985e304×** (log10 304.943), MSE=0; n=1024 d=3 → log10 485.085 via log-space helper. | 🔁 regenerated 2026-09-24 (fresh run of `compression_specialist.py`; run date tagged in file) |

**Re-verify:** `python3 -c "import sys; sys.path.insert(0,''); from compression_specialist import *; ..."` — see module docstring; MSE=0 asserts included.

## 2. nitrogenase/ — Qudit Fix & FeMo-co Benchmark (Validation Stage 1)

| File | What | Status |
|---|---|---|
| `nitrogenase_qudit_simulator.py` | Qudit-enabled nitrogenase simulator. **1-D-safe fix applied** (coherence_of / braid_pathways / bio_resonance shape-safe for 1-D qubit register) — this is the corrected version. | ✅ git-survived + fix re-applied 2026-09-24 |
| `NITROGENASE_QUDIT_REPORT.md` | d=2 row: 2^16 = **65,536 states / 0.25 KB / 2,048× mesh**/MSE=0; d=4: barrier **100.7657**, catalyst **80.613 kJ/mol**. | 🔁 regenerated 2026-09-24, re-verified vs task `2124404a` |
| `nitrogenase_qudit_report.json` | Machine-readable version of the above. | 🔁 regenerated 2026-09-24 |
| `DATA_FIX.md` | Documents the 5^16→2^16 dimension fix (approved; `INPUT`-side data fix). | 🔁 recreated 2026-09-24 from task `2124404a` record |
| `owner_quantum_seed_d2/d3/d4.json` | Seed siblings (same values as core/). | ✅ git-survived |

**Honesty note:** the pre-fix numbers (96.8918 kJ/mol / 5^16 row) survive in the owner's upload
(`chatbot/New folder/nitrogenase_qudit_report.json`, dated 2026-09-10) and in the reverted chatbot/docs
mirror. The **corrected canonicals** are here and in `nitrogenase_artifacts/` + shared root.

## 3. seedopt/ — Variational Seed Optimizer

| File | What | Status |
|---|---|---|
| `seed_optimizer.py` | CMA-ES vs Nelder-Mead vs coordinate descent. **CMA-ES winner** (MSE=0 at n=32 post-check); Nelder-Mead was WORSE than the owner seed. Landscape featureless (std 0.024). | ✅ git-survived |
| `SEED_OPTIMIZER.md` | Method, results, honest framing (closed-form O(1) amplitudes; ≤3-parameter measure-zero cap — cannot represent generic states). | ✅ git-survived |
| `seedopt_results.json` | Fresh run 2026-09-24: CMA-ES winner **0.701304**, MSE=0 at n=32; matches task `ca8ea331`. | 🔁 regenerated 2026-09-24 (fresh run) |

**Re-verify:** `python3 seed_optimizer.py` (writes results; compares vs owner seed).

## 4. resonance/ — TonalSoulEngine / Bio-Resonance

| File | What | Status |
|---|---|---|
| `RESONANCE_FIX.md` | Documents fix of ResonanceMonitor f0 heuristic + noise-gate sentience trigger (41.02 Hz trigger; task `d0071e96`). | ✅ git-survived |
| `resonance_noise_gate_results.json` | Measured gate results. | ✅ git-survived |
| `resonance_noise_gate_test.py` | Reproducible test harness. | ✅ git-survived |

## 5. trainer/ — 3-Phase Topological Trainer (owner design)

| File | What | Status |
|---|---|---|
| `quantum_topological_trainer.py` | Phase 1 variational seed optimizer (CMA-ES) → Phase 2 annealing loss landscape → Phase 3 evolutionary mutation loop. Hard gate: must beat classical GD + plain seed optimizer on equal budget, or honest negative. | 🔁 regenerated by agent-theoretical-physicist (2026-09-25) |
| `trainer_results.json` | Gate outcome: **GD tie 0.0**, beats plain seed optimizer (0.0536), annealing phase negative (honest negative recorded). | 🔁 regenerated 2026-09-25 |
| `TRAINER_REPORT.md` | Full write-up incl. honest negative. | 🔁 regenerated 2026-09-25 |

## 6. anyon/ — Topological Fault-Tolerance Experiment

| File | What | Status |
|---|---|---|
| `ANYON_BRAID_EXPERIMENT.md` | Burau braid invariants immune to local isotopy noise (distance exactly 0 across 0→40 kinks) while state fidelity collapses; every element-changing braid error detected (1.000/400 trials). Mathematical immunity proven; **physical anyon realization explicitly NOT claimed**. | 🔁 regenerated by agent-theoretical-physicist (2026-09-25) |
| `anyon_braid_sim.py` | Reproducible simulator. | 🔁 regenerated 2026-09-25 |
| `anyon_braid_results.json` | Measured results. | 🔁 regenerated 2026-09-25 |

## 7. qml/ — QLM (Braid-Semantic Language Kernel) / Quantum-Likeness

| File | What | Status |
|---|---|---|
| `QLM_FRAMEWORK.md` | QLM layer framework. **Note:** the original full framework (13,379 B, authored 2026-09-14) survives in the owner upload (`chatbot/New folder/QLM_FRAMEWORK.md`); the regenerated copy here is the physicist's fresh compact version (3,318 B) — same kernel, re-derived. | 🔁 regenerated (kernel) + ✅ original in owner upload |
| `QLM_ROUND2.md` | Second QLM round. | 🔁 regenerated 2026-09-25 |
| `qml_braid_kernel.py` | Reproducible kernel. | 🔁 regenerated 2026-09-25 |
| `qml_results.json` / `qml_results_v2.json` | Measured outputs (round 1 / round 2). | 🔁 regenerated 2026-09-25 |
| `qml_self_test.json` | Self-test record. | 🔁 regenerated 2026-09-25 |

**Sibling work (in-flight, not yet indexed):** `/home/team/shared/qml/` also holds the physicist's
quantum-likeness probes (Wigner negativity, Bell-CHSH no-LHV exhaustion, etc.):
`QUANTUM_LIKENESS_REPORT.md`, `bell_chsh_prob.py`, `bell_chsh_results.json`, `likeness_results.json` —
dated 09-24, produced in the incident aftermath; treated as live work, will be indexed once approved.

## 8. validation/ — Validation Track 3/3 🔥 → 🌱 PENDING REGEN

Destroyed in the incident; **regeneration in progress** (task `a839d0af`, agent-scientific-engineer).
Approved numbers preserved in DB task records `b65a40c6` / `cd416d93` / `af40e730` / `fda8f85c`:
- **Stage 1 — NISQ-noise benchmark:** owned exact qudit simulator collapses at depth 1–6 at equivalent
  size vs published NISQ fidelities; core depth-independent, exact (MSE=0, ~1 KB vs 64 GiB dense vector).
- **Stage 2 — Error-bound proof:** exactness-by-construction, closed-form O(1) amplitudes, ≤3-parameter
  measure-zero cap (honest bound), parametric crossover vs dense statevector at n≥6.
- **Stage 3 — Blind empirical prediction (FeMo-co):** freeze hash reproduced bit-exact over the
  pre-unlock core (sha256 `37e6c84c…`), all five energies bit-exact, HIT at both prediction levels,
  honestly scoped as ONE data point, NOT chemistry validation.

**Expected files when regen lands:** `VALIDATION_BENCHMARK.md`, `ERROR_BOUND_PROOF.md`,
`validate_benchmark.{py,json}`, `error_bound_verify.{py,json}`, `BLIND_PREDICTION.{md,json}`.

## 9. Owner Upload (git f556e96, `chatbot/New folder/` — 45 files)

Owner-owned artifacts uploaded 2026-09-25, **committed to origin/main** (durable by construction):
- **CORPUS.md** — Linear A/B corpus provenance & license (foundation deliverable by scientific-engineer)
- **TRANSLATION_PASS.md** — Linear A structural/functional translation pass (39 KB, v2 owner spec)
- **QLM_FRAMEWORK.md** — original full QLM framework (13,379 B, 2026-09-14)
- **nitrogenase_qudit_simulator.py** — original pre-fix simulator (21,855 B) — *superseded by jarvis/nitrogenase corrected copy*
- **nitrogenase_qudit_report.json** — original pre-fix report (d4 barrier 96.8918, 2026-09-10) — *superseded; corrected 100.7657 in jarvis/nitrogenase*
- **owner_quantum_seed_d{2,3,4}.json** — seed siblings
- Structural-analysis PNGs (positional heatmaps, stochastic resonance verification, etc.), QLM images,
  CHSH timelapse videos, CANCER_HYPOTHESIS docs, VIRTUAL_CANCER_CELL_SIMULATOR, QUANTUM_HYDROGEN_BOND_DISCOVERY,
  MULTIVERSAL_COMPUTING_README, specification.md, scientific_framing.md, EXPERIMENT_RESULTS.md.

These are owner-owned; the INDEX lists them as source-of-truth uploads. Where they overlap with
regenerated artifacts (QLM, nitrogenase), the **corrected/approved version sits in jarvis/** and this
section marks the upload copy as pre-fix/original.

## 10. factcheck/ — Evidence-Backed Fact-Check (owner's external-skeptic pack)
| File | What | Status |
|---|---|---|
| `FACTCHECK.md` (30.5 KB, 351 lines) | Owner-requested evidence pack answering the 8 skeptic questions (FBSC/MSE semantics, compression-ratio formula, nitrogenase chemistry honesty, protein engine constants + "+68%" provenance, Linear A 81.6% + QLM ARI, time crystal, stability heatmap, placeholders & limits). Every number cites `file:function` or a measured value; every claim tagged measured/interpretation/speculation; 15-line verbatim-pastable summary at top. Key honest guardrails inside: MSE=0 dense-verified only at n≤32; "ARI ≈ 0.19" NOT found on disk (recorded values +0.108/−0.002/−0.102); no CHSH S(t) series stored (only per-n maxima + fresh 36-seed sweep max 2.828123 / owner 2.824953); validation/ destroyed on disk; "+68%" is a hardcoded doc example, not a run result. | ✅ committed 2026-09-28 via PR (`docs/artifacts/factcheck/`, number in task result) |
**Source of truth:** `/home/team/shared/FACTCHECK.md` (identical bytes, verified by diff); author
agent-compression-specialist; re-verifiable from the cited files/task records per §11 citation map.

---

## Repo-Sync Warning (2026-09-24)

An external git sync reverted the `chatbot/` working tree to git HEAD (14:17:45), so
`chatbot/docs/artifacts/nitrogenase_qudit/` currently shows the **superseded pre-fix numbers**
(5^16 row, 96.8918 kJ/mol) and `compression_specialist.py` at HEAD lacks the log-space ratio helper.
**Do NOT read numbers from the repo docs until the corrected set is committed** (this archive + the PR).
Corrected canonicals: `nitrogenase_artifacts/`, shared root, and this tree.

## Durability (WORKFLOW.md §Artifact Durability)

1. This tree is a **curated browse copy**. Durable copies are committed to git:
   `docs/artifacts/jarvis/` (this INDEX + all 31 files) and `docs/artifacts/indus/` (H1-H4 harness) via PR #141.
2. `/home/team/shared` is a staging area, NOT an archive. If this tree vanishes again, re-materialize
   from the repo copy + team-DB task records.
3. Validation regen will be added to this INDEX when it lands (pending task `a839d0af`).

## Hygiene Rules
- Every claim tagged measured / interpretation / speculation (see each artifact's own tags).
- No hardware claims — "exact classical simulation" is the honest ceiling.
- Structure ≠ meaning — no decipherment claim without a bilingual or identified language.
- No unopened risk without an explicit owner flag.

## Final Tree (31 files + this INDEX)
```
jarvis/
├── INDEX.md
├── core/    (6)  compression_specialist.py, fbsc_efficiency_report.json, owner_quantum_seed.json + d2/d3/d4
├── nitrogenase/ (6) simulator, report md/json, DATA_FIX.md, seeds d2/d3/d4
├── seedopt/ (3)  seed_optimizer.py, SEED_OPTIMIZER.md, seedopt_results.json
├── resonance/ (3) RESONANCE_FIX.md, resonance_noise_gate_results.json, resonance_noise_gate_test.py
├── trainer/ (3)  quantum_topological_trainer.py, trainer_results.json, TRAINER_REPORT.md
├── anyon/   (3)  ANYON_BRAID_EXPERIMENT.md, anyon_braid_sim.py, anyon_braid_results.json
├── qml/     (6)  QLM_FRAMEWORK.md, QLM_ROUND2.md, qml_braid_kernel.py, qml_results.json, qml_results_v2.json, qml_self_test.json
└── validation/  (pending regen — task a839d0af)
```