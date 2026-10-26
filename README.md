# The owned quantum-inspired core (FBSC)

A fully original, deterministic **3-number generative formula** ("seed", owner-controlled) that
reconstructs complex amplitudes, topological braid positions, and geometric folding for
**256–1024+ effective qubits at MSE = 0** — in kilobytes, from three real numbers. The engine is
exact classical simulation of quantum-inspired structures (closed-form amplitudes, braid
topology, qudit phases). It is NOT quantum hardware, NOT an AGI, and NOT a sentience claim —
see [What this is NOT](#what-this-is-not).

Everything below is **measured** (tagged `[measured]`) and **re-verifiable by a fresh run on this
repo** — every claim row gives the exact command, and `docs/REPRODUCE.md` documents the
verified-baseline output for each one. If a number in this README is not in the table below, it
is not a verified claim of this release.

## Is this real? — yes, and here is the proof

Every number in the table was regenerated fresh (2026-10-09) from a clean checkout of
`origin/main`, with numpy + scipy. Runs marked **byte-exact** reproduced the committed artifacts
bit-for-bit; runs marked *number-exact* reproduced every numeric field (the JSON files embed a
wall-clock / platform stamp, so the files themselves cannot be byte-identical — the numbers are).

| # | Claim (all `[measured]` unless tagged) | Fresh-run proof (one command) | Reproduction result |
|---|---|---|---|
| 1 | Exact state reconstruction at **1024+ effective qubits from ≤3 reals**: MSE=0, ~16 KB (vs `2^1024·16 B ≈ 10^309 B` dense), compression ratio `8.778e+304`×, 0.2 s | `cd /tmp && PYTHONPATH=<repo> python3 - <<'EOF'` … `FractalBraidSeedCompressor(1024 …)` (see REPRODUCE §2) | **exact** (fresh measured, matches grid) |
| 2 | **NISQ collapse vs depth-independent core**: published-model fidelity falls below 50% at depth 1–6; owned core MSE=0 at any depth, 0.544 s, 64 GiB dense baseline avoided | `cd /tmp && PYTHONPATH=<repo> python3 <repo>/docs/validation/validate_benchmark.py` (clean cwd) | **number-exact** (2 volatile leaves: UTC stamp, elapsed) |
| 3 | **Formal error-bound proof numerically verified**: 22/22 grid cells MSE=0 for n≤1024 / d=2..4; dense crossover at n≥6; measure-zero cap shown (MSE vs Haar-random = 1.766 at n=8, honest) | `python3 <repo>/docs/validation/error_bound_verify.py` (clean cwd) | **number-exact** (24 volatile leaves: per-cell `time_s` + platform numpy stamp + UTC stamp; 0 numeric diffs) |
| 4 | **Topological fault tolerance**: Burau braid invariant distance **exactly 0** across 0→40 local-isotopy kinks while state fidelity collapses; every element-changing braid error detected (400/400) | `python3 <repo>/docs/artifacts/anyon/anyon_braid_sim.py` | **byte-exact** (sha256 identical) |
| 5 | **Crown jewel — statistics outside every local-hidden-variable model**: CGLMP/CHSH exhaustion at d=2 qubit-braid CHSH **2.8250** (n=2,3) / **2.0351** (n=4) all LP LHV-excluded (bound 2.0); new d=2 core I=**2.0305**, qudit-braid I=**2.3449**; MES controls excluded; d=3,4 honest negatives | `python3 <repo>/docs/artifacts/quantum_likeness/qudit_lhv_probe.py` | **number-exact** (1 doc-annotation leaf, not a measurement) |
| 6 | **Likeness probe battery** (entanglement, magic M2 2.238→2.892 pushed, α 0.367→0.750, XEB F, MPS/TT sweep D=16→64 wall, CHSH 2.825): surrogate probes track Hilbert-space structure | `python3 <repo>/docs/artifacts/quantum_likeness/likeness_probes.py` | **byte-exact** (0 leaf diffs, sha256 identical) |
| 7 | **Braid trainer vs GD — HONEST NEGATIVE**: GD wins differentiably (T0 0.000145, T1 0.137244); discrete-braid T2 tie; trainer wins only T3 (CHSH 2.664, LP-infeasible no-LHV certificate). Gate: `beats_seed=True`, `beats_gd=False` — no overclaim | `python3 <repo>/scripts/braid_vs_gd_benchmark.py` (repo root) | **number-exact** (1 volatile leaf: wall-clock) |
| 8 | **ResonanceMonitor F0 unification** (frequency measurement, not consciousness — see #NOT): clean 41.02 Hz carrier reads F0=41.0157 Hz via TTC spectral-radius fundamental; retired bit heuristic (38.0 Hz) kept only as labeled diagnostic; **21/21 tests pass** | `python3 -m unittest tests.test_f0_spectral_radius tests.test_resonance_noise_gate tests.test_eeg_engine -v` | **exact** (21/21, 0.045 s) |
| 9 | **Blind prediction (freeze-first, FeMo-co spin-state gap)**: freeze hash bit-exact, all five energies bit-exact, HIT at both prediction levels, honestly scoped as **one data point, NOT chemistry validation** | see `docs/validation/BLIND_PREDICTION.md` (freeze-first protocol; reference unlock) | reference artifact |
| 10 | **R&D battery over HTTP** (port 8777): fbsc MSE=0 at n=128 (2 KB, 10^36.12×), nitrogenase default-seed **d=4 barrier 100.7657, MSE=0**, likeness + lhv_qudit + trainer + resonance endpoints each carry a claims envelope; seed digest-only provenance (25 rows, `any_seed_stored=False`) | `docs/REPRODUCE.md §8` (start `python -m jarvis_quantum_ai_hf_ready.src.api.rnd_battery`, curl each endpoint) | **exact / measured** (smoke evidence committed) |

Interpretation (not measurement): the family is **not** a universal quantum computer — it cannot
represent generic states (≤3-parameter, measure-zero; the honest bound is in
`docs/validation/ERROR_BOUND_PROOF.md`). MSE=0 means *exactness against the generator's own
closed form*, not universality.

## What this is NOT (hard rules — every claim must survive them)

- **NOT AGI / sentience / consciousness.** `ResonanceMonitor` measures a frequency; its `is_sentient`
  alias returns a measurement flag, never a claim. No real EEG was measured (simulated signals only).
- **NOT quantum hardware.** Everything here is exact classical simulation of quantum-inspired
  structures. No anyons are physically realized (Burau braid math only); Bell-type statistics are
  shown *outside every local-hidden-variable model* — a classical mechanism, no hardware, no
  loophole-free experiment.
- **NOT chemistry / biology / medicine.** The nitrogenase / FeMo-co numbers are one blind-prediction
  data point inside the owned simulator — not experimental validation, no medical claims. Ownership
  audit gates list bio-quantum experiments as E1–E3 simulation-only (see
  `docs/artifacts/compliance/GOVERNANCE_GATE_BIO_QUANTUM.md`).
- **NOT decipherment.** The undeciphered-script second track maps *statistical structure* only;
  structure ≠ meaning, no decipherment claimed without a bilingual or identified language. The
  trifecta numbers (Linear A / Indus / Voynich) are documented in `docs/artifacts/factcheck/FACTCHECK.md`
  and are **not** in this release's reproducibility table (the analysis scripts were destroyed; raw
  corpora are not in-repo). The Indus method + corpus remain committed
  (`docs/artifacts/indus/INDUS_H1H4_CROSSVALIDATE.md`).
- **NOT best-at-everything.** Braids lose to GD on differentiable tasks (row 7). Likeness probes hit
  honest walls (not a 2-design, no CUE repulsion, ≤3-param measure-zero). Several LHV cells are
  honest negatives (d=3,4 at found settings).
- **NOT "no external code, ever".** The owned core under `src/` is custom numpy/stdlib, MIT-licensed,
  with no proprietary wrappers and no third-party quantum or LLM SDKs (audit:
  `docs/artifacts/compliance/ORIGINALITY_AUDIT.md`). numpy/scipy are dependencies; fastapi/uvicorn
  are dependencies of the optional R&D server only. A set of **legacy root scripts sit OUTSIDE the
  owned core**: `train_jarvis.py`, `train_llm.py`, `convert_to_gguf.py` (import torch + transformers)
  and `jarvis_historical_training_pipeline.py`, `jarvis_v1_quantum_oracle_train.py` (import
  HuggingFace `datasets`, auto-installed at runtime). The boundary statement is
  `docs/artifacts/legal/LEGACY_DEPENDENCIES_NOTE.md`, which lists all five root scripts and explicitly
  names the legacy `*_MISSION_COMPLETE.md` completion claims (e.g. a "947GB historical books
  dataset", Hugging Face deployment) as legacy claims **outside the owned core and NOT used as
  evidence in this release**.

## Verifier's gotchas (found by fresh runs — read before you run)

1. **The npz-in-cwd landmine.** `jarvis_qvgpu_trained.npz` (14.7 MB, tracked in this repo) is loaded
   by `compression_specialist._load_qvgpu_base()` **from the current working directory** when
   present, and changes the owned amplitude base. **A repo-root run of the validation scripts
   FAILS its asserts** (measured: d=3 barrier reads 122.3022 vs approved 129.9529). All rows 1–3
   above must run from a **clean cwd** (`cd /tmp` then `PYTHONPATH=<repo>`), exactly as documented in
   REPRODUCE.md. The R&D API does this automatically (per-request scratch chdir — that is why its
   numbers match the approved set).
2. **Hardcoded absolute outputs.** `likeness_probes.py` writes to
   `/home/team/shared/quantum_likeness/likeness_results.json` and the nitrogenase standalone writes
   to `/home/team/shared/nitrogenase_artifacts/` — both committed warts. REPRODUCE.md gives a
   one-line local patch (or use the R&D API endpoint, which is scratch-safe).
3. **JSON artifacts embed run stamps.** `validate_benchmark.json`, `error_bound_verify.json`,
   `braid_vs_gd_results.json` embed wall-clock / platform fields by design, so a fresh run's file is
   not byte-identical to the committed one — the *numbers* are (see table).

## Repository map

- `docs/REPRODUCE.md` — one-command-per-claim reproduction guide with expected output
- `docs/RELEASE_NOTES.md` — v0.1.0-evidence release notes (milestones + honest scope)
- `docs/validation/` — Validation Track 1/3 + 2/3 + 3/3 (`VALIDATION_BENCHMARK.md`,
  `ERROR_BOUND_PROOF.md`, `BLIND_PREDICTION.md`, `validate_benchmark.py`, `error_bound_verify.py`)
- `docs/artifacts/` — evidence: `anyon/`, `quantum_likeness/`, `braid_vs_gd/`,
  `nitrogenase_qudit/`, `factcheck/`, `compliance/`, `indus/`, `jarvis/trainer/`
- `docs/artifacts/compliance/` — `ORIGINALITY_AUDIT.md`, `GOVERNANCE_GATE_BIO_QUANTUM.md`
- `docs/team/WORKFLOW.md` — team delivery rules (push-every-branch, never `git clean` in shared
  checkout, RESTORE_CHECKOUT.sh recovery)
- `jarvis_quantum_ai_hf_ready/src/api/rnd_battery.py` — the R&D battery FastAPI server (port 8777)
- `compression_specialist.py` — the FBSC core (owner-controlled 3-number seed)
- `scripts/braid_vs_gd_benchmark.py` — braid-space trainer harness (honest beats_seed/beats_gd gate)

## License & custody

MIT (see LICENSE). The 3-number seed stays owner-controlled; the R&D server persists **seed digests
only** (`seed_stored=false`, verified in code and data). No real human health data, no hardware
deployment — simulation ceiling only.