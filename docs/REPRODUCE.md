# REPRODUCE.md — one command per claim, from a clean clone

Every claim in the release is re-verifiable by a fresh run on a clean checkout of `origin/main`.
This file gives the exact commands, the expected output, and the measured runtime, for each one.

All commands were run and verified **fresh on 2026-10-09** on a detached copy of `origin/main`
(`commit e7ca4c3`) with the venv described below. Where a command must run from a particular
working directory, that is stated — **the #1 gotcha is the npz-in-cwd trap, so read §0 first.**

## 0. Environment and the ONE gotcha that will otherwise bite you

```bash
git clone <repo-url> chatbot && cd chatbot
python3 -m venv .venv && . .venv/bin/activate
pip install numpy scipy            # ~30 MB; fastapi+uvicorn only for §8 (server)
```

**Gotcha (measured):** `compression_specialist._load_qvgpu_base()` loads
`jarvis_qvgpu_trained.npz` — which is **tracked at the repo root** — from the *current working
directory* when present, and uses it as the amplitude base. A **repo-root run** of the validation
scripts therefore **FAILS** (measured: `validate_benchmark.py` assert `(3, 122.3022, 129.9529)`).
The approved numbers assume the owner fallback base, i.e. **a clean cwd**. All owned-core
reproductions in §2–§3 below must therefore run from a directory that does **not** contain that
npz, e.g.:

```bash
mkdir -p /tmp/repro && cd /tmp/repro        # npz NOT here
export PYTHONPATH=/path/to/chatbot          # repo root (module imports)
python3 /path/to/chatbot/docs/validation/error_bound_verify.py
```

The R&D battery API (§8) does this for you automatically (per-request `os.chdir` to a scratch
dir); its numbers therefore match the approved set regardless of where the server was started.

**Also:** two artifact scripts write to hardcoded absolute paths
(`likeness_probes.py` → `/home/team/shared/quantum_likeness/likeness_results.json`,
nitrogenase simulator → `/home/team/shared/nitrogenase_artifacts/`). §6 gives a one-line local
patch; the API endpoints (§8) are scratch-safe and need no patch.

**Runtimes** below are wall-clock on this box (python 3.12, numpy 2.5.4, scipy 1.18.1). JIT/no-GPU:
everything is plain numpy.

## 1. Quibit-scale attack — exact reconstruction at 1024+ effective qubits (row 1)

Claim: ≤3 real numbers (owner seed `(0.57721, 1.618034, 2.71828)`) exactly reconstruct
1024-qubit amplitudes; MSE=0; ~16 KB stored; dense ambient would be `2^1024·16 B ≈ 10^309 B`.

```bash
cd /tmp/repro && export PYTHONPATH=/path/to/chatbot
python3 - <<'EOF'
import compression_specialist as cs
c = cs.FractalBraidSeedCompressor(n_effective_qubits=1024, seed=(0.57721, 1.618034, 2.71828))
a, p, m = c.reconstruct()
print(m["effective_qubits"], "qubits | MSE", m["reconstruction_mse"],
      "| mem", m["memory_kb"], "KB | ratio %.3e" % m["compression_ratio"])
EOF
```

Expected output (measured): `1024 qubits | MSE 0.0 | mem 16.0 KB | ratio 8.778e+304`
Runtime: **0.20 s**. The full grid (d=2..4, n=8..1024) is verified by §3.

## 2. Validation Track Stage 1 — NISQ benchmark vs the owned simulator (row 2)

Claim: standard NISQ noise model collapses fidelity below 50% within depth 1–6 at n=8..54, while
the owned simulator is depth-independent (MSE=0) and exact — reproducing the approved headline.

```bash
cd /tmp/repro && export PYTHONPATH=/path/to/chatbot
python3 /path/to/chatbot/docs/validation/validate_benchmark.py
```

Expected output (last lines, measured):
```
 typical      F10=0.053 d50=1 d1%=18 crossover(8/32/54)=7/1/1
 best         F10=0.376 d50=6 d1%=66 crossover(8/32/54)=25/4/1
 Owned sim d=3..6: {'3': 129.9529, '4': 100.7657, '5': 76.0663, '6': 50.14} (all match approved <=0.001, MSE={'3': 0.0, '4': 0.0, '5': 0.0, '6': 0.0})
 d=2 frozen approved value: 144.7362 (provenance in JSON)
 d=4 Hilbert dim = 4^16 = 2^32: True | dense d=2 n=32 = 64.0 GiB | owned elapsed 0.54s
Wrote validate_benchmark.json
```
Runtime: **0.7 s**. Exit code 0 = pass (the script asserts `|Δ| ≤ 0.001` against approved
values). The JSON is refreshed with a new UTC stamp + wall-clock; all *numbers* match the
committed artifact — file is not byte-identical **by design** (2 volatile fields).

## 3. Validation Track Stage 2 — error-bound proof, numerically verified (row 3)

Claim: exactness-by-construction, closed-form O(1) amplitudes, ≤3-parameter measure-zero cap,
parametric crossover vs dense statevector at n≥6.

```bash
cd /tmp/repro && export PYTHONPATH=/path/to/chatbot
python3 /path/to/chatbot/docs/validation/error_bound_verify.py
```

Expected output (last lines, measured):
```
  d=4 n=64    stored=    4096 B  dense-ambient=     5.4e+39 B  MSE=0.00e+00  t=0.2000s
  d=4 n=128   stored=    8192 B  dense-ambient=     1.9e+78 B  MSE=0.00e+00  t=0.2057s
  d=4 n=256   stored=   16384 B  dense-ambient=    2.1e+155 B  MSE=0.00e+00  t=0.2169s
  d=4 n=512   stored=   32768 B  dense-ambient=   ~10^309.8 B  MSE=0.00e+00  t=0.2464s
Analytic crossover: dense 2^n·16B >= 1KB artifact at n >= 6; n=32 -> 64 GiB vs 1.25 KB (ratio 53687091)
RAM crossovers: {'64GB-laptop': 32, '1TB-node': 36, '4TB-hpc': 38}
Cap test (n=8): MSE vs generator-own state = 0.000e+00 | MSE vs Haar-random = 1.766e+00 (searched 400 seeds)
```
Runtime: **5.2 s**. All 22 grid cells MSE=0 (fresh vs committed: **0 numeric diffs**; 24 volatile
fields: per-cell `time_s`, `platform.numpy` version stamp, `generated`). The cap test
(MSE vs Haar-random = 1.766, not 0) is the *honest, reported* measure-zero bound — this family
cannot represent generic states.

## 4. Topological fault tolerance (row 4)

```bash
cd /path/to/chatbot && python3 docs/artifacts/anyon/anyon_braid_sim.py
```
Expected (measured): isotopy-invariant distance **0.000000** across 0→40 kinks; outage flips
`400/400 = 1.000`; collapse depths `58 / 124 / 408` gates for early/typical/best NISQ.
Runtime: **0.3 s**. **Byte-exact**: regenerated `anyon_braid_results.json` sha256 =
`1c337eb0…527a5` (identical to committed).

## 5. Crown jewel — no-local-hidden-variable exhaustion probe (row 5)

```bash
cd /tmp/repro && export PYTHONPATH=/path/to/chatbot
python3 /path/to/chatbot/docs/artifacts/quantum_likeness/qudit_lhv_probe.py
```
Expected (measured, d=4 tail): all FBSC-core / FBSC-braid cells report `LHV_exists=True`
(honest negatives at d=3,4 for found settings); the LP-excluded cells are the d=2 rows:
CHSH 2.8250 / 2.8250 (n=2,3), 2.0351 (n=4); new d=2 core I=2.0305; qudit-braid I=2.3449;
MES controls I=2.7851 / 2.3321 excluded; product controls ~1.9 not. Runtime: **24 s**.
Fresh vs committed: **1 leaf differs** — the committed JSON carries an added *documentation*
annotation `meta.d3_n2_artifact_note` (the d=3,n=2 solver-dependent non-facet observation,
I=1.8117 < LHV bound 2, explicitly never claimed as a violation). **All measurements are
identical.**

## 6. Likeness probe battery (row 6)

```bash
# local patch for the hardcoded absolute OUT_PATH (the script writes its results to
# /home/team/shared/quantum_likeness/... as committed — patch it to write locally):
cp /path/to/chatbot/docs/artifacts/quantum_likeness/likeness_probes.py /tmp/repro/lp.py
sed -i "s#/home/team/shared/quantum_likeness/likeness_results.json#/tmp/repro/likeness_results.json#" /tmp/repro/lp.py
cd /tmp/repro && export PYTHONPATH=/path/to/chatbot
python3 /tmp/repro/lp.py
```
Expected (measured): entropy/negativity/magic/XEB all numeric; CHSH 2.825 (n=2,3) / 2.0351 (n=4);
MPS/TT sweep reaches the **D=16→64 bond-dimension wall** (reported honestly — this restricts the
sweep); push phase: owner seed score=0.658 (α=0.367, M2=2.238, F=8.692) → pushed seed
score=0.907 (α=0.750, M2=2.892, F=3.662). Runtime: ~2–4 min. **Byte-exact**: regenerated
`likeness_results.json` sha256 = `7f9d3b00…334e2` (identical to committed; 0 leaf diffs).

## 7. Braid trainer vs GD — honest negatives (row 7)

```bash
cd /path/to/chatbot && python3 scripts/braid_vs_gd_benchmark.py
```
Expected (measured, equal budget 800):
```
T0_smooth_regression           0.000145     0.292349     0.250391  beats seedopt (loses GD)
T1_adapter_multitarget         0.137244     0.210738     0.188340  beats seedopt (loses GD)
T2_invariant_matching               N/A     0.000000     0.000000  tie
T3_nolhv_chsh                       N/A    -1.956037    -2.663916  trainer wins
```
Runtime: **5–6 s** (± wall clock; the harness's `wall_clock_seconds` is the only field that
differs from the committed JSON). Honest gate: `beats_seed=True`, `beats_gd=False`. The T3 win is
a no-LHV certificate (CHSH 2.664 via LP exclusion), not a claim that braid training beats GD —
on differentiable tasks, GD wins.

## 8. R&D battery API (port 8777) (row 10)

```bash
cd /path/to/chatbot && pip install fastapi uvicorn   # server-only deps
python3 -m jarvis_quantum_ai_hf_ready.src.api.rnd_battery &   # uvicorn on 127.0.0.1:8777
curl -s http://127.0.0.1:8777/api/rnd/health
```
Then (all measured 2026-10-09):
- `POST /api/rnd/fbsc {"seed":[0.57721,1.618034,2.71828],"n_qubits":128}` → `mse=0.0`,
  `compression_ratio_log10=36.12`, `memory_kb=2.0`
- `POST /api/rnd/nitrogenase {}` (seed omitted → owner default, clean-cwd scratch) →
  **d=4 barrier `100.7657`, MSE=0** (the approved number; a supplied truncated seed gives a
  different value — handler uses OWNER_SEED only when none is provided)
- `POST /api/rnd/lhv_qudit {"seed":[...]}` → d=2 sweep I=2.5477 / 2.4237 / 2.5984, all
  LP LHV-EXCLUDED vs bound 2.0
- `POST /api/rnd/trainer` → `beats_gd=false`, `beats_seed=true` (honest gate)
- `POST /api/rnd/resonance` → `f0_measured_hz≈41.0` at 41.02 Hz target, SNR ≈54.6 dB,
  claims envelope: *spectral event, not consciousness*
- `GET /api/rnd/runs` → 25 provenance rows, `any_seed_stored=false` (digest only)

Seed hygiene: raw seeds are never persisted (SQLite stores digests; verified in code and in the
data). The nitrogenase handler patches the simulator's OUT_DIR to a per-request scratch dir and
restores it in `finally` — verified (`docs/artifacts/nitrogenase_qudit/nitrogenase_seed_hygiene_verdict.md`).

## 9. ResonanceMonitor F0 tests (row 8)

```bash
cd /path/to/chatbot && python3 -m unittest tests.test_f0_spectral_radius tests.test_resonance_noise_gate tests.test_eeg_engine -v
```
Expected (measured): **Ran 21 tests ... OK** (0.045 s) — 5 F0 + 14 noise-gate + 2 EEG.
Covers: clean 41.02 Hz carrier reads F0≈41.0157 Hz (the retired bit heuristic read 38.0 Hz and
could never reach its own 40 Hz threshold); noise/impulse/mains inputs do **not** trigger;
`f0=None` for bits-only callers; `is_sentient` is an alias to the measurement flag, never a claim.

## 10. Validation Track Stage 3 — blind prediction (row 9; reference, freeze-first)

Not a rerunnable numeric script — it is a **freeze-first protocol**: the freeze hash, pre-unlock
core predictions, and post-unlock energies are committed. Reproduce the *claim reading* by
following `docs/validation/BLIND_PREDICTION.md`: freeze hash reproduced bit-exact over the
pre-unlock core; all five energies bit-exact; HIT at both prediction levels within pre-frozen
bands. Scope: **one data point, NOT chemistry validation**.

## Baseline table (2026-10-09 fresh runs vs committed artifacts)

| Artifact | Fresh result | Byte-exact? | Deltas |
|---|---|---|---|
| `anyon_braid_results.json` | sha256 `1c337eb0…` | ✅ | none |
| `likeness_results.json` | sha256 `7f9d3b00…` | ✅ | none |
| `qudit_lhv_results.json` | all cells | numeric-equal | 1 doc-only meta note field |
| `validate_benchmark.json` | all numbers | numeric-equal | `generated_utc`, `owned_elapsed_s` |
| `error_bound_verify.json` | all numbers | numeric-equal | 22×`time_s`, `platform.numpy`, `generated` |
| `braid_vs_gd_results.json` | all numbers | numeric-equal | `wall_clock_seconds` |
| tests 21/21 | OK | — | — |
| HTTP battery | 8 endpoints | — | smoke: `docs/artifacts/jarvis_server_restart_smoke_20261009.txt` |

## Known warts (kept honest, not hidden)

- npz-in-cwd base swap (see §0) — the R&D API is immune; script runs must use a clean cwd.
- hardcoded absolute OUT paths in `likeness_probes.py` and nitrogenase standalone (see §0, §6).
- JSON artifacts embed run stamps / platform fingerprint by design.
- d=3,n=2 LHV row is solver-dependent non-facet (I<2, documented, never claimed).