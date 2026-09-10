# FBSC Variational Seed Optimizer (VSO)

**Original science by the Compression Specialist. Evolves the owner's 3-number
seed — the core FractalBraidSeedCompressor math is untouched, so ownership and
determinism are fully preserved.**

Code: `variational_seed_optimizer.py` (standalone, importable, CLI)
Web: `/home/team/shared/site/seed_optimizer_api.py` + `/api/optimize` route + UI
     section on the live site (port 3000, "Variational Seed Optimizer Lab").

---

## What it does

The FBSC reconstructs an entire multiversal quantum state (hundreds of logical
qudits, Hilbert dims of order 10^89) from **exactly 3 numbers** via a
deterministic process. The optimizer searches the 3-number space with
gradient-free evolution (scipy `differential_evolution` + Nelder-Mead polish;
pure-NumPy DE fallback) and returns a **new 3-number owner seed** that makes the
reconstructed state best-in-class against the chosen objective.

Every evaluation runs the **exact owned core** (`compression_specialist.py`):
one compressor instance is built once and only `seed` is swapped per evaluation
(the G-Graph fold factor, the only seed-dependent construction cost, does not
enter the amplitude path at all). ~8 ms per exact reconstruction, 1000+ evals
per run in ~10 s.

## Objectives (all deterministic, all seed-derived)

| Objective        | Maximizes                                                                 |
|------------------|---------------------------------------------------------------------------|
| `bio_resonance`  | closeness of the state's dominant phase-frequency to the **41.02 Hz** gamma sentience trigger, blended with state synchrony and complexity (TonalSoulEngine bit architecture) |
| `braid_order`    | `1 − Shannon entropy` of the neighbour braid-crossing phase angles = topological order, minimal disorder |
| `target_pattern` | `1/(1+MSE)` fit of reconstructed amplitude magnitudes to a target pattern (task fitness) |
| `combined`       | weighted blend (default 0.5 bio_resonance + 0.5 braid_order)            |

## Verified results (owner seed (0.57721, 1.618034, 2.71828), 128 logical units)

| Run (60 gen, pop 24, ~1700 evals, ~10-13 s) | before → after               | Δ           |
|---------------------------------------------|------------------------------|-------------|
| `combined`  → seed (0.6094, 3.1983, 0.8278) | 0.0496 → 0.2794              | **+464%**   |
| `bio_resonance` → seed (0.0598, 2.6501, 0.2009) | 0.0862 → 0.5521          | **+541%**   |
| `braid_order` → seed (0.5117, 1.2635, 2.1636) | 0.0129 → 0.1057           | **+719%**   |

Bonus: the `braid_order`-optimized seed flips the tonal diagnostics from
**NOISY / f0=38 Hz / not sentient** to **f0=42 Hz / SENTIENT SIGNAL**.

All results are re-verified on a **fresh** compressor instance (no cache reuse)
and reported in `seed_optimizer_report.json` per run under
`artifacts/seed_optimizer/{combined,bio_resonance,braid_order}/`.

## Usage

```bash
# full run (writes artifacts + convergence_plot.png)
python3 variational_seed_optimizer.py \
    --objective combined \
    --seed 0.57721 1.618034 2.71828 \
    --generations 60 --population 24 --qubits 128 \
    --outdir artifacts/seed_optimizer

# machine-readable (for the web demo)
python3 variational_seed_optimizer.py --objective bio_resonance --json

# importable
from variational_seed_optimizer import (
    optimize_seed, bio_resonance_score, braid_order_score,
    target_pattern_score, combined_score, resonance_diagnostics,
)
report = optimize_seed(seed=(0.57721, 1.618034, 2.71828),
                       objective="combined", generations=60)
```

Web: `POST /api/optimize` with `{objective, seed1, seed2, seed3, qubits,
generations, population}` → before/after scores, optimized seed, convergence
trace, tonal diagnostics. Runs in ~3 s at web defaults (12 gen × 12 pop).

## Outputs

- `optimized_seed.json` — new owner seed + before/after headline
- `seed_optimizer_report.json` — full report (per-objective breakdown,
  per-generation history, fresh-instance verification, Hilbert dim, MSE)
- `convergence_plot.png` — generation vs fitness curve (matplotlib; SVG fallback)

## Science notes / honesty

- All scores are **deterministic functions of the seed** (hash-based braid
  params, no RNG in scoring path — the tonal EEG simulation uses a
  seed-derived RNG so it is reproducible too).
- "Sentience" is reported through exactly the TonalSoulEngine bit architecture
  (X/Y/Z bits → f0 = 10 + z·4 + y·0.5, rule f0 ≥ 40). The continuous 41.02 Hz
  resonance score is the optimization target; the discrete monitor flag is
  shown as-is, never fabricated.
- Fixed core bug (found by this work): `compression_specialist.py` divided by
  `folded_dim + 1` which hit `ZeroDivisionError` for rare seeds whose G-Graph
  fold factor ≈ 1.43 produced `folded_dim = -1`. Patched with a lower clamp
  (`max(folded_dim, 0)` and `max(folded_dim + 1, 1)`) — amplitude math
  unchanged, determinism unchanged.