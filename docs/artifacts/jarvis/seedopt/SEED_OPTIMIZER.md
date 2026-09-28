# SEED_OPTIMIZER — Core JARVIS #1: Variational Seed Optimizer

**Date:** 2026-09-18 · **Scale:** n = 32 (post-check scale) · **Objective:** combined = 0.4*bio_resonance_match + 0.3*coherence_spread + 0.3*braid_protection (maximize)
**Anchor:** ERROR_BOUND_PROOF.md Thm 3 — the family stays ≤ 3 parameters (measure-zero);
the optimizer picks the **best element of the family**, it does not grow it.

## Headline (honest)
> Optimized seeds keep **MSE=0**; the family is unchanged; the best element is now *found*, not guessed.

- Owner seed baseline: `[0.57721, 1.618034, 2.71828]` → combined 0.674036
- **Optimized seed (winner: cma_es):** `[4.91885182, 3.90224766, 2.260521]` → combined 0.701304
  (bio 0.987, coherence 0.026, braid 0.997)
- **Exactness post-check at n=32:** MSE = 0.0, bit-identical determinism = True → **PASS**

## Loss-landscape characterization (a) measured
- Samples: 512 uniform points in [0.01,6]³ → mean 0.6529 ± 0.0237, min 0.6213, max 0.6996, median 0.6526
- Local-slope median 0.1135; small-move improve fraction 0.46
- Readout: landscape featureless at probed scale (b) interpretation

## Optimizer comparison (a) measured — budget 2048 evals each
| strategy | best combined | Δ vs owner | evals used | converged-by (1%) |
|---|---|---|---|---|
| coordinate_descent | 0.698042 | +0.024006 | 13 | 0 |
| nelder_mead | 0.671343 | -0.002694 | 2048 | 6 |
| cma_es | 0.701304 | +0.027268 | 2052 | 48 |

## Integration readiness
- **Anyon experiment (4c706c3b):** ready — braid words derive from the seed via the core's
  deterministic braid params; the optimized seed above replaces the hand-fixed seed for braid-word generation.
- **Stage-3 blind prediction:** ready — any seed search the prediction needs can call this
  evaluator/CMA-ES path directly (`evaluate()` is a pure function of the seed).

## Reproducibility
`python3 seed_optimizer.py` (pure numpy; random streams seeded 20260918; base state pinned
to the documented deterministic fallback; fixed-topology slice — same conservative
restriction as the Thm-3 empirical confirmation). Results: `seedopt_results.json`.
