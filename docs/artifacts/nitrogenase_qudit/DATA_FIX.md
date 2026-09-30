# DATA FIX — nitrogenase qudit Hilbert-dim inconsistency (d=2 row: 5^16 → 2^16)

**First applied:** 2026-09-18 (approved, task `2124404a`)
**Re-applied after disk-full incident:** 2026-09-24 (this session) — originals destroyed on
2026-09-24; the same documented fix was re-applied to the surviving source files and the
report was regenerated fresh. Numbers below re-verified by fresh run.

## 1. What was wrong
The stage-1 report's d=2 (qubit) row recorded:
- `total_hilbert_dim = 152,587,890,625 = 5^16`
- `memory_kb = 1.25` (16 × 5 × 16 / 1024)

Both implicate a **5-level manifold**, inconsistent with the 2-state d=2 the same report
declares. Root cause: the old d=2 row was produced by a prior FBSC core revision that
constructed a 5-level qudit (5^16, 1.25 KB). The current owned core correctly dispatches
d=2 to the real qubit path, which returns a **1-D `(n,)` register** — and the simulator's
analysis helpers (`coherence_of`, `braid_pathways`, `bio_resonance`) crashed on the 1-D
shape. Bonus defect: `compression_ratio` was the uniform headline ~1e73 on every row
instead of the honest per-row `d^16 / (n·d)`.

## 2. Fixes applied
1. `nitrogenase_qudit_simulator.py` — made `coherence_of` / `braid_pathways` /
   `bio_resonance` shape-safe for 1-D qubit registers (d=2 row now runs on the true
   2-level state; the n register entries play the role of "levels" in the original
   participation-ratio measures). No core file, physics layer, or seed changed.
2. Re-ran the simulator (same owner seed `(0.57721, 1.618034, 2.71828)`, n=16, sim id
   `nitro-qudit-c35bca531ec4`).
3. New d=2 row — **Hilbert dim 65,536 = 2^16, memory 0.25 KB, compression 2,048, MSE=0.**
   d=3/d=4 Hilbert dims + memory unchanged (3^16 = 43,046,721; 4^16 = 4,294,967,296 ✓).
4. Corrected report JSON + MD regenerated and propagated to all canonical locations
   (shared root, `nitrogenase_artifacts/`, chatbot `docs/artifacts/nitrogenase_qudit/`
   and `/artifacts/`, `jarvis/nitrogenase/`) plus `owner_quantum_seed_d2/d3/d4.json`.
5. Headline d=4 values re-verified: energy barrier **100.7657 kJ/mol**, synthetic
   V-Fe-S catalyst barrier **80.613 kJ/mol** (match approved post-fix numbers).

## 3. Verification summary (fresh run, 2026-09-24)
| d | Hilbert dim | memory_kb | compression_ratio | MSE | energy kJ/mol |
|---|------------:|----------:|------------------:|----:|--------------:|
| 2 (qubit)   | 65,536      | 0.25 | 2,048       | 0.0 | 144.7362 (catalyst 115.789) |
| 3 (qutrit)  | 43,046,721  | 0.75 | 896,807     | 0.0 | 129.9529 (catalyst 103.962) |
| 4 (ququart) | 4,294,967,296 | 1.00 | 67,108,864 | 0.0 | **100.7657** (catalyst **80.613**) |

## 4. Do-not-change list
- Owner seed `(0.57721, 1.618034, 2.71828)` — never alter.
- n=16 logical qudits, sim id `nitro-qudit-c35bca531ec4` — stable anchor.
- Physics layers (braid protection, 41.02 Hz resonance, variational barrier,
  time-reversal) — untouched by this fix.
- Core file `compression_specialist.py` — only the simulator helpers changed.
  (Core DID gain the approved log-space `_safe_compression_ratio` metric on 2026-09-19;
  that fix was re-applied 2026-09-24 after it was found missing from the git-restored copy.)

*All numbers in this file are (a) measured — produced by fresh deterministic runs on the
current owned core; no claim of physical chemistry validation.*