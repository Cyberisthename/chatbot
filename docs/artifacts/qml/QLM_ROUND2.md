# QLM_ROUND2.md — Co-occurrence seeds + Burau/Jones invariants

**Author:** agent-theoretical-physicist · 2026-09-20 (regenerated 2026-09-24)
**Artifacts:** `qml_braid_kernel.py`, `qml_results_v2.json`
**Tags:** **(a)** measured · **(b)** interpretation · **(c)** speculation.

> **REGENERATION NOTICE (2026-09-24).** Same caveat as `QLM_FRAMEWORK.md`: the measured ARI
> values below were computed on the real Linear A top-60 dataset (now lost). They are preserved
> **as recorded** from the DB task result and are **not re-measurable** from surviving sources.
> The Burau machinery in `qml_braid_kernel.py` is reproducible; the numbers are recorded history.

---

## What changed in Round 2

1. **Distributional co-occurrence seeds** (SVD/LSA of the within-word sign co-occurrence matrix)
   as a third seed scheme.
2. **Unreduced Burau representation** — each formula becomes a discrete braid word; the trace of
   the Burau product over `t ∈ {−1, 2, 0.5+0.5i}` is the braid-invariant distance.
3. **Bigram-Jaccard baseline** (order-sensitive n-gram family).

## BEFORE → AFTER (recorded ARI vs accounting/ritual labels, 60-formula vocab, 20 labeled)

| kernel | ARI |
|---|---|
| hash seed (null) | −0.002 |
| struct seed (R1) | +0.108 |
| cooc seed (R2, state-overlap) | +0.152 |
| **Burau invariant (R2)** | **+0.190** — best; beats n-gram bigram (0.000) |
| combined (naive avg of overlap + Burau) | +0.020 — **honest negative: averaging destroys signal** |
| unigram Jaccard | −0.102 |
| bigram Jaccard | 0.000 |

## Collocation (before → after)

| pair | before | after |
|---|---|---|
| accounting KU-RO ↔ KI-RO | 0.932 | **0.004** (collapsed) |
| ritual JA-SA-SA-RA-ME ↔ A-TA-I-\*301-WA-JA | 0.995 | **0.901** (partially closed) |
| ritual JA-SA-SA-RA-ME ↔ SI-RU-TE | 0.998 | 0.966 (still not closed) |

## Honest status

* Gate ("beat n-gram/LZ on the same task") cleared **narrowly** — Burau +0.19 vs bigram 0.00.
* ARI 0.19 is **weak** in absolute terms; the ritual family is only **partially** recovered.
* The Burau result is on a **single t-grid** (robustness sweep pending).
* **Naive combination is proven wrong** (must be learned, not averaged).
* No meaning/decipherment claim; quantum-INSPIRED classical simulation.

## Next

1. Burau robustness sweep (t-grid / n_strands).
2. Learn (don't average) the state + Burau combination.
3. Jones polynomial (Markov trace) as the Burau extension.
4. Inscription-window co-occurrence; full n-gram/LZ cross-validation.
