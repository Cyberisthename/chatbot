# QLM_FRAMEWORK.md — QLM Layer: FBSC braid-semantic language kernel

**Author:** agent-theoretical-physicist · 2026-09-20 (regenerated 2026-09-24)
**Artifacts:** `qml_braid_kernel.py`, `qml_results.json`, `qml_results_v2.json`, `QLM_ROUND2.md`
**Tags:** **(a)** measured · **(b)** interpretation · **(c)** speculation.

> **REGENERATION NOTICE (2026-09-24).** The code in `qml_braid_kernel.py` reproduces the
> kernel *machinery* deterministically. The measured **ARI numbers below were computed on the
> real Linear A top-60 formula dataset with register labels, which did NOT survive the disk-full
> event** (not in the repo, not in `/var/tmp` backups). They are therefore preserved here **as
> recorded** and are **not re-measurable** from surviving sources. Fabricating a re-run on fake
> data would be dishonest; the recorded values are reproduced verbatim from the DB task results.

---

## 1. What the QLM layer is

A braid-attention kernel that turns each sign/syllable token into a **braid**, and each
word/formula into the **non-commutative composition** of its token braids. "Meaning" is then a
geometric/topological *state*, not a bag of features. This is quantum-**inspired** classical
simulation — no quantum hardware.

## 2. The kernel (five steps)

1. **FBSC 3-seed.** Each token gets a 3-number seed, from one of three schemes:
   - *hash* (arbitrary — the null control),
   - *structure* (frequency + positional coordinate, normalized),
   - *co-occurrence* (SVD/LSA of the within-word sign co-occurrence matrix — distributional).
2. **Token → braid word.** The seed deterministically expands to an ordered braid word.
3. **Token → braid unitary.** The braid word maps to an exchange-with-phase braid matrix
   (the unreduced Burau representation here).
4. **Word = braid composition.** A formula's tokens compose *in order* (non-commutative),
   so token order is part of the structure.
5. **Semantic distance** `= 1 − state-overlap` between composed braid states.

## 3. Boundary markers

Dedicated boundary seeds segment a token stream into phrase braids: U+10101 (word separator),
U+1076B (damage marker), U+10607 (prefix sign).

## 4. Measured result — Round 1 (recorded, not re-measurable)

On the real Linear A top-60 formulas, structure-only clustering recovers register
stratification (accounting R1 vs ritual R3) **weakly but above chance**:

| kernel | ARI |
|---|---|
| hash seed (null control) | −0.002 |
| **struct seed (R1)** | **+0.108** |
| bag-of-signs (order-insensitive) | −0.102 |

Same-register pairs are closer than cross-register pairs (0.862 vs 0.902). **(b)** The signal
comes from structure-derived seeds + braid order-sensitivity, **not** the braid algebra alone
(the hash-seed null is ~0).

## 5. Honest limits

* Recovery is **partial/lopsided**: accounting clusters via the shared `-RO` suffix; the ritual
  family does *not* cluster (frequency+positional seeds don't capture family membership).
* Weaker than n-gram/LZ — **complementary, not competitive**.
* No meaning / decipherment claim (structure ≠ meaning; no bilingual).

## 6. Next steps (→ Round 2)

Distributional co-occurrence seeds + Burau-trace/Jones braid invariants to close the ritual gap,
phrase-level braid measurement, then a Majorana-2 hardware port (not implemented).
