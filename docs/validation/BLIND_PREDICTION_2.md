# Second Blind Prediction — [2Fe-2S] Singlet–Triplet Gap (Freeze-First)

**Protocol:** freeze-first — the prediction and its derivation in
`BLIND_PREDICTION_2.json` were written and hash-committed **before any
reference value was opened**. This is the "second blind prediction" data
point mandated by the Validation Track 3/3 Stage-3 follow-up
(`docs/validation/BLIND_PREDICTION.md`, §4.3, line: *"a second, truly blind
target with a single published number (e.g., the [2Fe-2S] singlet-triplet gap
from specific DMRG/CASPT2 studies) run through the same freeze-first
protocol"*).

**Tags:** (a) = measured (direct stack output), (b) = interpretation (model
reading with stated assumptions), (c) = speculation (physical-match
hypothesis, evaluated only after reference unlock). No untagged claims.

---

## 1. Target and the honesty constraint

The target is the **singlet–triplet (S=0 → S=1) energy gap of the [2Fe-2S]²⁺
model cluster** (oxidized, diferric iron–sulfur dimer, two high-spin Fe³⁺
sites) as reported by a peer-reviewed multireference computational study
(DMRG, CASPT2, NEVPT2, or equivalent). This is a **single published number**.

The stack's own binding constraint (unchanged from Stage 3): *predict only
what the owned stack can actually compute*. The stack has **no spin
Hamiltonian and no [2Fe-2S] model** — its native observable is the
catalytic-barrier surrogate ladder `E_d` (d = 2..6) at the owner seed. It
cannot produce a native iron–sulfur dimer gap; manufacturing one is
forbidden. Accordingly this document (i) states plainly what the stack does
and does not compute, (ii) reuses the **exact Stage-3 derived quantity** — the
per-unit manifold splitting between the spin-only (d=2) and spin×oxidation
(d=4) Hilbert-space descriptions of the same seed — with **zero new free
parameters**, and (iii) compares that quantity **once** against **one**
published single number under the frozen rule below.

**Why this is a new data point (not a Stage-3 repeat):** Stage 3 compared the
per-unit scale against **broad published scales** (35–350 cm⁻¹ DMRG,
10–600 cm⁻¹ HDE, 3,497–6,994 cm⁻¹ manifold — ranges). This second test holds
the **same stack-determined prediction** to a **single, specific, sharper
number** extracted by a mechanical pre-registered rule. The prediction value
is fixed by the stack (no knob, no re-derivation, no possibility of tuning
the prediction to the reference); the new datum is the hard single-number
comparison.

**Explicitly NOT predicted (same honest ceiling as Stage 3):**
- a native physical [2Fe-2S] singlet–triplet gap (the stack has no spin
  Hamiltonian, no exchange couplings, no [2Fe-2S] model);
- any chemical accuracy claim for the `E_d` numbers (they are surrogate
  outputs of the approved stack, unchanged since Stage 3);
- any falsification of Validation Track stages 1–2 by a miss (representation
  exactness was proven separately in `ERROR_BOUND_PROOF.md` and is independent
  of this chemistry-level comparison);
- any hardware claim (exact classical simulation is the honest ceiling).

---

## 2. What the stack actually computes (fresh-run, pre-freeze, measured)

Fresh run on 2026-10-26 using the exact origin/main core blobs (hashes in
`BLIND_PREDICTION_2.json`), owner seed (0.57721, 1.618034, 2.71828),
n = 16 units, deterministic (MSE = 0 everywhere):

| d | energy barrier E_d (kJ/mol) | coherence | braid protection | Hilbert dim |
|---|-----------------------------|-----------|------------------|-------------|
| 2 (qubit, "spin-only") | (a) 144.7362 | 0.983987 | 0.997216 | 2^16 = 65,536 |
| 3 (qutrit) | (a) 129.9529 | 0.796321 | 0.996810 | 3^16 |
| 4 (ququart, "spin × oxidation") | (a) 100.7657 | 0.876241 | 0.995139 | 4^16 = 2^32 |
| 5 | (a) 76.0663 | 0.862775 | 0.997701 | 5^16 |
| 6 | (a) 50.1400 | 0.894684 | 0.995527 | 6^16 |

All values (a) measured; deterministic — numerical uncertainty 0. These five
numbers are **bit-identical** to the Stage-3 frozen table (fresh-run
reproducibility re-verified for this document), i.e. the pre-unlock core has
not changed.

---

## 3. The derived prediction (frozen — written before any reference value)

The only spin-related energy *difference* the model defines at zero added
parameters is the **manifold splitting** between Hilbert-space descriptions
of the same seed; the only no-new-parameter distribution over the 16 model
units is uniform (same interpretation step as Stage 3):

```
ΔE_mf = E(d=2) − E(d=4) = 144.7362 − 100.7657 = 43.9705 kJ/mol   (derivation (b), inputs (a))
      = 43.9705 × 83.593472 cm⁻¹ = 3,675.65 cm⁻¹                  (manifold scale, units constant (b))
Δε     = ΔE_mf / 16 = 2.748156 kJ/mol per unit ≡ 229.73 cm⁻¹      (per-unit scale, (b))
```

**Frozen prediction (BLIND_PREDICTION_2.json):**
1. **(primary, per-unit spin-splitting scale)** Δε = **229.73 cm⁻¹**
   (2.748156 kJ/mol per unit) — the Stage-3 primary, re-tested here against a
   **single published number** rather than a scale range.
2. **(secondary, manifold scale)** ΔE_mf = **3,675.65 cm⁻¹** (43.97 kJ/mol).
3. **(ordering, (a))** E_2 > E_3 > E_4 > E_5 > E_6 — adding spin/oxidation
   levels lowers the model's characteristic energy monotonically; any derived
   spin splitting is therefore *positive* (excited manifold above ground). For
   the target cluster this corresponds to the physical expectation that the
   antiferromagnetically coupled diferric dimer has a **singlet ground state
   with a triplet first excited state** (gap > 0).

**Tolerance (set a priori, pre-freeze):** numerical uncertainty = 0
(deterministic stack). Interpretational comparison band = the same factor-2
window used in Stage 3 around the primary prediction: **[114.9, 459.4] cm⁻¹**;
secondary band = ±50% around the manifold scale: **[1,837.8, 5,513.5] cm⁻¹**.
A reference value outside *both* bands is a **MISS**. Inside either band is a
**HIT** (primary band preferred; membership recorded for both).

**Conversion constants (frozen, fixed before reference extraction):**
1 kJ/mol = 83.593472 cm⁻¹; 1 kcal/mol = 349.755 cm⁻¹; 1 meV = 8.065544 cm⁻¹;
1 eV = 8,065.544 cm⁻¹; 1 hartree = 219,474.63 cm⁻¹. The extracted reference
value is converted **once** with these constants and reported in both the
original units and cm⁻¹.

---

## 4. Reference extraction rule (frozen — written before any reference value)

The reference is extracted **after** this document is committed and pushed,
by the following **mechanical** rule (no manual cherry-picking):

1. **Search** the peer-reviewed / preprint literature (arXiv full text and
   abstracts, publisher abstracts/DOIs) for the singlet–triplet gap of the
   reduced-oxidation **diferric [2Fe-2S]²⁺** iron–sulfur dimer model.
2. **Acceptable sources:** peer-reviewed multireference studies (DMRG,
   CASPT2, MS-CASPT2, NEVPT2, RASPT2) or high-level broken-symmetry
   treatments that report an **explicit numerical singlet–triplet (S=0 → S=1)
   energy difference** for an iron–sulfur dimer model in the 2+ (diferric)
   charge state. Values reported as exchange couplings **J** alone (in any
   convention) are NOT accepted as the target number (J → gap conversion
   involves convention assumptions; Stage-3's HDE 10–600 cm⁻¹ range already
   covered couplings; this test targets the gap itself).
3. **Scan order (mechanical):** sources are scanned in the order in which the
   designated search returns them (first result first). Take the **first**
   accepted source whose text contains an explicit numerical singlet–triplet
   gap value for the [2Fe-2S]²⁺ dimer model.
4. **Extraction:** record the number, its unit, its exact context sentence,
   and the DOI/arXiv identifier. Convert **once** with the frozen constants
   above.
5. **If no accepted source yields an explicit single number after scanning
   the first five distinct sources returned:** the outcome is recorded
   honestly as **"no suitable single-number reference found"** (a VOID data
   point — neither hit nor miss), which the plan explicitly accepts as a
   legitimate outcome ("No suitable target found" is an acceptable, honest
   result). No substitute reference is then invented.

**Rules of engagement (hard):**
- NO band adjustment after extraction; ONE comparison, ONE verdict.
- The reference value is NOT consulted before the freeze push completes.
- Only the AFTER section below the freeze line may be added post-unlock; the
  frozen content above is never edited (hash must still verify).

---

## 5. Freeze line

Freeze digests (computed pre-push over the exact pre-unlock code and
parameters; full manifest in `BLIND_PREDICTION_2.json`):
- owner seed digest (sha256): `c35bca531ec4a1a30b30af971df07f2276964114a366f6cb660f78475af98ed0`
- code blob `compression_specialist.py` (sha256): `9ca44863dfbeac9c95911ed5a3b790b45c619e0bb89b376e805b339ac2b92e5f`
- code blob `nitrogenase_qudit_simulator.py` (sha256): `8d7eda3f075ceac04df0c96a8ac47086de1e1f15ca15b91a55758bc8c3478154`
- origin/main commit pinned: `a680d23d66caae54b73248042deead430015e85a`
- freeze_sha256 (over the byte-exact frozen file `BLIND_PREDICTION_2.frozen.json`): `af4d8fdfdfccd14ac2e187079b86800c161effd7fd62dbd724c43ebe00ade977`

---

## 6. AFTER — comparison with the published reference (unlocked after freeze)

> Nothing below this line existed before the freeze push. Written only after
> the frozen content above was committed and pushed to
> `origin/docs/blind-prediction-2`, and only after the reference extraction
> rule in §4 was run mechanically for the first time.