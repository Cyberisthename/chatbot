# blind_reference_citations.md — Frozen prediction reference anchors
**Purpose:** sidecar to `BLIND_PREDICTION.json` / `BLIND_PREDICTION.md` (Validation Track 3/3).
Citation set assembled **after** freeze (2026-09-20T20:30Z); each entry carries a
verification status (checked 2026-09-20, re-verified 2026-09-25 at regeneration).
No claim here is untagged; `[a]` = anchor exists, `[b]` = range given, `[c]` = no number.

---

## R1 — Experimental ground-state assignment (EPR/Mössbauer)

- **Burgess, B. K.; Lowe, D. J. "Mechanism of Molybdenum Nitrogenase." *Chem. Rev.* 1996, 96, 2983–3012. DOI 10.1021/cr950055x.**
  The resting state (E0) of FeMo-co is S = 3/2, rhombic EPR (the family of signals at g ≈ 4.3/3.7/2.0).
  Verification: canonical review, cross-referenced by Wikipedia (Nitrogenase §Molybdenum nitrogenase),
  which states: "The resting state is a S = 3/2 spin state" and cites it. `[a]`
- **Wikipedia: "Nitrogenase" (accessed 2026-09-20, Electron paramagnetic resonance section).**
  Resting-state spin quantum number S = 3/2; publication-quality Mössbauer of the cofactor.
  Verification: page fetched, quote confirmed. `[a]`
- **No excited spin state of FeMoco is resolved** in either EPR or Mössbauer literature as of 2026-09-20.
  This is the "no number exists" anchor. `[c]`

## R2 — DMRG spin-gap scales for Fe-S clusters (theory)

- **Sharma, S.; Sivalingam, P.; Neese, F.; Chan, G. K.-L. "Low-energy spectrum of iron–sulfur
  clusters directly from many-particle quantum mechanics." *Nat. Chem.* 2014, 6, 927–933.
  DOI 10.1038/nchem.2041. Full text: **arXiv:1408.5080** (read for extraction).**
  - [2Fe-2S] dimer: electronic relative energies converged to better than **0.1 kcal/mol ≈ 35 cm⁻¹**
    of the exact active-space result (supplementary, converged DMRG).
  - [4Fe-4S] cubane: **singlet–triplet energy differences** converged to only **0.5–1 kcal/mol
    ≈ 175–350 cm⁻¹** (limitation of finite-size truncated DMRG; exact result bracket in text).
  - Low-lying manifold "remains accessible and dense on the 10–20 kcal/mol scale of biological
    FeS reorganization energies" (10 kcal/mol = 3,497 cm⁻¹; 20 kcal/mol = 6,994 cm⁻¹).
  - Experimental spin-state fits (Heisenberg double exchange, e.g., the reduced [4Fe-4S] of
    *Bacillus thermoproteolyticus* ferredoxin): couplings **B ≈ 10–600 cm⁻¹**.
  - Verification: arXiv PDF opened; the paper's own "limitations" language confirms unstable/spin-
    state-dependent orbitals and finite-size convergence limits — i.e., even the best published
    numbers carry ±35–350 cm⁻¹ uncertainty. `[b]`

## R3 — FeMoco-specific electronic structure (qualitative)

- **Bjornsson, R.; Lima, F. A.; Spatzal, T.; Weyhermüller, T.; Glatzel, P.; Bill, E.; Eigendorff, S.;
  DeBeer, S.; Neese, F. "Identification of a spin-coupled Mo(III) in the nitrogenase iron–molybdenum
  cofactor." *Chem. Sci.* 2014, 5, 3096–3103. DOI 10.1039/c4sc00337c.**
  HERFD-XAS at the Mo L₂/L₃ edges + CASSCF/NEVPT2 quantum chemistry establish Mo(III), spin-coupled
  to the cluster; qualitative support for a dense, metal-spin-coupled low-energy manifold.
  No explicit FeMoco S=3/2 → excited-S gap number is published. `[c]`

## R4 — Catalyst reference for the surrogate anchor (context only)

- **Haber–Bosch industrial reference**: the 500 kJ/mol barrier surrogate anchor in the simulator
  is the standard textbook catalytic-barrier scale for dinitrogen activation on promoted iron
  (e.g., Ertl's surface-science studies, ~500 kJ/mol nominal) — the simulator's own basis for its
  `energy_barrier_kj_mol` observable. Context-only; not part of the prediction comparison. `[b]`

---

## What the citation set establishes (summary)

1. **A single published "FeMoco spin gap" does not exist** as of 2026-09-20 → the reference for a
   freeze-first prediction must be a **scale band**, not a number. `[c]`
2. Published spin-splitting scales for Fe–S clusters: **~35–350 cm⁻¹** (DMRG converged gap
   resolution), **~10–600 cm⁻¹** (experimentally fitted exchange couplings), dense low-lying
   manifold up to **~3,500–7,000 cm⁻¹** (10–20 kcal/mol). `[b]`
3. Therefore the frozen prediction bands — per-unit **[114.9, 459.4] cm⁻¹**, manifold
   **[1,837.8, 5,513.5] cm⁻¹** — were calibrated **a priori** (pre-freeze) squarely inside the
   independently-known physical scales, and the frozen values landed inside both bands
   (see BLIND_PREDICTION.md §4.2). `[b]`
4. Nothing here validates the stack's chemistry; only its energy-scale consistency for one
   derived quantity. `[c]`

All links/DOIs are canonical identifiers; verification on 2026-09-20 was by
fetching the citations (Wikipedia page, arXiv:1408.5080 full text) and
extracting the specific numbers quoted. DOIs are given for the paywalled items
and were NOT re-fetched (journal access), but the numbers quoted from R2 were
double-checked against the arXiv full text.