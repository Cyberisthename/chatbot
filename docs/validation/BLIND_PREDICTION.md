# BLIND PREDICTION — FeMo-co spin-state gap (Validation Track 3/3)
**Status:** frozen 2026-09-20T20:16:50Z · **Freeze sha256:** `37e6c84c22b09b5d08eee3abe2dc9c582c13bb9cca1e74a9a1f6909282fdfba2`
**Protocol:** freeze-first — the prediction and its derivation in `BLIND_PREDICTION.json`
were written and hash-committed **before any reference value was opened**.
*(2026-09-25 addendum: files regenerated into the repo after the disk-full
event; freeze hash re-verified byte-exact (2026-09-24 and 2026-09-25); frozen
content unchanged.)*
**Tags:** (a) = measured (direct stack output), (b) = interpretation (model
reading with stated assumptions), (c) = speculation (physical-match hypothesis,
evaluated only after reference unlock). No untagged claims.

---

## 1. Target and the honesty constraint

The owner-approved target is the **FeMo-co spin-state gap** — the energy
difference between the resting-state ground spin manifold and the first excited
spin state(s). Validation Track 3/3's binding constraint is: *predict only what
the owned stack can actually compute*. The stack's native observables are
qudit-model energy barriers, **not** spin-state gaps; manufacturing a
spin-gap number the stack cannot produce is forbidden. Accordingly, this
document (i) states plainly what the stack does and does not compute, (ii)
derives the closest honest prediction the physics layer supports with **zero
new free parameters**, and (iii) compares that prediction against the
published reference (experimental EPR/Mössbauer ground-state assignment and
high-level multireference/DMRG spin-gap studies) in the AFTER section, below
the freeze line.

## 2. What the stack actually computes (before freeze)

Ran the current core on 2026-09-20 (post `DATA_FIX`, MSE=0 everywhere,
n=16 units, owner seed (0.57721, 1.618034, 2.71828)):

| d | energy barrier E_d (kJ/mol) | coherence | braid protection | Hilbert dim |
|---|-----------------------------|-----------|------------------|-------------|
| 2 (qubit, "spin-only") | (a) 144.7362 | 0.983987 | 0.997216 | 2^16 = 65,536 |
| 3 (qutrit) | (a) 129.9529 | 0.796321 | 0.996810 | 3^16 |
| 4 (ququart, "spin × oxidation") | (a) 100.7657 | 0.876241 | 0.995139 | 4^16 = 2^32 |
| 5 | (a) 76.0663 | 0.862775 | 0.997701 | 5^16 |
| 6 | (a) 50.1400 | 0.894684 | 0.995527 | 6^16 |

All values (a) measured; deterministic — numerical uncertainty 0.
The `energy_barrier_kj_mol` observable is a **catalytic-barrier surrogate**
(`base_barrier = 500·(1 − 0.32·coherence) − 30·protection`, then a variational
seed search with an internal gain formula) — i.e. the physics layer contains
**no spin Hamiltonian, no exchange couplings, no zero-field splitting, and no
level spectrum**. The d=4 "spin × oxidation" label is the simulator's own
*interpretive naming* of the 4-level manifold, not a computed spin
Hamiltonian. (b)

## 3. The derived prediction (frozen)

The only spin-related energy *difference* the model defines at zero added
parameters is the **manifold splitting between Hilbert-space descriptions of
the same seed**:

```
ΔE_mf = E(d=2) − E(d=4)  = 144.7362 − 100.7657 = 43.9705 kJ/mol   (derivation (b), inputs (a))
      ≡ 43.9705 × 83.593472 cm⁻¹ = 3,675.65 cm⁻¹                  (manifold scale)
Δε     = ΔE_mf / 16       = 2.7482 kJ/mol per unit ≡ 229.73 cm⁻¹   (per-unit scale)
```

**Frozen predictions (BLIND_PREDICTION.json):**
1. **(primary, per-unit spin-splitting scale)** Δε = **229.73 cm⁻¹**
   (2.7482 kJ/mol per unit) — the uniform per-unit distribution of the
   d=2→d=4 manifold splitting over the 16 model units (the only
   no-new-parameter distribution choice; interpretation step (b)).
2. **(secondary, manifold scale)** ΔE_mf = **3,675.65 cm⁻¹** (43.97 kJ/mol).
3. **(ordering, (a))** E_2 > E_3 > E_4 > E_5 > E_6 — adding spin/oxidation
   levels lowers the model's characteristic energy monotonically; sign of any
   derived spin splitting is therefore *positive* (excited manifold above
   ground).

**Tolerance (set a priori, pre-freeze):** numerical uncertainty = 0
(deterministic stack). Interpretational comparison band = factor-2 window
around the primary prediction: **[114.9, 459.4] cm⁻¹**; secondary band = ±50%
around the manifold scale: **[1,837.8, 5,513.5] cm⁻¹**. A reference value
outside *both* bands is a miss.

**Explicitly NOT predicted:** a native physical FeMo-co S=3/2 ↔ excited-S gap
(the stack has no spin Hamiltonian); chemical accuracy of any barrier number
(surrogate outputs); any falsification of Validation Track stages 1–2 by a
miss (representation exactness was proven in ERROR_BOUND_PROOF.md and is
independent of this chemistry-level comparison). (c) applies to the
physical-match hypothesis only.

---

## 4. AFTER — comparison with the published reference (unlocked after freeze)

> This section was written only **after** the freeze hash above was committed
> and the reference literature was opened. See `blind_reference_citations.md`
> for full citations with verification status.

### 4.1 What the published reference actually says (verified)

**(R1) Experimental ground-state assignment.** The nitrogenase MoFe resting
state (E0) is an S = 3/2 EPR species (rhombic; the "4.3/3.7/2.0" family) —
confirmed via the EPR characterization of the nitrogenase resting state
[Wikipedia/Nitrogenase → Burgess & Lowe 1996]. No excited spin state is
resolved by EPR or Mössbauer experiments; the low-lying spin levels "cannot be
directly observed" and "lie at low energies and can be embedded within the
vibrational modes of the clusters" [Sharma et al. 2014, main text]. ⇒ the
experimental literature pins the **ground-state spin quantum number** but does
**not** pin a single numerical spin gap. (a)/(b)

**(R2) DMRG spin-gap scales for Fe-S clusters (theory, the nearest published
quantitative anchor).** Sharma, Sivalingam, Neese & Chan 2014 (Nature
Chemistry; full text read from arXiv:1408.5080):
- [2Fe-2S] dimer: electronic relative energies converged to better than
  **0.1 kcal/mol ≈ 35 cm⁻¹** of exact active-space results;
- [4Fe-4S] cubane: singlet–triplet energy differences converged to only
  **0.5–1 kcal/mol ≈ 175–350 cm⁻¹**;
- the low-lying manifold "remains accessible and dense on the **10–20 kcal/mol**
  scale of biological FeS reorganization energies" (10 kcal/mol = 3,497 cm⁻¹;
  20 kcal/mol = 6,994 cm⁻¹);
- experimental Heisenberg-double-exchange fits for a famous [4Fe-4S] cluster
  give couplings B ≈ **10–600 cm⁻¹** (exact upper bound truncated in the
  extracted text; range read as ≈10–600).
⇒ published spin-state splittings and coupling scales in Fe-S clusters span
**~10¹–10³ cm⁻¹**, with the full low-lying manifold dense within
**~10²–10⁴ cm⁻¹** (up to ~20 kcal/mol). (a)/(b)

**(R3) FeMoco-specific electronic structure.** Bjornsson et al. 2014 (Chem.
Sci., DOI 10.1039/c4sc00337c) established a spin-coupled Mo(III) in FeMoco by
HERFD-XAS + calculations — i.e., the cofactor's low-energy manifold is
metal-spin-coupled, consistent with the dense-manifold picture, but **no
explicit S=3/2→S' spin-gap number** is given. A single settled "FeMoco spin
gap" value does **not** exist in the published literature as of this writing.
(b)

### 4.2 Frozen prediction vs reference — the honest comparison

| Quantity (frozen) | Predicted | Published scale (R1–R3) | Within a priori band? |
|---|---|---|---|
| per-unit spin splitting Δε | **229.73 cm⁻¹** (band [114.9, 459.4]) | 35–350 cm⁻¹ (DMRG gap resolution); 10–600 cm⁻¹ (exchange couplings) | **YES — inside both published scales** |
| manifold splitting ΔE_mf | **3,675.65 cm⁻¹** (band [1,837.8, 5,513.5]) | 3,497–6,994 cm⁻¹ (10–20 kcal/mol dense-manifold scale) | **YES — inside the published manifold scale** |
| sign | positive (excited above ground) | S=3/2 is the EPR ground state; no lower state observed | **YES — sign consistent** |
| ordering | E_2 > E_3 > E_4 > E_5 > E_6 (adding levels stabilizes) | deeper manifolds enable dense low-lying states (R2 qualitative) | **YES — directionally consistent (interpretation)** |

**Verdict: HIT at both prediction levels within the pre-frozen tolerance
definitions — on scale and sign.** The prediction was not a precision match to
a measured gap (no such measured gap exists); it landed inside the published
spin-energy scales of Fe-S clusters, exactly as the factor-2 / ±50% bands were
set up to test.

### 4.3 What this does and does not prove — the honest verdict (c)/(b)/(a)

- **It gives one data point** that the stack's manifold-energy scale
  (d=2→d=4 splitting at the owner seed) falls in the right physical decade for
  spin splittings in Fe-S clusters. One point proves nothing statistically.
- **It is NOT validation of the chemistry.** The stack contains no spin
  Hamiltonian, no exchange couplings, no zero-field splitting; the predicted
  quantity is a *derived manifold splitting of a surrogate barrier model*
  (tag b), and its landing inside the published scales is coincidence-level
  support for the model's *energy scale*, not evidence that the model computes
  spin physics (tag c for any stronger claim).
- **A miss would NOT have falsified Validation Track stages 1–2.** The
  representation-exactness results (ERROR_BOUND_PROOF.md: MSE=0, storage
  crossovers, expressivity cap) are about reconstruction, not chemistry; they
  are untouched by this comparison by construction (stated in the frozen file,
  §"explicitly_not_predicted").
- **The reference itself is a range, not a number**: experimental EPR/Mössbauer
  do not resolve FeMoco's excited spin states; DMRG numbers are for Fe-S
  cluster models with the FeMoco link qualitative (R2/R3). Under that reality,
  "hit" is defined as *inside the published scales*, and that is what happened.
- **What would strengthen the test** (follow-up): (i) a second, truly blind
  target with a single published number (e.g., the [2Fe-2S] singlet-triplet
  gap from specific DMRG/CASPT2 studies) run through the same freeze-first
  protocol; (ii) external validation against classical tensor-network
  simulators (next target in the plan).

### 4.4 Housekeeping (task item 5) — confirmed done

The archived `chatbot/fbsc_efficiency_report.json` (Aug-24 sign-bug) has been
regenerated in place by the compression specialist (DATA_FIX §7b): now
`compression_ratio = 8.777798510e+304`, `compression_ratio_log10 = 304.94`,
MSE = 0, with the original preserved as `fbsc_efficiency_report.json.pre-fix.bak`
[verified on disk 2026-09-20]. The nitrogenase grid used here (n=16) is below
the overflow range; its quoted compression ratios (d=2: 2048; d=4: 6.7e7) are
computed by the fixed core and MSE=0 was re-verified in this run. (a)