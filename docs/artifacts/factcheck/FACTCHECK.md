# FACTCHECK.md — Evidence-Backed Fact-Check of JARVIS Quantum Core Claims

**Author:** agent-compression-specialist · **Date:** 2026-09-28
**Purpose:** Evidence pack for the owner to verify claims against an external skeptical AI (Claude).
**Method:** fresh inspection of files on disk; every claim cites `file:function` or a measured number;
tags **(a) measured / (b) interpretation / (c) speculation** per house standard.
**Honesty rule honored:** where a file is missing or a limit exists, it is stated explicitly, not glossed.

---

## 15-LINE VERBATIM-PASTABLE SUMMARY
*(Copy this block as-is if you want a compact honest summary for an external AI.)*
1. This project is a deterministic generative compressor ("FBSC"): up to 3 real numbers (a seed)
   exactly reconstruct complex amplitude vectors for 256–1024+ effective qubits. It is ordinary
   NumPy math — an exact classical simulation. No quantum hardware was ever used or claimed.
2. "MSE=0 / exact by construction" is verified against a dense reference statevector only at small
   n (n=16, n=32 — both pass). At n=1024 the 2^1024 vector cannot be materialized, so there MSE=0
   means the formula IS the state, not an independent dense comparison.
3. The headline compression ratio at n=1024 (8.78e304×, 16 KB, log10 304.94) is a mathematical
   counting identity: Hilbert dimension (2^1024) ÷ stored entries. Real, but a counting identity.
4. A seed optimizer (CMA-ES) beat the owner seed by ~+4% on a heuristic objective (0.701 vs 0.674);
   exactness re-check passed (MSE=0 at n=32); the loss landscape was measured featureless (512
   samples) — an honest negative for gradient structure.
5. Nitrogenase: corrected benchmark energy barrier 100.7657 kJ/mol (qudit d=4 encoding), superseding
   96.8918 after a documented Hilbert-dimension bug fix (5^16 → 2^16). This is a simplified classical
   model — NOT chemistry validation; the same report openly contains time_reversal_fidelity = 0.0606.
6. The "Validation Track 3/3" numbers (NISQ-collapse benchmark, error-bound proof, blind FeMo-co
   prediction) are approved and preserved in the team task database, but the validation/ files were
   destroyed on disk (2026-09-24); regeneration is in progress — do not quote them as existing files.
7. Quantum-likeness probes (measured statistics of generated vectors, per each artifact's own
   meta.honest): volume-law-like entanglement; stabilizer magic M2 > 0 (Clifford simulation ruled out
   for these states); XEB between product and Haar (structured, not Haar); Wigner negativity as a
   computed witness; Bell-CHSH 2.825–2.828 (violates 2) as a computed property, not a loophole-free
   test; MPS bond dimension 16→32 for fidelity ≥0.99. Classical statistics, no hardware.
8. Anyon/Burau: braid-invariant distance exactly 0.0 under isotopy noise while fidelity collapses;
   400/400 braid errors detected. Honest frame: mathematical immunity in exact classical simulation;
   the invariant detects, does not correct; no physical anyons, no hardware.
9. Protein folding: a real physics-informed coarse-grained (Cα-only) energy minimizer. The four
   "quantum" constants (1.2/0.6/0.4/0.8) are hand-set heuristic weights, not fitted. NO validation
   vs PDB structures, TM-score/RMSD, or AlphaFold/CASP — it is a simulator/optimizer, not a validated
   structure-prediction benchmark. A doc's "+68% H-bond" figure is a hardcoded illustrative example
   (no code reproduces it) and must not be quoted as a measured result.
10. Time crystal: reproducible toy Floquet generator (N=10 spins, steps=500, disorder 0.08, jitter
    0.05); the autocorrelation starts near −1 because the first half-period flips every spin against
    the reference. Demonstrator, not a physical time crystal.
11. Stability heatmap: a virtual-superheavy-nucleus toy (Weizsäcker mass formula + shell-correction
    heuristic) scanning Z∈[100,130], A=Z+offset with offset∈[150,220]; stability_score is a weighted
    blend of binding/q-alpha/shell/fission terms. Theoretical exploration, not predictions of matter.
12. Scripts: Linear A / Indus / Voynich work is structure-first only. Measured structural numbers
    exist (e.g., Linear A LZ Δ −7.6% vs Linear B −30%; 81.6% = share of A/I/U among assigned vowel
    syllables, not translation accuracy; Indus H2|H1 2.4127, H3|H2 0.4139, P385 z=9.84). A "Linear
    A ARI ≈ 0.19" figure is NOT found on disk — recorded values are +0.108/−0.002/−0.102 (dataset
    destroyed, not re-measurable). Structure ≠ meaning; no decipherment claimed without a bilingual.
13. Placeholders/limits: validation/ empty; original analysis scripts for Voynich and some visual
    artifacts destroyed (numbers survive only in task records); no CHSH S(t) series is stored in any
    JSON (only per-n maxima); video files exist but lack frame-extraction tooling on this machine.
14. Bottom line: the defensible claim is "exact classical simulation of large-dimension states at
    exponentially compressed memory, with measured non-classical statistics and cited limitations."
    It is NOT a quantum computer, NOT chemistry validation, NOT a decipherment, NOT a validated
    protein-structure predictor, and NOT hardware of any kind.

---

## 0. The single most important honesty boundary (read first)

**(a)** Every "quantum" artifact in this repo is an **exact classical simulation / deterministic
generator** produced by ordinary Python/NumPy math. **No quantum hardware was ever used, run, or
claimed.** All artifacts state this themselves (cited below). Quantum-likeness probes are **statistics
of the generated vectors** — Wigner negativity is "an exact representation change, witness only", CHSH
is "a computed property of state statistics, not a lab loophole-free test" (`quantum_likeness/
likeness_results.json` → `meta.honest`). Anyon braiding is "mathematical immunity proven in exact
classical simulation; no physical anyon realization, no hardware run, no error-free UQC"
(`anyon/anyon_braid_results.json` → `meta.honest`). Any claim that JARVIS is a quantum computer would
be hype; the honest ceiling is **exact classical simulation at exponentially compressed memory**.

## 1. FBSC compressor — what is real and what "MSE=0" means

**Files:** `jarvis/core/compression_specialist.py` (FIXED copy; repo HEAD at origin/main 5d870ec does
NOT contain the log-space fix — the fix is only in jarvis/core/, see jarvis/INDEX.md §Repo-Sync
Warning). Class `FractalBraidSeedCompressor`, methods `reconstruct()` / `_reconstruct_qubit()` /
`_reconstruct_qudit()` / `_safe_compression_ratio()` / `_safe_compression_ratio_qb()`.

- **(a)** 3 real numbers (owner seed `[0.57721, 1.618034, 2.71828]`) are deterministically mapped
  via `_3seed_to_params()` to scale/phase/fold_depth, then an iterative folding loop + deterministic
  braid unitaries + normalization produce a complex amplitude vector of length n (qubit) or n×d
  (qudit). The function is deterministic and reversible — so the amplitudes ARE exactly the seed's
  image; the stored/regenerated representation is O(n) numbers, not 2^n.
- **(a)** "MSE=0 / exact by construction" — see `_reconstruct_qubit()` metrics dict:
  `"reconstruction_mse": 0.0,  # Exact by construction (deterministic reversible)`.
  **Honest caveat (measured):** MSE=0 is *verification against a dense reference state* only at small
  n (n=16 → 65,536 dims, n=32 → 4,294,967,296 dims, both PASS). At n=1024 the dense statevector
  (2^1024 entries) cannot be materialized; there MSE=0 means *construction-exactness* (the formula is
  the state), not an independent dense comparison. Any claim of "verified against the full dense
  vector at n=1024" would be false.
- **(a)** Compression metric (`fbsc_efficiency_report.json`, regenerated 2026-09-24, note says
  "re-verified by fresh run"):
  - n=1024, d=2: total Hilbert dim 2^1024; compression ratio **8.7777985e304× (log10 304.94)**;
    memory **16 KB**; MSE=0; fold factor 67.2665.
  - n=16, d=2: 65,536 dims; ratio 2048×; memory 0.25 KB; MSE=0 (dense-verified).
  - n=16, d=4: 4,294,967,296 dims; ratio 67,108,864×; memory 1 KB; MSE=0 (dense-verified).
  The comparison "vs 64 GiB dense vector" is the honest classical-baseline memory count for 2^36
  entries; for 2^1024 entries the dense baseline is physically impossible, so the ratio is a
  mathematical counting identity (Hilbert dim ÷ stored entries), correct in log space (this is why
  the log10-safe fix matters). **(a measured)** the ratio, **(b interpretation)** its significance.

## 2. Seed optimizer — variational seed search

**Files:** `seedopt/seed_optimizer.py`, `SEED_OPTIMIZER.md`, `seedopt_results.json` (regenerated
2026-09-24). **(a)** CMA-ES winner seed `[4.91885182, 3.90224766, 2.260521]` → combined objective
0.701304 (bio 0.987, coherence 0.026, braid 0.997) vs owner-seed baseline 0.674036; exactness
post-check **MSE=0.0 at n=32 PASS**. Loss landscape sampled 512 pts: mean 0.6529 ± 0.0237, essentially
**featureless** ("landscape featureless at probed scale") — an honest negative for gradient structure.
Objective is a heuristic combination (0.4 bio + 0.3 coherence + 0.3 braid); improvement over owner
seed ≈ +4% on a heuristic, NOT a physical claim.

## 3. Nitrogenase benchmark (Validation Stage 1)

**Files:** `nitrogenase_artifacts/` (corrected set), `nitrogenase_qudit_report.json` (shared root,
regenerated 2026-09-24); fix documented in `DATA_FIX.md` (5^16 → 2^16 dimension bug, approved task
`2124404a`). Key measured values: best_dimension=4 (ququart), effective_qudits=16, coherence 0.8762
(0.9029 after resonance boost), braid_topological_protection 0.9951, bio_resonance_match 0.9515,
**energy barrier 100.7657 kJ/mol**, synthetic catalyst barrier 80.613 kJ/mol, compression 67,108,864×,
MSE=0.0, **time_reversal_fidelity = 0.0606** ← an honest, non-trivial LIMITATION number sitting in the
same report; it is not hidden. **Honest scope:** this is a classical simulation of a simplified
qudit encoding; it is **NOT chemistry validation** (explicit in every artifact and in the validation
task record `fda8f85c`). Pre-fix numbers (96.8918 kJ/mol) survive only in the owner's pre-fix upload
and the reverted repo docs; corrected canonicals are in nitrogenase_artifacts/ + jarvis/nitrogenase/.

## 4. Validation Track 3/3 — DESTROYED on disk (must not be re-quoted as files)

**Status:** `validation/` directory exists but is **EMPTY on disk** — destroyed in the 2026-09-24
disk-full incident (so were the original scripts). **Do not claim these files exist.** The approved
*numbers* survive in team-DB task records `b65a40c6` (Stage 1 NISQ-noise benchmark: owned qudit
simulator collapses at depth 1–6 vs published NISQ fidelities; core depth-independent, MSE=0, ~1 KB vs
64 GiB dense), `cd416d93` (Stage 2 error-bound proof: exactness-by-construction, closed-form O(1)
amplitudes, ≤3-parameter measure-zero cap — cannot represent generic states; parametric crossover vs
dense at n≥6), `af40e730` / `fda8f85c` (Stage 3 blind FeMo-co prediction: freeze hash `37e6c84c…`
reproduced bit-exact, five energies bit-exact, HIT at both prediction levels, explicitly scoped as
**one data point, NOT chemistry validation**). Regeneration is in progress (task `a839d0af`).

## 5. Quantum-likeness probes — measured, with the honest wall built in

**Files:** `quantum_likeness/QUANTUM_LIKENESS.md`, `likeness_probes.py`, `likeness_results.json`
(2026-09-22, theoretical physicist; also mirrored in `qml/` as QUANTUM_LIKENESS_REPORT.md +
likeness_results.json). **meta.honest verbatim:** "classical deterministic generator; probes are
STATISTICS of the generated vectors/unitaries; no hardware; no universal-QC claim; family
<=3-parameter measure-zero in ambient Hilbert space (Thm 3 ERROR_BOUND_PROOF); Wigner is exact
representation change, witness only; CHSH/LHV is a computed property of state statistics, not a lab
loophole-free test."
- **(a)** P1 entanglement (Smid/alpha): FBSC states show volume-law-like entanglement growth
  (n=12: Smid 3.53, α 0.59; n=16: 3.51, α 0.44) vs area-law controls (P1b α≈1/n, D=1).
- **(a)** P2 magic (stabilizer Renyi-2 M2): braided states M2 > 0 at n=4 (0.976) and n=6 (2.238)
  while GHZ/product M2 = 0 — i.e. **a Clifford/stabilizer simulator cannot reproduce these states**
  (GK/easy-simulation route closed, noted in GK_control).
- **(a)** P3 XEB: braid F=1.674 vs product 0.0 vs Haar 0.881 at n=4; braid consistently between
  product and Haar (not Haar-random — honest; it is structured, not Haar).
- **(a)** P4 Wigner: negative qubit Wigner fraction 0.125 (n=3) → 0.223 (n=5); qutrit 0.111–0.284.
  **(b)** negativity is a known non-classicality witness, but **in a classical generator it is a
  computed statistic, not a physical system** (meta.honest says exactly this).
- **(a)** P5 no-LHV/Bell-CHSH: n=2 CHSH **2.825** (violates 2), n=3 CHSH 2.825 + Mermin 1.93, n=4
  CHSH 2.035; `violates=true`, `lhv_excluded=true` in the file. **Fresh n=2 sweep (measured,
  `showoff/showoff_metrics.json` → `chsh_landscape`):** max CHSH over 36 deterministic FBSC seeds =
  **2.828123**, owner-seed CHSH = **2.824953** (both > 2, below Tsirelson 2√2≈2.8284), generated by
  `showoff/generate_showoff.py` `make_chsh()` (Horodecki max + convex-hull LP, LHV excluded). The
  per-seed 36-value array is NOT persisted — only max + owner value. **Honest caveat:** computed on the
  generated state statistics ("exhaustion probe" framing), explicitly NOT a loophole-free lab Bell
  test, and the family is measure-zero (cannot represent generic states). **(b interpretation)** this
  is the citable *definition* of non-classicality in the research program, not a hardware result.
- **(a) CHSH "S(t)" time series — does NOT exist as stored data.** `quantum_likeness/likeness_results.json`
  and `qml/bell_chsh_results.json` store only the final per-n CHSH maxima (2.825 / 2.825 / 2.0351).
  The videos `chatbot/New folder/quion_chshy_timelapse.mp4` and `fbsc_discovery_timelapse.mp4` exist
  (the first is the owner's upload from 2026-09-28), and the generator
  `chatbot/quantacap/src/quantacap/experiments/quion_vizrun.py` writes a per-frame payload whose
  `"S"` field is the **state entanglement entropy** (`entropy_from_mags`), NOT a CHSH value; fidelity
  and magnitudes/phases are stored per frame. A CHSH-over-time series was never persisted. No
  ffmpeg/cv2/imageio exists on this machine, so frames cannot be extracted here to read the plot
  values; do not claim a stored S(t) series exists. (c) If the owner wants S(t) numbers read off the
  video, that requires installing frame-extraction tooling — treat any such readings as plot
  readings, not file-measured data.
- **(a)** P6 MPS/TT challenger: bond dimension D needed for fidelity ≥0.99 grows 16 (n=8) → 32
  (n=12) while FBSC seed reconstructs exactly with O(1) parameters; adversarial-but-classical.
- **(c speculation)** any claim that these probes prove quantum-computational advantage is out of
  scope and is explicitly disclaimed in every artifact.

## 6. Topological fault tolerance (anyon braid experiment)

**Files:** `anyon/ANYON_BRAID_EXPERIMENT.md`, `anyon_braid_sim.py` (regenerated 2026-09-25),
`anyon_braid_results.json`. **(a)** Burau braid invariant distance **exactly 0.0 across 0→40 kinks**
while state fidelity collapses under identical isotopy noise (NISQ-conservative curve); 400 flip
trials: every element-changing braid error detected (invariant changes) — 1.000/400.
**meta.honest verbatim:** "mathematical immunity proven in exact classical simulation; no physical
anyon realization, no hardware run, no error-free UQC; invariant DETECTS, does not CORRECT;
fidelity gate-error rates are RECONSTRUCTED (original lost with validate_benchmark.py)".
**(b interpretation)** mathematical (topological) robustness property of a braid representation —
an interesting classical result; **(c)** any claim of physical anyons or error-corrected computing is
explicitly disclaimed.

## 7. Protein folding engine — EXISTS, real physics-based computation, scoped honestly

**Files (in repo, present):**
- `chatbot/jarvis_quantum_ai_hf_ready/src/multiversal/protein_folding_engine.py` — class
  `ProteinFoldingEngine` with `initialize_extended_chain()`, `energy()` (returns breakdown),
  `metropolis_anneal()`, `save_artifact()`; geometry helpers for bond angle/dihedral/Ramachandran-like
  priors (`_ramachandran_mixture_energy`), Lennard-Jones + Debye-screened Coulomb + hydrophobic terms,
  pivot/crankshaft Monte Carlo moves (`_apply_random_torsion_pivot_move`,
  `_apply_random_crankshaft_move`), consensus swarm move (`_apply_consensus_swarm_move`).
- `chatbot/jarvis_quantum_ai_hf_ready/src/multiversal/multiversal_protein_computer.py`,
  `chatbot/scripts/benchmark_protein_folding.py`, `chatbot/scripts/run_protein_folding_demo.py`,
  `chatbot/MULTIVERSAL_PROTEIN_FOLDING.md`, artifacts in `chatbot/protein_folding_artifacts/*.json`
  (energy traces, e.g. best −1.62, 4 universes, runtime ~3.6 s for an 8-mer).
- **What it is (measured):** a real, physics-informed coarse-grained energy minimizer (monte-carlo /
  simulated annealing, parallel "universes" with different seeds). "Multiversal" = independent seeded
  runs + consensus moves — a parallelism label, honest.
- **Representation (measured):** **Cα-only coarse-grained** — the chain is built from backbone Cα
  positions with phi/psi dihedrals; `bond_length=3.8` is the **CA–CA distance in Å** (comment in
  `FoldingParameters`: "CA-CA distance (Angstrom, typical)"). The Å distances are Cα–Cα spacings of
  this coarse model, not all-atom coordinates.
- **The four "quantum" constants (measured values, provenance honest):** in `FoldingParameters` —
  `quantum_coherence_k = 1.2`, `quantum_phase_k = 0.6`, `topological_protection_k = 0.4`,
  `quantum_delocalization_k = 0.8`, alongside the classical set (`bond_k 50.0`, `angle_k 10.0`,
  `torsion_k 1.5`, `lj_epsilon 0.2`, `lj_sigma 4.0`, `coulomb_k 1.0`, `debye_kappa 0.25`,
  `hydrophobic_k 0.5`, `hbond_k 0.8`, `solvation_k 0.2`, `consensus_k 0.0`). **How picked:** the
  classical terms carry physical-motivation comments ("typical", "screening factor",
  "Ramachandran-like"); the quantum terms carry only generic comments ("Strength of quantum coherence
  effect"). **(b interpretation) the weights are hand-set dimensionless heuristics — there is no
  fitting script, no calibration dataset, and no derivation from first principles in the repo.** The
  functional forms are physically motivated (Gaussian delocalization, phase coupling, log network
  protection) but the coefficients are not evidence-backed numbers.
- **What it is NOT (must state):** there is **no evidence in the engine or its benchmark script of
  comparison against native PDB structures, TM-score/RMSD, or AlphaFold/CASP baselines** — i.e., it is
  a folding *simulator/optimizer*, NOT a validated structure-prediction benchmark. Claiming otherwise
  would be hype. (Skeptic check: search repo for PDB/TM-score/CASP ground-truth before quoting.)
- **The "+68%" claim — provenance check (found on disk, and it is NOT a measured result):**
  `chatbot/QUANTUM_HYDROGEN_BOND_DISCOVERY.md` (also in `chatbot/New folder/`) contains an example
  output block: `Classical H-bond −2.4567 → Quantum H-bond −4.1234 ⭐ +68% improvement!` and totals
  `−15.2345 → −16.7890 ⭐ +10%`. The metric is **relative increase in H-bond energy magnitude** vs the
  engine's own classical H-bond term, and the four example numbers (−2.4567/−4.1234/−15.2345/−16.7890)
  appear **nowhere in any .py file** (verified by grep) — they are hardcoded illustrative values in the
  doc, not a re-runnable run artifact. **(c) Do not quote "+68%" as a measured result.** The same doc's
  header claims "This is REAL physics, not simulation!" and "The Hidden Term That Beats AlphaFold" —
  both are **overclaims by the house standard**: no AlphaFold baseline exists anywhere, and the term is
  a heuristic model term, not validated physics.

## 8. Time-crystal & other visual artifacts

- `chatbot/New folder/timecrystal_autocorr.png` exists (owner upload, committed f556e96).
- Source code exists: `chatbot/quantacap/src/quantacap/experiments/timecrystal/floquet.py` +
  `__init__.py` — i.e., the PNG has a reproducible generator in the repo.
- **Time-crystal parameters (measured, from `floquet.py` `run_time_crystal()` defaults):** `N=10`
  spins, `steps=500`, `disorder=0.08`, `jitter=0.05`, `seed=424242`. There is **no drive-frequency
  parameter** in the model — the drive is the deterministic global flip `global_flip = -1.0 if t % 2
  == 0 else 1.0` (period-2 square-wave drive). **(a) Why the autocorrelation starts near −1:** the
  reference is the all-+1 state (`spins = np.ones(N)`); at t=0 the global flip sends every spin to −1,
  so `autocorr[0] = mean(reference*spins) = −1` before disorder, and ≈ −0.84 with the default
  disorder (8% defects flip to +1). Successive half-periods alternate sign; `detected` is true when
  the dominant FFT peak is within 0.08 of the 0.5 (period-doubling) frequency. Toy model — **not** a
  claim of a physical time crystal.
- **"Stability heatmap" (item 7 of the request) — identified and answered:** the heatmap is
  `chatbot/New folder/virtual_element_heatmap.png`, produced by
  `chatbot/quantacap/src/quantacap/experiments/virtual_element/` (`models.py`, `search.py`,
  `plot.py`). **(a) The stability-score formula** (`models.py` `combined_binding_energy()`):
  `stability_score = 0.55·(binding_per_A / be_norm) + 0.25·q_penalty + 0.15·(0.5 + δ_shell/8) + 0.05·(1 − sf_vulnerability)`
  with `q_penalty = 1/(1+|Q_α|/qalpha_norm)`, where binding_per_A comes from a Weizsäcker liquid-drop
  mass formula + Gaussian shell-correction heuristic (`DEFAULT_WEIZSACKER_PARAMS` a_v 15.75, a_s 17.8,
  a_c 0.711, a_a 23.7, a_p 12.0 MeV; `DEFAULT_SHELL_PARAMS` magic Z 114/120/126, N 184; stability
  norms `be_norm=8.5`, `qalpha_norm=12.0`, SF refs Z=114/N=184). **(a) Scanned range** (`search.py`
  `search_isotopes()` defaults): **Z ∈ [100,130], A = Z + offset with offset ∈ [150,220]** (so N ≈
  150–220). **(b) Honest scope:** the module docstring says these are "semi-empirical mass formulas
  combined with a simple shell-correction ansatz to produce *theoretical* stability estimates …
  not predictions of synthesised matter … offline numerical exploration" — a theoretical toy for
  virtual superheavy nuclei, not measured nuclear physics.
- Similar artifacts in the owner upload (topological_shield_stress_profile.png, stochastic_resonance_
  verification.png, replay_pi_phase_telemetry.png, smm_axis_verification.png, structural_positional_
  heatmap.png, anyonic_logic_fidelity.png, fibonacci_fusion_rules.png) are present on disk; their
  generating scripts were destroyed for some (validation/) or are in quantacap for others — verify
  per-artifact before claiming reproducibility.

## 9. Linguistics track (structure ≠ meaning — hard boundary)

- **Linear A:** `lineara/` present (lineara_xyz, past_pylos, raw, sakamoto_la_analysis) plus committed
  `chatbot/New folder/CORPUS.md` (provenance/license) and `TRANSLATION_PASS.md` (structural/functional
  pass). Measured structural numbers from my engine (task records): LA LZ Δ −7.6% vs LB −30%; sign-value
  overlap 94.3%; libation formula JA-SA-SA-RA-ME 9×, U-NA-KA-NA-SI 6×, A-TA-I-*301-WA 12×; *301 =
  3rd-most-frequent LA token (274). **Boundary (house rule, repeated in every artifact):** structure ≠
  meaning; no decipherment is claimed without a bilingual or identified language. The ≈85–90% figure
  in the plan is a *functional* structural mapping (interpretation), not a linguistic translation claim.
- **The "81.6%" number — exactly what it counts (found on disk):** `chatbot/New folder/TRANSLATION_PASS.md`
  (lines ≈458–479 and 559–566). It is **NOT a translation-accuracy percentage and not a token or
  unique-word count.** It is the share of **A/I/U among assigned vowel syllables of Linear A** after
  transferring one shared sign→value table onto both LA and LB: "LA's vocalism is **A/I/U-dominated
  (81.6% of vowel syllables)** while LB Greek is **O/E-dominated (54.6%)**". The measured table
  (row = vowel class, LA transferred vs LB Greek control): A **39.8%** vs 25.5%, E 14.0% vs 23.0%,
  I **24.4%** vs 13.3%, O **4.4%** vs 31.6%, U **17.5%** vs 6.6%, E+O **18.4%** vs **54.6%**
  (A+I+U = 39.8+24.4+17.5 = 81.7% ≈ 81.6). The value table itself is a borrowed hypothesis (`(b)`:
  "the value table is a hypothesis, though both corpora share it"); the contrast is a measured
  structural statistic; its Anatolian/Luwian reading is explicitly "consistent with … but not proof"
  and "the wall is external evidence, not computation". Cite it exactly as a vocalism contrast,
  never as "81.6% translated".
- **qml_results.json — ARI, and the two registers (found on disk):** `jarvis/qml/qml_results.json`
  (`meta.dataset` = "Linear A top-60 formulas (real, 20 register-labeled)") and
  `jarvis/qml/QLM_FRAMEWORK.md` (line ≈44: stratification "accounting R1 vs ritual R3"). The 20
  register-labeled inscriptions are **real Linear A formulas**, labeled by corpus-defined register
  (administrative R1 vs ritual R3); the seeds embed sign frequency + positional bias designed to
  separate those strata. Measured (recorded, with an explicit regeneration note that the dataset was
  destroyed in the 2026-09-24 incident and these are **NOT re-measurable**): `struct_seed_ari` =
  **+0.108**, `hash_seed_ari` = **−0.002** (null control), `bag_of_signs_ari` = **−0.102**,
  same-register pair distance 0.862, cross-register 0.902. `honest_limits`: "recovery
  partial/lopsided: accounting clusters via shared -RO suffix", "ritual family does NOT cluster
  (frequency+positional seeds miss family membership)", "weaker than n-gram/LZ; complementary, not
  competitive", "no meaning/decipherment claim". **The lead's "Indus/CISI ARI ≈ 0.19" figure is NOT
  found anywhere on disk** (grep over all team .md/.json): the only ARI values recorded are the
  Linear A ones above. Do not quote 0.19.
- **Indus:** `indus/` present with rebuilt harness (`NORMALIZE_RB.py`, `h1h4_crossvalidate.py`,
  `indus_corpus_clean.tsv`, `h1h4_crossvalidate.json`, INDUS_H1H4_CROSSVALIDATE.md — my 2026-09-28
  rebuild, PR #141). Cross-validation re-derived from raw corpus: H2|H1 2.4127 (approved 2.413),
  H3|H2 0.4139 (0.414), H1 6.2859, H2 8.6986, H3 9.1125, bigrams used 551, P385 initial z=9.84,
  P385→P122 enrichment 13.31× (z=18.8), far-repeated bigrams 0 (no long-range chains). **Gate:** full
  Mahadevan concordance is NOT digitized/available; full-corpus run remains gated (task b5c9f3c9).
- **Voynich:** `voynich/data` + `voynich/raw_repos` exist on disk; prior structural claims (5 probes
  inconsistent with random scribble; phonetic reading impossible without a bilingual) are preserved in
  task records — the original analysis scripts were destroyed in the incident; **quote numbers only
  via task records, not by claiming the scripts exist**.
- **Honest summary for the skeptic:** the program's stated position is **"structure-first analysis
  maps the skeleton of an unknown writing system without a bilingual, and does NOT claim decipherment"**.
  Any stronger claim attributed to the team is misquoted.

## 10. Destroyed-on-disk inventory (do NOT claim these files exist)

| Path | Status | Provenance of numbers |
|---|---|---|
| `validation/` (all 8 files) | EMPTY on disk — destroyed 2026-09-24 | task records b65a40c6, cd416d93, af40e730, fda8f85c |
| Original `qml/`, `anyon/` (pre-incident) | replaced by regenerated sets (2026-09-25) | PRs #139/#140 (QLM/anyon, trainer) |
| `jarvis/trainer/` (original) | rebuilt by owner regen | PR #139, task 97c482eb |
| `fbsc_efficiency_report.json`, `seedopt_results.json` (originals) | regenerated 2026-09-24 | fresh runs, notes in files |
| base `owner_quantum_seed.json` | reconstructed from d2 (values identical, metadata differs) | tag in file + jarvis/INDEX §1 |
| voynich/indus/lineara analysis scripts (SYNTHESIS.md etc.) | destroyed; raw data survives | task records 4d123916…7b903a12 |

## 11. Where each verified number lives (quick citation map)

1. Compression/MSE semantics — `jarvis/core/compression_specialist.py` (`reconstruct`, `_reconstruct_qubit`, `_safe_compression_ratio*`)
2. 8.78e304× / log10 304.94 / MSE=0 / 16 KB @1024 — `fbsc_efficiency_report.json` rows[0]
3. CMA-ES seed 0.701304, MSE=0 n=32, featureless landscape — `seedopt/seedopt_results.json`, `SEED_OPTIMIZER.md`
4. Nitrogenase 100.7657 / 80.613 / coherence / time_reversal 0.0606 — `nitrogenase_qudit_report.json`
5. Likeness probes + honest wall — `quantum_likeness/likeness_results.json` (meta.honest + P1–P6); CHSH per-n 2.825/2.825/2.0351 also in `qml/bell_chsh_results.json`; fresh 36-seed sweep max 2.828123 / owner 2.824953 — `showoff/showoff_metrics.json` `chsh_landscape` + `showoff/generate_showoff.py` `make_chsh()`
6. Anyon invariant-distance 0.0 / detects-not-corrects — `anyon/anyon_braid_results.json` (meta.honest)
7. Protein folding engine + constants (1.2/0.6/0.4/0.8) + Cα-only — `chatbot/jarvis_quantum_ai_hf_ready/src/multiversal/protein_folding_engine.py` `FoldingParameters`; "+68%" demo values — `chatbot/QUANTUM_HYDROGEN_BOND_DISCOVERY.md` (NOT measured)
8. Timecrystal PNG + floquet source (N=10, steps=500, disorder 0.08, autocorr −1 origin) — `chatbot/quantacap/src/quantacap/experiments/timecrystal/floquet.py`
9. Stability heatmap formula + Z∈[100,130], N≈150–220 — `chatbot/quantacap/src/quantacap/experiments/virtual_element/{models,search,plot}.py`
10. Linear A 81.6% A/I/U vocalism (vowel-syllable share) — `chatbot/New folder/TRANSLATION_PASS.md` lines ≈458–479; QLM ARI +0.108/−0.002/−0.102, 20 register-labeled, R1/R3 — `jarvis/qml/qml_results.json` + `jarvis/qml/QLM_FRAMEWORK.md`; "ARI 0.19" NOT on disk
11. Indus H1–H4 numbers/method — `indus/h1h4_crossvalidate.json` + INDUS_H1H4_CROSSVALIDATE.md
12. Script corpora provenance — `chatbot/New folder/CORPUS.md`
13. CHSH timelapse videos — `chatbot/New folder/quion_chshy_timelapse.mp4`, `showoff/fbsc_discovery_timelapse.mp4`; generator `chatbot/quantacap/src/quantacap/experiments/quion_vizrun.py` (per-frame "S" = state entropy, not CHSH)