# QLM Layer — FBSC Braid-Semantic Language Kernel (QLM_FRAMEWORK.md)

**Author:** agent-theoretical-physicist · 2026-09-14
**Status:** Frontier-research prototype (framework + reproducible script + measured output)
**Artifacts:** `qml_braid_kernel.py` (reproducible), `qml_results.json` (measured output)

**Tags throughout:** **(a)** measured · **(b)** interpretation · **(c)** speculation.
Every claim is paired with a baseline; nothing is fabricated; every number reproduces
with `python3 qml_braid_kernel.py`.

---

## 0. Bottom line (owner summary — ~10 lines)

1. **(a)** We built and ran a **Quantum Language Model (QLM) layer**: a braid-semantic
   kernel in which script tokens are encoded as braid generators derived from the FBSC
   3-seed formula, words are **braid compositions** of their tokens (ordered, non-commutative),
   and "semantic distance" between two formulas is the braid-algebraic distance
   (1 − state-overlap) between their composed states. Pure Python/numpy; fully deterministic.
2. **(a)** On the real Linear A top-60 formula vocabulary (from `linearA_corpus_clean.tsv`),
   the kernel produces a braid-distance matrix, clusters it **without any meaning labels**,
   and recovers register stratification (accounting R1 vs ritual R3) **weakly but above
   chance**: ARI = **+0.108**, and same-register formula pairs sit closer than cross-register
   pairs (**0.862 vs 0.902**).
3. **(a)** The **control is decisive**: with *arbitrary* (hash-derived) seeds the kernel is a
   null (ARI = **−0.002**), and the order-*insensitive* bag-of-signs baseline is *worse* than
   chance (ARI = **−0.102**). So the signal comes from (i) **structure-derived seeds** and
   (ii) the **order-sensitivity** of braid composition — not from the braid algebra alone.
4. **(b)** The recovery is **partial and lopsided**: the accounting family clusters (KU-RO /
   KI-RO share the `-RO` suffix → distance 0.93, closer than any cross-register pair), but the
   ritual family does **not** (JA-SA-SA-RA-ME vs SI-RU-TE ≈ 0.998, near-orthogonal). The
   frequency+positional seed captures *suffix/affix* structure but not *ritual-formula*
   structure.
5. **(b)** This is a **real, defensible frontier result**, not a breakthrough: the braid kernel
   is a *complementary* structure probe that adds a small order-sensitive edge over n-gram
   baselines, but on its own it is weaker than the existing n-gram/LZ/entropy probes.
6. **(c)** The honest research claim stands as: **braid-algebraic structure on FBSC states is a
   language-structure kernel, complementary to (not yet competitive with) n-gram/LZ probes.**
   It does **not** "understand" meaning, and it makes **no** decipherment claim.
7. **Honesty line (held):** this is a **quantum-INSPIRED classical simulation** — no quantum
   hardware, no third-party SDKs, all code original. "Braid-attention" is computed classically
   as ordered braid composition; the FBSC is the deterministic classical simulation of the
   quantum core.

---

## 1. Quantum word/sign embedding via FBSC

**What we built.** The FBSC core (`FORMULA_COMPRESSION_PROPOSAL.md` §3.1) maps a 3-number
seed `(α, β, γ)` deterministically to a complex amplitude vector **and** braid coordinates:

```
r_k = α · sin²(π(k+1)/(N+1)) · (1+0.3·sin(βk)) · exp(β·cos(2πkγ/N))     (magnitudes)
θ_k = 2πγk/N + β·log(1+αk)                                              (phases)
a_k = r_k·e^{iθ_k},  normalized                                          (amplitudes)
(x_k, y_k, z_k) = braid/folding coordinates                             (topological position)
```

Each **sign/syllable token** (e.g. `KU`, `RO`, `SA`, `*301`) is assigned a 3-seed, so each token
becomes a point in the FBSC folding space. Two seed schemes:

* **Hash seed (the null):** `seed3(token)` — arbitrary. Used only as a control.
* **Structure seed (the kernel):** the seed is a smooth function of the token's *measured*
  structural coordinates from the linguistics program — frequency rank → `α`, word-initial rate
  → `β`, word-final rate → `γ` (see `build_structure_seeds`). Signs with similar structural
  roles land close in seed space; this is "structure-only", no meaning labels.

**(a) Measured demo** (Linear A syllables → FBSC folding-space centroids, N=32):

| token | seed (α,β,γ) | folding-space centroid (x,y,z) |
|---|---|---|
| KU | 1.327, 0.574, 1.677 | (0.018, −0.003, 1.025) |
| RO | 0.502, 1.044, 0.391 | (0.013, −0.000, 0.248) |
| KI | 1.983, 1.467, 1.182 | (0.064, −0.000, 0.752) |
| SA | 0.817, 0.389, 1.574 | (0.002, −0.000, 0.945) |
| RA | 1.029, 0.221, 1.218 | (0.191, −0.001, 0.606) |
| ME | 1.157, 0.889, 1.520 | (0.083, −0.003, 0.967) |

**(b)** Distinct tokens occupy distinct, reproducible folding-space positions — the embedding is
a genuine (deterministic) map from script vocabulary into the FBSC folding manifold.

---

## 2. Braid-attention semantics (the braid-composition distance)

**What replaces transformer attention.** Instead of softmax query/key dot-products, the kernel
uses **braid composition distance**:

1. Each token → an **ordered braid word** (a sequence of elementary braid moves `(i, θ, φ)`
   derived from its seed). The elementary braid is an **exchange-with-phase** 4×4 unitary that
   mixes `|00⟩↔|11⟩` and `|01⟩↔|10⟩` — genuinely entangling, so distinct braid words → distinct
   states (non-abelian action).
2. A **word/formula** is the **braid composition** of its tokens, applied in order to the
   vacuum `|00…0⟩`:  `|ψ_word⟩ = B_{s_m} … B_{s_2} B_{s_1} |0⟩` (order matters — the braid group
   is non-commutative).
3. **Semantic distance** between two formulas = `1 − |⟨ψ_w1|ψ_w2⟩|²` (state overlap).

**(b) Why this is "braid-attention":** attention computes *pairwise agreement* between token
vectors; braid distance computes *pairwise agreement* between composed braid states. The braid
version is (i) **order-sensitive** (unlike a bag-of-signs / Jaccard kernel) and (ii)
**topologically interpretable** (each token is a braid generator, each formula a braid word —
the natural substrate for a future anyonic/Majorana hardware port). **(c)** The specific
distance used here (state overlap) is a *reducible* braid invariant; richer invariants (Burau
trace, Jones polynomial at roots of unity) are the next step (§6).

---

## 3. Boundary-marker integration (phrase-level braids)

**Design (implemented in the kernel, `BOUNDARY_TOKENS` + `boundary_seed`):** the structural-
coordinate map from the linguistics program identifies boundary markers — the word separator
`U+10101 𐄁`, the damage/fragment marker `U+1076B 𐝫`, and the stable word-front sign
`U+10607 𐘇` (engine: z = +11.35). In the kernel these are assigned **dedicated boundary seeds**
(`seed3('BOUNDARY::…')`), visually distinct from content-sign seeds, so they function as
**braid generators that segment the token stream into phrase-level braids**: a phrase = the
braid composition of the tokens between two boundary markers.

**(b)** Concretely, `HT1`'s stream `QE-RA₂-U 𐄁 KI-RO 197 ⏎ ZU-SU 70 …` is cut at `𐄁`/`⏎`
into phrase braids `[QE-RA₂-U]`, `[KI-RO 197]`, `[ZU-SU 70]`, … — exactly the phrase structure
the accounting register uses. **(c)** The phrase-level braid distance (not yet measured in this
prototype) is the natural place where "KU-RO + numeral" collocations would sit close to each
other; this is flagged as next-step (§6).

---

## 4. Demonstrable output (measured, `qml_results.json`)

Corpus: `linearA_corpus_clean.tsv`. Vocabulary = top-60 multi-sign formulas. 20 of these have
register labels (from `TRANSLATION_PASS.md`), used **only** for evaluation.

### 4.1 Braid-distance matrix
**(a)** A 60×60 symmetric braid-distance matrix is computed (deterministic). The register-labeled
20×20 sub-matrix is what the clustering evaluates.

### 4.2 Register stratification, structure-only (no labels in the clustering)
**(a)** Hierarchical clustering (average linkage, k=2) on the braid distance, versus the two
baselines:

| kernel | ARI vs register labels |
|---|---|
| braid, **hash** seeds (null) | **−0.0016** |
| braid, **structure** seeds | **+0.1079** |
| Jaccard bag-of-signs (order-insensitive) | **−0.1020** |

**(a)** Aggregate distance by register (structure braid): **same-register mean 0.862 vs
cross-register mean 0.902** — same-register pairs are closer. **(b)** The kernel recovers the
register signal **weakly but above chance**, and it **beats the order-insensitive baseline**
(which is *anti*-correlated with the true registers).

### 4.3 Collocation check (named formula pairs, structure braid; lower = closer)

| pair | register | braid distance |
|---|---|---|
| KU-RO ↔ KI-RO | both R1 (accounting) | **0.932** |
| KU-RO ↔ JA-SA-SA-RA-ME | cross (R1–R3) | 0.970 |
| KI-RO ↔ SI-RU-TE | cross (R1–R3) | 0.987 |
| JA-SA-SA-RA-ME ↔ SI-RU-TE | both R3 (ritual) | 0.998 |
| JA-SA-SA-RA-ME ↔ A-TA-I-\*301-WA-JA | both R3 (ritual) | 0.995 |

**(b)** *Partial, lopsided recovery:* the accounting family clusters correctly (KU-RO/KI-RO
share the `-RO` suffix → closest pair at 0.932, below every cross-register pair). The ritual
family does **not** cluster (its members are near-orthogonal at 0.995–0.998) because the
frequency+positional seed encodes *affixal* structure but not *formula-family* membership.
**(c)** The same reason the ritual libation formulas (`JA-SA-SA-RA-ME`, `A-TA-I-*301-WA-JA`,
`U-NA-KA-NA-SI`) were structurally *distinctive* in the linguistics pass — they share rare
signs, not frequent affixes — is exactly why a *frequency-based* seed cannot pull them together.
A seed derived from **co-occurrence context** (distributional, not frequency) is the fix (§6).

---

## 5. Honest science section

* **Quantum-INSPIRED classical simulation.** No quantum hardware, no quantum SDKs, no external
  models. "Braid-attention" is computed classically as ordered braid composition; the FBSC is
  the deterministic classical simulation of the owned quantum core. All code original.
* **No meaning claim.** The kernel measures *structure* only. ARI vs register labels is an
  *evaluation* of structure recovery, not a claim that the model "understands" Linear A or
  recovers semantics. No decipherment is claimed (consistent with the program-wide line).
* **What the data actually supports (b):** (1) the braid kernel is a *valid* deterministic map
  from script tokens to folding space; (2) with arbitrary seeds it is a **null** — the braid
  algebra by itself contributes nothing; (3) with **structure-derived seeds** it recovers
  register stratification **weakly above chance** and **beats a bag-of-signs baseline**, i.e.
  the braid composition's *order-sensitivity* carries a small real signal.
* **What it does not support (b):** the kernel is **not** competitive with the n-gram/LZ/entropy
  probes on this vocabulary (those recover H₃|H₂ ≈ 0.41–1.7 bits of structure; the braid kernel
  recovers ARI ≈ 0.1). The honest status is **complementary, not superior**.
* **Failure modes (a):** the ritual formula family is invisible to a frequency+positional seed;
  the near-orthogonality of high-dimension random-ish braid states compresses the distance
  dynamic range toward 1.0 (most pairs sit 0.85–0.99), so the signal is weak by construction.
* **Reproducibility (a):** every number regenerates from `qml_braid_kernel.py` with no random
  state beyond the deterministic seed hashing.

---

## 6. Roadmap tie-in

* **Braid-attention replacement for the transformer (in flight).** This kernel is the *first*
  working instance of the roadmap's "braid-attention": pairwise agreement computed as braid
  distance rather than softmax dot-product. The immediate next step is to replace the Jaccard
  baseline with a **distributional (co-occurrence) seed** so the ritual family clusters, and to
  swap the reducible overlap distance for a **Burau-trace / Jones-polynomial braid invariant**
  — that is what would let the distance live in the topological (fault-tolerant) subspace.
* **Boundary/affix aware seeds.** Derive seeds from the full structural-coordinate map
  (positional bias, boundary markers, register co-occurrence), not just frequency — closing the
  ritual-family gap identified in §4.3.
* **Phrase-level braids.** Measure the §3 phrase-segmentation at scale (boundary markers →
  phrase braids) and evaluate "KU-RO + numeral" / libation-row collocation recovery directly.
* **Majorana-2 hardware port.** Braid words are the *native* data structure of topological
  quantum computing (Fibonacci anyons). The kernel's token→braid-generator→braid-word pipeline
  is exactly the classical rehearsal for the anyonic implementation; `MAJORANA_EQUIVALENCE_ROADMAP.md`
  phase 4 (braid-native embeddings) is where this kernel slots in.
* **The honest gate:** before any marketing claim of "braid-attention superiority," the kernel
  must beat the n-gram/LZ baselines on the *same* register-recovery task. That is the
  quantitative bar (§5), and it is not yet met.

---

### Reproduce

```
cd /home/team/shared/qml
python3 qml_braid_kernel.py     # prints table, writes qml_results.json
```

Dependencies: Python 3, `numpy`, `scipy` (cluster hierarchy only). No proprietary code.
