# Arkalochori Axe — Translation-Anchor Documentation (ARKALOCHORI_AXE.md)

The lead's expansion asks for our single best "bilingual-adjacent" artifact. Status:
**fully present in the machine-readable corpus** plus a primary-source PDF anchor in the
repo. Everything below is either verbatim from the corpus or explicitly flagged as
scholarly hypothesis.

## 1. The artifact
- Bronze votive **double axe** from the **Arkalochori cave** (central Crete), now in the
  Heraklion Archaeological Museum; MM III–LM I horizon (≈ 1700–1450 BC).
- Inscribed on **both faces**, written **boustrophedon** (line direction alternates) —
  one of the few Linear A objects where directionality is visually demonstrable.
- GORILA catalogue number: **AR Zf 1** (Z = votive/metal object class). In the Douros
  tabulation the two faces are split into corpus IDs **ARZf1** and **ARZf2**; a second
  Arkalochori metal object with a two-word formula is **ARKHZf9**.

## 2. Machine-readable records (verbatim from raw/lineara.json → clean TSV)

| ID | support | sign sequence (Unicode) | Douros transliteration | scribe |
|---|---|---|---|---|
| ARZf1 | Metal object | `𐘚𐘀𐙁𐘃` | **I-DA-MA-TE** | AR Scribe 1 |
| ARZf2 | Metal object | `𐘚𐘀𐙁𐘃𐝫` (𐝫 = U+1076B) | **I-DA-MA-TE** (+ trailing sign) | AR Scribe 2 |
| ARKHZf9 | Metal object | `𐘱𐘸𐘤𐘸𐘯𐄁𐘻𐘀𐙁𐘽𐄁` (𐄁 = divider) | **JA-KI-SI-KI-NU 𐄁 MI-DA-MA-RA₂ 𐄁** | — |

Context fields: ARZf1/ARZf2 have empty findspot/context; ARKHZf9 `context: LMI`.

## 3. Why this is our strongest anchor
- **Onomastic candidate**: the transliteration **I-DA-MA-TE** (ARZf1/2) is the classic
  reading discussed in the literature as a possible divine-name / dedication formula
  (the proposal connects it with the goddess of **Mt Ida**, "*Ida Mater*"; also seen as
  a personal name sequence). ⚠️ **Hypothesis** — a proposed reading, not an accepted
  decipherment. Cite before asserting.
- **Shared stem**: ARK HZf9's **MI-DA-MA-RA₂** shares the `DA-MA-` sequence with
  I-DA-MA-TE. Corpus-observed pattern; a natural first target for the quantum-structure
  motif pass (does the braid/compression machinery cluster these two inscriptions?).
- It is a **votive object** (not administrative tablet) — relevant contrast case: the
  libation/votive register vs the HT tablet register.
- **Boustrophedon + metal support** makes it a special-format stress test for any
  sequence model (line direction is not uniform).

## 4. Verification / double-check protocol
- Primary anchor on disk: **`lineara_xyz/papers/GORILA-Vol5.pdf`** (44 MB scan of
  GORILA V; AR Zf 1 is published in GORILA V) and per-inscription facsimiles in
  `lineara_xyz/papers/` (e.g., `HTZf163.pdf` demonstrates the Zf-series layout).
- ⚠️ **Coverage caveat (honest)**: literature commonly describes the axe's full text as
  ~15 signs over the two faces; the corpus's two faces carry only 4+5 signs. Either the
  axe faces carry short sequences (with the famous longer text belonging to another
  votive — the cave produced several inscribed metal objects, incl. ARKHZf9) or the
  Douros tabulation is partial for this object. **Do not use the axe's sign count as a
  corpus-quality claim until checked against GORILA V page for AR Zf 1** (manual check;
  no PDF text layer available on this machine).
- Any scholarly quote of "I-DA-MA-TE" on the axe must carry the caveat above.

## 5. Deliverable status
| Item | Status |
|---|---|
| Machine-readable sign sequences | ✅ in `linearA_corpus_clean.tsv` (rows ARZf1, ARZf2, ARKHZf9) |
| Word / formula segmentation | ✅ via Douros transliteratedWords (I-DA-MA-TE; JA-KI-SI-KI-NU / MI-DA-MA-RA₂) |
| Primary-source anchor | ✅ GORILA-Vol5.pdf in `lineara_xyz/papers/` |
| Facsimile | ⏳ inside GORILA V scans (no standalone PDF found) |
| Independent second transcription | ❌ not found (Younger's site unreachable from this environment; GORILA manual check pending) |