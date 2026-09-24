# Linear A & Linear B Corpus — Provenance & License (CORPUS.md)

**Status:** foundation deliverable · **Owner:** scientific-engineer · **Date:** 2026-09-12
**Scope:** machine-readable Linear A corpus (undeciphered target) + Linear B corpus
(deciphered positive control), with a single-reviewable, auditable transform.

---

## 1. Chain of custody (Linear A)

| Step | What | Where | Notes |
|---|---|---|---|
| 1 | Primary scholarly edition | **GORILA** — Godart, L. & Olivier, J.-P., *Recueil des Inscriptions en Linéaire A* (Études Crétoises 21, I–V), École Française d'Athènes, 1976–1985 | Digitized volumes hosted at cefael.efa.gr (the explorer's README cites 1970, the date of the project's start; the printed volumes appeared 1976–85). **© École Française d'Athènes** — rights declared per-record in the dataset. |
| 2 | Machine tabulation (word breaks, ideograms, numerals) | **George Douros** — "Linear A" Unicode fonts + inscriptions spreadsheets, users.teilar.gr/~g1951d | Distributed free for scholarly use; this is where word segmentation comes from. **Verify exact reuse terms with legal before commercial use.** |
| 3 | Explorer + JSON corpus | **mwenge / lineara.xyz** (repo `mwenge/lineara.xyz`, a.k.a. LinearAExplorer) — "A tool for exploring the Linear A corpus" | Produces `LinearAInscriptions.js` (1 record per inscription) from Douros' CSV + GORILA image scans. **No LICENSE file in this repository** (GitHub API license endpoint returns 404). |
| 4 | Extraction into JSON | compression specialist, `raw/parse.js` (`eval` of the JS map → `raw/lineara.json`) | Byte-identical data, reformatted; `parse.js` retained for audit. |
| 5 | Clean TSV (this task) | `NORMALIZE.py` → `linearA_corpus_clean.tsv`, `linearA_sign_inventory.tsv`, `lineara_stats.json` | No re-encoding; see GROUND_RULES.md. |

**Raw input integrity**
- `raw/lineara.json` — 1,721 records — SHA-256 `6b17065f5fda215de08a1d878831592f7d97086baf892694a839ca961adae5a7`
- (both hashes re-recorded in `lineara_stats.json` under `raw_inputs`)

## 2. Chain of custody (Linear B — deciphered control)

| Step | What | Where | Notes |
|---|---|---|---|
| 1 | Scholarly editions | Knossos: **CoMIK** (Killen & Olivier, *Corpus of Mycenaean Inscriptions from Knossos*); Pylos: ***The Palace of Nestor* / Pylos tablet publications** (Blegen & Rawson, University of Cincinnati); plus Thebes, Mycenae, Tiryns corpora | **© University of Cincinnati** — the dataset's dominant per-record `imageRights` value (986 Pylos records). |
| 2 | Explorer + JSON corpus | **mwenge / linearb.xyz** (sibling tool, same author/schema as lineara.xyz; the old `LinearBExplorer` repo redirects here) | Produces `LinearBInscriptions.js`. **No LICENSE file** (404 on GitHub license endpoint). |
| 3 | Extraction into JSON | compression specialist, `raw/parse.js` → `raw/linearb.json` | |
| 4 | Clean TSV (this task) | `NORMALIZE.py` → `linearB_corpus_clean.tsv`, `linearB_sign_inventory.tsv` | |

**Raw input integrity**
- `raw/linearb.json` — 5,832 records — SHA-256 `aba865cc101a83f6a1c29f3ee619814415cc44c838fd90d272946d411ac00b88`

## 3. Corpus contents & verification (independent, from the clean files)

### Linear A (target — undeciphered)
- **1,721 inscriptions** (site split: Haghia Triada 1110, Khania 226, Phaistos 66, Knossos 59, Zakros 53, Palaikastro 25, Malia 22, Thera 18, + smaller sets)
- **8,967 Linear A sign tokens** from **322 distinct signs** (Unicode block U+10600–U+107FF; Unicode encodes ~353 LA signs)
- **2,049 Aegean-number tokens** (30 distinct, U+10100–U+1013F)
- **1,315 distinct clean word tokens** (Douros segmentation; 4,640 tokens) — an unusually compact lexicon, consistent with a partly formulaic/administrative corpus
- 1,712 / 1,721 records carry Douros transliteration with word breaks; all carry `support` (object type); 592 have scribe; 1,063 have findspot
- Verification check: 1 stray character in the Linear B block (U+10041) inside one LA record — kept verbatim, flagged in sign inventory

### Linear B (positive control — deciphered Mycenaean Greek)
- **5,832 inscriptions** (site split: Knossos 4223, Pylos 986, Thebes 363, Mycenae 87, vases from Thebes/Tiryns/Khania, …)
- **48,874 Linear B sign tokens** from **88 distinct signs** (U+10000–U+1007F) — matches the ~87–88-sign syllabary
- **6,210 distinct clean word tokens** (37,226 tokens), plus English `translated_words` per record
- Median 8 words/record; every record has a Unicode sign string + syllabic transliteration + (mostly) English gloss

### Sanity notes
- LA sign-token average ≈ 5.2 signs/inscription vs LB ≈ 8.4 — consistent with known corpus profiles (many short LA ritual/administrative labels; LB tablets with longer formulaic lists).
- LB distinct-sign count (88) ≈ the deciphered syllabary size ⇒ the LB arm is a reliable *known-truth* benchmark for any method (entropy, motif, graph, compression) that we also run on LA. Any claim made on LA that contradicts the same method's LB behavior should be treated as suspect.

## 4. License status — READ THIS

1. **Neither upstream repo (lineara.xyz / linearb.xyz) carries a software license.** The data files are republished inside them without an explicit license grant.
2. The underlying **transcriptions** are scholarly editions whose publishers retain copyright (**© École Française d'Athènes** for GORILA; **© University of Cincinnati** for Pylos; **Hallager, Boulotis, Karnava, Ph Saperstein** for specific photographs as recorded per-record). Douros' tabulation is distributed free for scholarly use on his site.
3. **What we are using:** machine-readable *transcription data* (sign sequences, transliterations, translations) for research/statistical purposes — not the copyrighted photographs. Transcriptions of ancient texts are widely treated as factual data, but commercial redistribution rights are **not self-evident**.
4. **Action required from legal manager:** audit GORILA (EFA) and Douros reuse terms before any public/commercial release of the corpus files; if needed, (a) cite rather than ship the corpus, or (b) restrict the corpus to internal research as we did for the Indus subset. The scripted derivation chain (steps above) is the audit trail; the raw files are untouched.

## 6. Expansion-scan additions (2026-09-12, extended-translation-anchor task)

Results of the maximum-effort hunt ordered by the lead; **coverage correction included**.

### 6.1 Coverage audit (corrects the "895 without transcription" figure)
- Direct audit of raw/lineara.json: **only 13/1,721 records lack any sign text**
  (sample: HTW231d–f, HTWc3023, KHZc106, MOZb2?, MOZb3?, PETSWc) and **9 lack
  transliteration**. The "895" figure = records missing the *redundant flattened*
  `transcription` field only; their `parsedInscription` (line-structured sign text) is
  present. Treat 1,708 records with text as the effective corpus; the 13 text-less
  records are kept as rows with empty sign_seq so the id space is complete.

### 6.2 Primary-source anchor (GORILA PDFs, in-repo)
`lineara_xyz/papers/` contains **GORILA-Vol1–5.pdf** (École Française d'Athènes)
plus per-inscription facsimile PDFs (e.g. HTZf163.pdf, KNZa*, IOZa15.pdf…) and
Carratelli 1945 (Monumenti Antichi). These scans are the **double-check anchor** for
any transcription (GROUND_RULES.md §7). No PDF text layer → checks are manual;
electronic extraction unavailable in this environment.

### 6.3 PAST — Pylos archival dataset (Linear B control, independent source)
- Repo: `aschimmenti/PAST-PylosArchiveSignsAndTablets` (cloned to `past_pylos/`).
- `past_pylos/data/Pylos.json`: **1,077 Pylos tablets, 2,982 tokens**; schema
  `{"PY 1178 Aa": ["MUL 3"], "PY 1180 Aa": ["puro", "miratija MUL"], ...}`
  (syllabogram words + ideograms + numerals, per-line arrays). Per-sign PNG crops in
  `past_pylos/data/<series>_<n>/` (e.g. `LB_Aa_1178_r1_1_102.png` = row 1, sign 1,
  sign #102) — image-level ground truth for sign recognition.
- License: **no LICENSE file found in repo** — flag for legal; academic project.
- Use: a second, independent Pylos control alongside raw/linearb.json (cross-validate
  the shared Pylos subset).

### 6.4 Other surveyed sources (2026-09-12)
- `sakamoto6000-png/linear-a-structural-analysis` — MIT-licensed code repo; **no
  corpus data** (README-only clone; independent LA analysis code, not new texts).
- John Younger's "Linear A texts in phonetic transcription" — **unreachable** from this
  environment (external academic hosts blocked); referenced as the canonical secondary
  transcription for future verification.
- DĀMOS (Database of Mycenaean at Oslo) — no public bulk download / GitHub arm found;
  the mwenge linearb.xyz JSON + PAST Pylos.json stand in as the control corpora.
- Cypro-Minoan — see CYPRO_MINOEAN.md: **no public machine-readable corpus exists**;
  gap recorded with signary (Unicode U+12F90–U+12FFF, 109 signs; Noto OFL font).

### 6.5 New/updated files (this expansion)
```
ARKALOCHORI_AXE.md   — axe records, I-DA-MA-TE / MI-DA-MA-RA₂ anchors, verification plan
CYPRO_MINOEAN.md     — bridge rationale, acquisition attempt, honest gap record
GROUND_RULES.md      — +§6 sign-function conventions, +§7 double-check protocol
CORPUS.md (this)     — coverage correction, new sources
README.md            — expanded inventory
past_pylos/          — PAST Pylos dataset clone (independent LB control)
sakamoto_la_analysis/— surveyed repo (no data)
```

## 5. File inventory (core deliverable)
```
raw/lineara.json, raw/linearb.json           upstream JSON (untouched, hashed)
raw/LinearAInscriptions.js, LinearBInscriptions.js   upstream JS (source of the JSON)
raw/parse.js                                 compression specialist's eval-to-JSON
NORMALIZE.py                                 auditable raw→clean converter
linearA_corpus_clean.tsv                     id, site, findspot, context, support, scribe,
                                             imageRights, n_la_signs, n_nums, n_lines,
                                             sign_seq (Unicode), word_seq_source (Douros),
                                             words_clean (derived), n_words_clean
linearB_corpus_clean.tsv                     id, site, label, context, scribe, imageRights,
                                             n_lb_signs, n_nums, n_lines, sign_seq,
                                             word_seq_source, words_clean, n_words_clean,
                                             translated_words (English gloss)
linearA_sign_inventory.tsv / linearB_sign_inventory.tsv   codepoint, glyph, block, count
lineara_stats.json                           verified aggregate stats + input hashes
README.md, GROUND_RULES.md, RESEARCH_BRIEF.md
```
