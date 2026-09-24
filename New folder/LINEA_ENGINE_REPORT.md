# Linear A — Statistics + FBSC/LZ Compression vs Linear B Control

**Engine:** `lineara_engine.py` (pure Python + NumPy/Matplotlib, seed 2026, self-contained)
**Result JSON:** `quant_report_A_vs_B.json` | **Normalized corpora:** `la_sequences.json`, `lb_sequences.json`
**Date:** 2026-08-24 (session) | **Tags:** (a)=measured (b)=interpretation (c)=speculation

---

## 0. Bottom line (a/b)

1. **(a)** Linear A is statistically **non-random in sign order**: real reading-order text is
   measurably more compressible than token-shuffled controls (LZ −7.6% to −12.4% depending on
   tokenization; bigram enrichment up to ~250–400× over independence).
2. **(a)** Linear A's sequential structure is weaker than a deciphered language under the
   identical pipeline (readable-core LZ −7.6% vs Linear B −30.0%; H3|H2 0.88 vs 2.55 b) —
   *magnitude superseded by the size-matched control: ≈1.7× under a fair matched-inventory
   control, statistically indistinguishable with text type matched (see correction footnote,
   §4)*.
3. **(b)** A large part of the measured order in Linear A is epigraphic rather than strictly
   linguistic: the unreadable-fragment marker 𐝫 (U+1076B) is 21% of all tokens and heavily
   first-position biased (z≈+15σ); removing damage markers, numerals and ideograms leaves a
   "readable core" whose residual order (−7.6%) is real but shallow.
4. **(c)** The readable-core profile (weak bigram preference, no trigram productivity, large
   inventory 326 types in 5,839 tokens) is compatible with a signary containing a compact set of
   frequent formulaic strings rather than a fully productive syllabic orthography. **No
   decipherment is claimed; this is a measured structural fingerprint only.**

---

## 1. Data & provenance (binding caveats)

| Corpus | Source (real, scholarly-digitized) | Entries | With text | Sites/loci |
|---|---|---|---|---|
| Linear A | `mwenge/lineara.xyz` `LinearAInscriptions.js` (GORILA-derived) | 1,721 | 826<sup>†</sup> | Hagia Triada, Phaistos, Knossos, Zakro, … |
| Linear B | `mwenge/linearb.xyz` `LinearBInscriptions.js` (deciphered-language control) | 5,832 | 5,534 | Knossos, Pylos, Thebes, Mycenae, … |

<sup>†</sup> 826 = records with a flattened `transcription` field, which this engine's sign
tokenizer consumes. Per GROUND_RULES.md §7 this is a **no-flattened-field count, not a content
gap**: only 13/1,721 records are truly text-less (the other 895 carry `parsedInscription` /
Douros transliteration and are used by the linguistic layer, `lineara_linguistics.py`).

- Not hand-verified full critical editions; a digitization of published corpora (GORILA /
  DĂMOS lineage). This engine's sign-level statistics use the 826 records with a flattened
  `transcription` field (see † above); the linguistic layer additionally draws on the 1,712
  records with Douros transliteration. Only 13/1,721 records are truly text-less (GROUND_RULES.md §7).
- Linear A transcriptions encode damaged/unreadable signs; MWenge uses the code point
  U+1076B (and literal `\U0001076b` runs) as the fragment marker → token `UNK`. It is 21.2% of
  Linear A tokens (2,128/10,018) vs ~0% in Linear B — a corpus-quality fact, reported and
  controlled for.
- Reading order = the order given in the digitized transcription (Linear A/B are both
  left-to-right scripts); fragment punctuation `[ ]` dropped. Tokens = individual script
  characters (Linear A/B signs, Aegean numerals, ideograms) with `UNK` for damage runs.
- **Normalized upstream input:** the approved Foundation TSVs (`linearA_corpus_clean.tsv`,
  `linearB_corpus_clean.tsv`, `linearA_sign_inventory.tsv`, per `GROUND_RULES.md`,
  agent-scientific-engineer) are the canonical normalized corpora; the `raw/` digitizations
  used here (`raw/lineara.json`, `raw/linearb.json` via `raw/parse.js`) remain the byte-pinned
  origin of the counts above (SHA-256 in `CORPUS.md` §1–2).

---

## 2. Method (identical pipeline, both scripts)

Classical statistics + Kolmogorov-style compression probes, mirroring the Indus IVS engine:

- **Entropies:** unigram H1, bigram H2, trigram H3, conditional H2|H1 and H3|H2
  (bits; relative frequency estimates; no smoothing — sparsity noted below).
- **Controls (three per corpus):** token-shuffle (same alphabet, random order),
  sequence-shuffle (randomize signs within each inscription), Markov-1 (Laplace-smoothed
  bigram successor resampling, 15% uniform reset — avoids chain collapse).
- **Compressibility:** LZ (gzip level 6) ratio compressed/raw on the concatenated
  sign stream, real vs each control. Δ% = real-vs-token-shuffle relative difference.
- **Motifs:** bigram/trigram counts with expected counts under independence
  (frequency product × window count); enrichment = observed/expected; min count 3 (2) for
  2-grams (3-grams) to suppress hapax noise.
- **Positional bias:** z-score of observed first-position rate vs unigram share.

**Variants (for comparability and robustness):**
- *all* — every script character (numerals + ideograms + damage markers included);
- *syllables-only* — script-block signs only, numbers/ideograms excluded (UNK kept);
- *readable core* — syllables-only AND damage markers removed (cleanest writing-system core).

---

## 3. Primary results (a — measured)

| Variant | Corpus | Sides | Tokens | Types | H1 | H2|H1 | H3|H2 | LZ real | LZ tok-shuf | Δ% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all | Linear A | 825 | 10,018 | 358 | 6.03 | 4.49 | 1.81 | 0.2285 | 0.2607 | **−12.4%** |
| all | Linear B | 5,534 | 67,949 | 240 | 6.41 | 4.85 | 2.48 | 0.2024 | 0.2583 | **−21.6%** |
| syllables | Linear A | 823 | 7,967 | 327 | 5.71 | 4.25 | 1.79 | 0.2214 | 0.2670 | **−17.1%** |
| syllables | Linear B | 4,771 | 48,874 | 88 | 5.71 | 4.77 | 2.55 | 0.1815 | 0.2589 | **−29.9%** |
| readable | Linear A | 785 | 5,839 | 326 | 6.65 | 4.27 | 0.88 | 0.2495 | 0.2701 | **−7.6%** |
| readable | Linear B | 4,771 | 48,874 | 88 | 5.71 | 4.77 | 2.55 | 0.1815 | 0.2592 | **−30.0%** |

Control anchors (a): seq-shuffle LZ: LA 0.2506 / LB 0.2515; all Markov-1 ratios ≈ shuffle level
for Linear B (0.2596) but collapse for Linear A (0.0520) — itself a measured artifact of the
damage-marker's self-transition bias (see §5). Entropy curves: `entropy_curves_LA_LB.png`;
compressibility: `lz_compressibility_LA_LB.png`.

**Reading the table (b):** every variant shows real order (Δ < 0 always), and the deciphered
  control is always stronger (ratio 1.7–4× in this unmatched, engine-convention table — the
  fair size-matched ratio is ≈1.7×, see correction footnote §4). The 88-type Linear B syllable inventory matches the known
~90-sign syllabary; the 326-type Linear A "readable" inventory includes many rare/uncertain
signs — inventory size relative to corpus (5.6% vs 0.18% type-token) makes Linear A's
entropy estimates noisier, so cross-corpus entropy comparisons are indicative, while the
within-corpus Δ% (each corpus against its own shuffle) is the cleaner signal.

---

## 4. Cross-corpus positioning incl. Indus (a, indicative)

> **Correction footnote (approved size-matched control, `size_matched_report.md`, agent-engineer):**
> the earlier "~4× weaker structure" framing is **retired**. (a) Matched at equal token counts
> (T = 5,839, 5 seeds), the LB>LA LZ-Δ gap is ≈**1.7×** (−17.6% vs −10.4%) — the ordering
> survives, the *magnitude was overstated*. (b) With text type matched (short admin labels),
> the scripts are statistically **indistinguishable** (H3|H2 0.85 vs 0.88; LZ −7.4% vs −10.4%):
> the trigram-productivity gap is a corpus-composition property (long tablets vs short labels),
> not a demonstrated script difference.

Same probes run earlier on the Indus script (Mohenjo-daro CISI subset; 179 sides, 1,003
tokens, 182 signs — see `/home/team/shared/indus/`):

| Corpus | N tokens | Δ% LZ vs shuffle | H2|H1 | H3|H2 |
|---|---|---|---|---|
| Linear B (deciphered language) | 48,874 | **−30.0%** | 4.77 | 2.55 |
| Linear A (all graphemes) | 10,018 | −12.4% | 4.49 | 1.81 |
| Indus script | 1,003 | −9.7% | 2.41 | 0.41 |
| Linear A (readable core) | 5,839 | −7.6% | 4.27 | 0.88 |

(b) Under the original (unmatched) pipeline, known-language Linear B outranks the
undeciphered scripts on every structural axis; the magnitude of that gap is superseded by the
size-matched control above (≈1.7× fair; ≈indistinguishable with text type matched). The Indus
and Linear A
readable-core values are surprisingly close (−9.7% vs −7.6%): both look like writing systems
whose sign order is constrained **shallowly** (strong formulaic bigrams, weak/absent trigram
productivity), unlike the productive syllable sequencing of Linear B. Caveats: different
alphabets, token standards, corpus sizes — indicative positioning only (a), inference (b).

---

## 5. Motifs, positional bias, network (a — measured; (b) where labeled)

**Linear B readable core — top bigrams with standard decipherment values (b, standard
Ventris-Chadwick sign values — the script is deciphered):**
`𐀁𐀐 e-ke (341)`, `𐀵𐀰 to-so (310)`, `𐀒𐀵 ko-to (299)`, `𐀞𐀫 pa-ro (298)`, `𐀒𐀺 ko-wo (255)`,
`𐀙𐀵 na-to (251)` — i.e., the most frequent digrams are actual Mycenaean Greek word/lexeme
fragments, confirming the compression probe detects real morphology in the control (b).

**Linear A readable core — top bigrams (sign pairs, count):** `𐙂𐘁 (39)`,
`𐘳𐘚 (22)`, `𐘳𐘅 (21)`, `𐘞𐘴 (21)`, `𐘞𐘽 (20)`, `𐘞𐘞 (20)`. Biggest enrichment
(observed/expected, count≥3): `𐘼𐘼 (249.9×)`, `𐙱𐘍 (134.9×)`, `𐘉𐘪 (74.4×)`, `𐘛𐘯 (67.5×)`,
`𐙰𐘆 (61.9×)`. (b) Sign-doubling (𐘼𐘼) and a handful of persistent collocations dominate;
the enrichment ranks are long-tail (most LA bigrams occur once) — compatible with a compact
formulaic core + rare-sign noise (§0.4). Network: `cooccurrence_network_LA.png`;
motif map: `fbsc_motif_map_LA.png`.

**Positional bias (z, first-position vs unigram share, readable core):**
- Linear A: `𐘇 +11.35`, `𐚨 +4.72`, `𐙓 +4.45`, `𐙐 +4.25`, `𐙹 +4.09` — (b) 𐘇 has a stable
  word-front role (in all-graphemes the damage marker tops the list at +15.05, i.e., fragments
  begin at damaged areas — an epigraphic artifact).
- Linear B: `𐀀 a- +20.07` (b: the Greek /a-/ initial class and proclitics; a known deciphered
  signature), `𐁁 +10.03`, `𐀓 +9.52` …
- Visual: `positional_bias_LA.png` (heatmap: first / middle / last rates ÷ unigram rate).

---

## 6. Limitations (honest)

- Digitized-subset corpora, not hand-verified editions; Linear A heavily fragmentary (21%
  damage markers) — all numbers carry this grain.
- Entropy estimates unsmoothed; small Linear A corpus → hapax bigram dominance; cross-corpus
  entropy comparisons indicative only.
- LZ/gzip is an approximate Kolmogorov probe on concatenated streams; token-shuffle is the
  primary null; Markov-1 collapses under damage-marker persistence (reported and explained, not
  hidden).
- No phonetic/phonological claims for Linear A: sign values are unknown (that IS the open
  problem); "types" are Unicode codepoints, and identical-looking signs across fragments were
  not manually adjudicated.

## 7.5 Linguistic layer — transferred values, formula rediscovery, phonotactics (NEW)

Run: `lineara_linguistics.py` → `linguistics_report.json` + `formula_signal_map_LA.png` +
`word_length_LA_vs_LB.png`. Values used are the **dataset's own DĂMOS-convention
transliterations** (1,712 entries) — nothing invented; `*NNN` = undeciphered sign; RA₂/PA₃
homophone subscripts normalised for inventory counts.

**(a) Sign-value inventory & overlap with Linear B**
| Level | LA | LB | notes |
|---|---|---|---|
| distinct value types (raw) | 60 | 89 | Subscript-homophones merged |
| **shared (raw)** | **52** | — | **86.7% of LA inventory** |
| **shared (clean, minus editorial tags: L, CAP, GAL, VAS …)** | **50/53** | — | **94.3% of LA inventory** |
| LA values NOT in LB | `JU`, `ZU` | | only two (b): signs with LA-specific values |
| LB values NOT in LA | jo, wo, we, so, no, pe, mo, do, qo … | | LB later additions |
| undeciphered `*N` types | 30 (562 tokens) | | `*301` ×274 = the classic libation sign |

**(a) The libation formula falls out automatically.** Top 5-sign sequences in the whole
corpus at transliteration level: `A-TA-I-*301-WA` (12×), `TA-I-*301-WA-JA` (11×),
**`JA-SA-SA-RA-ME` (9×)**, `SA-SA-RA-ME-U` (6×), `SA-RA-ME-U-NA` (6×),
**`U-NA-KA-NA-SI` (6×)**, `I-*301-WA-JA-JA` (5×), `I-PI-NA-MA-SI / PI-NA-MA-SI-RU /
NA-MA-SI-RU-TE` (5× each). I.e., the known first three rows of the libation formula
(*ja-sa-sa-ra-me u-na-ka-na-si a-ta-i-*301-wa-ja*) + the *i-pi-na-ma-si-ru-te* chain are the
most frequent multi-sign strings in the corpus — pure frequency mining **rediscovers the
philological consensus without any decipherment assumption** (a+b). The explicit probe list:
`JA-SA-SA-RA-ME` in **9** inscriptions (IOZa2, IOZa6, IOZa9, IOZa12, IOZa16, PLZf1, PSZa2,
TLZa1, PKZa27); core `-SA-SA-RA-ME` in **12**; `U-NA-KA-NA-SI` in **6** (IOZa2, IOZa9, KOZa1,
PKZa8, SYZa2, PKZa27) — the "Za" = offering-table text-class, exactly where the formula is
attested. Note (a): counts include the shared core across JA-/A- initial variants (9 vs 12).

**(a) Phonotactics of the transferred readings are language-plausible** (b: interpretation):
- Syllable shapes: LA **96% CV or V** (CV 3,980 / V 523 of 4,681) — the same CV-dominant
  syllabary profile as Linear B (CV 39,183 / V 6,902). No impossible heavy-cluster inventory.
- Word lengths: LA mean **1.89** syllables (median 1, max 9) vs LB mean **2.80** (median 3,
  max 11). (b) LA's shorter tokens fit its administrative/offering genre and the formula's
  tight grouping; (c) does not itself argue for/against a language — a syllable script with
  short lexemes would look like this.
- Word-edge distributions differ (a): LA strongly starts KU/KA/SI and ends KA/KU/RO; LB starts
  a/e/o- and ends -jo/-ro/-to — the two scripts show **distinct positional profiles** even
  under the same value system (b: separate phonotactic/morphological habits).

**Bottom line for the lead's question** (b): with 94% of the LA signary transferring Linear B
values and the famous formula falling out automatically at the top of the frequency ranking,
the value-transfer approach is *empirically grounded on this corpus* — but it still does not
read Linear A: the undeciphered `*N` signs (esp. *301 in the formula) and unknown LAn-signs
carry the load at exactly the points where a decipherment would need them. See H5–H9 below.

---

## 8. Formal hypotheses (falsifiable, tagged)

Indus-style hypotheses retained from the quantitative layer:
- **H1 (positional head-sign).** "𐘇 (and the damage-marker art) preferentially occupy
  sequence-initial position" — LA 𐘇 z=+11.35. Test: full DĂMOS edition; falsified if bias
  vanishes outside the MWenge subset.
- **H2 (fixed collocations).** "𐘼𐘼, 𐙱𐘍, 𐘉𐘪 … are stable formulas, not noise" —
  enrichment 62–250× at low counts. Test: count stability in the full corpus.
- **H3 (weaker-than-language order).** "LA readable-core LZ Δ (−7.6%) and H3|H2 (0.88 b)
  sit far from a deciphered language (LB −30%, 2.55 b)" — superseded in magnitude by the
  size-matched control (fair-control ≈1.7×; text-type matched ≈indistinguishable, see
  correction footnote §4); retained as: "at matched token counts LB remains measurably more
  compressible, but the dramatic gap was overstated." Falsified if a fuller corpus closes
  even the fair-control gap.
- **H4 (no long-range scaffold).** "No stable trigram+ formula beyond the libation rows
  exists in LA" — testable on full corpus; a falsified H4 is a major positive signal.

New from the linguistic layer:
- **H5 (formula recovery).** "Frequency mining recovers the known libation rows as the top
  5-sign strings" — MEASURED (a): ja-sa-sa-ra-me 9×, u-na-ka-na-si 6×, a-ta-i-*301-wa 12×.
  Falsifiable: an independent corpus re-computation must reproduce ≥8 of the 12 inscriptions.
- **H6 (value transfer coverage).** "≥90% of LA's syllabic inventory shares LB values" —
  MEASURED clean 94.3% (a). Falsified if the full edition adds many new non-LB values.
- **H7 (genre stratification).** "The formula rows cluster in the offering-table class (Za);
  word-length profile differs by genre" — (a) Za inscriptions dominate the top hits.
  Test: genre-tagged frequency table on full corpus.
- **H8 (phonotactic separability).** "LA and LB differ in word-edge value profiles under the
  same value system" — MEASURED (a) init/final distributions differ; falsifiable with a
  distance test on full corpora.
- **H9 (undeciphered-sign load).** "The undeciphered *N signs (esp. *301) occupy exactly
  the formulaic hotspots where meaning would be carried" — (a) *301 is the 3rd most common
  LA token (274) and sits inside a-ta-i-*301-wa-ja; (c) reading *301 in the formula would be
  the highest-value decipherment step, and any proposal is testable against the co-occurrence
  pattern quantified here.

---

## 9. Next steps (for the team)

1. Pull the full DĂMOS/GORILA Linear A corpus (this subset is 1,721/≈1,500+ known sides → extend);
2. Word-level analysis on Linear B's deciphered segmentation (`transliteratedWords` present in
   the dataset) to calibrate sensitivity, then transfer calibrated probes to Linear A;
3. FBSC-style seed compression of the top Linear A sign streams (the compression-specialist
   deliverable) — test whether the formulaic core (top 5-sign chains incl. the libation rows)
   admits a short generative model;
4. Cross-validate the 𐘇-initial and 𐘼𐘼/𐘉𐘪 collocations and the formula-row counts against
   epigraphic literature (Godart–Olivier concordance; DĂMOS) before treating them as secure
   (c → b);
5. Formal test of H9: propose a reading for *301 constrained by its co-occurrence profile
   (3rd-most-frequent LA value token at 274, behind KU/KA; embedded in a-ta-i-*301-wa-ja, 12×)
   and score it against the quantified context — the highest-value decipherment step this
   pipeline can now support.

---

## 10. Files in this delivery

`lineara_engine.py`, `lineara_linguistics.py`, `quant_report_A_vs_B.json`,
`linguistics_report.json`, `la_sequences.json`, `lb_sequences.json`,
`LINEA_ENGINE_REPORT.md`, `entropy_curves_LA_LB.png`, `lz_compressibility_LA_LB.png`,
`cooccurrence_network_LA.png`, `fbsc_motif_map_LA.png`, `positional_bias_LA.png`,
`formula_signal_map_LA.png`, `word_length_LA_vs_LB.png`, `raw/` (original JS corpora +
parsed JSON + parser).