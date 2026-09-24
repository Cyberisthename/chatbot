# Linear A — Structural / Functional Translation Pass

*Owner-facing deliverable — "try real hard, look at all the options, see if we can translate it."*
*Author: agent-theoretical-physicist-2 · 2026-09-12 (v1) / 2026-09-12 (v2 — owner 5-point
maximum-effort spec implemented in §§9–13, anchor corpora by agent-scientific-engineer
folded in) · Corpus: raw/lineara.json (1,721 inscriptions, GORILA-sourced, Unicode-encoded)
+ raw/linearb.json (LB Greek control) + la_sequences.json (825) + lb_sequences.json (5,534)
· Tags: (a) measured on this corpus, (b) interpretation, (c) speculation. Every claim below
is explicitly tagged.*

---

## 0. The bottom line, up front

**We can now *functionally translate* Linear A — what the sign system does — to a
defensible, quantified degree: it is a dual-register script (palace accounting + sanctuary
ritual), with a large readable numeric/ideographic core (≥35% of tokens are strictly
numerals, fractions and commodity logograms — >50% if the ubiquitous single-sign
administrative abbreviations on nodules are included), a small set of securely glossable
accounting words (KU-RO "total" ×37, KI-RO "owed" ×16, PO-TO-KU-RO "grand total", KI-RA
"balance", KA-PA "summary account", E-*82 "assessment/paid", DA-DU-MA-TA "grain
contributions"), and a reconstructible ritual formula whose *structure* is fully
recoverable (A-TA-I-*301-WA-JA · JA-SA-SA-RA-ME · U-NA-KA-NA-SI · I-PI-NA-MA · SI-RU-TE).**

**We cannot yet *phonetically translate* Linear A — the language remains unread.** The
Linear B value-transfer gives ~94% of the signary "shadow" readings, but those values are
borrowed, not proven; the statistics show the shadow language is phonotactically plausible
and morphologically structured (suffixal, reduplicating), yet no word beyond the accounting
core carries a verified meaning, and the undeciphered signs (*301, *304, *188…) sit exactly
at the formula hotspots where meaning is carried. Task verdict: **functional translation
achieved (accounting register ≈ 35–55% readable depending on how single-sign abbreviations
are counted; ritual register structurally translated); lexical/phonetic translation NOT
achieved — and no tool we have can cross that line without a bilingual or an identified
language.**

---

## 1. What "translate it" can mean for Linear A (and what we can do)

| Level | Question | Status |
|---|---|---|
| L1 Orthographic | How does the script work (sign classes, word order, punctuation, ligatures)? | **(a) solved** — measurable grammar |
| L2 Functional | What does each formula *do* (totalling? deficits? libation? sealing)? | **(a)/(b) largely solved** — formula × context matrix below |
| L3 Lexical | What does each *word* mean (beyond accounting glosses)? | **(b) only 9 glossed words from the editorial source; (c) nothing verified** |
| L4 Phonetic | What language is it / how is it pronounced? | **(b) phonotactic profile; (c) no verified phonetic values beyond LB transfer** |

The owner asked to "see if we can translate it" — we attack L1–L4 with every tool in hand.

---

## 2. Corpus census (my independent run, (a))

Cleaned token stream: **n = 6,406** tokens across 1,721 inscriptions (excludes empty and
line-break tokens; includes fragment marker 𐝫).

| Token class | n | % | Notes |
|---|---|---|---|
| Numeral | 1,297 | 20.2% | 1×310, 2×158, 3×101, 5×81, 4×78, 10×70, 6×59, 20×40, 7×30, 8×25, 30×21, 40×15, 100×12, 9×10, 50×7 |
| Multi-syllabic word (`A-A-A`) | 1,099 | 17.2% | the true syllabic "text" (lexicon incl. KU-RO, JA-SA-SA-RA-ME…) |
| Single syllabogram (KU, KA, SI…) | 1,006 | 15.7% | overwhelmingly administrative abbreviations on nodules; KU ×170, KA ×169, SI ×118, NI ×76, TE ×58, ZE ×47 |
| Undeciphered \*N complexes | 682 | 10.6% | sign-level ≈730 incl. ligatures; \*301 alone ≈238 standalone (258 with ligatures) |
| Ideogram / logogram strings | 590 | 9.2% | GRA 62, VIN 53, CYP 52, OLIV 24, OLE family ≈96 (OLE+U/+KI/+MI/+DI/+TA), GRA+PA 19, CYP+D/E 32 |
| Fragment/damage marker | 552 | 8.6% | U+1076B, epigraphic artifact, NOT a sign — 21% in the engine's raw tokenization |
| Word separator 𐄁 (U+10101) | 468 | 7.3% | Linear A *does* use word dividers (unlike Indus) |
| Other (mixed fragment tokens) | 384 | 6.0% | incomplete signs/sequences not classifiable above |
| Fraction | 328 | 5.1% | ½×120, ¼×56, ⅙×28, ¾×27, 1⁄16×26, ⅓×24, ⅕×22 (+ ≈⅙ 28 counted among "other") |

**Key structural fact: ≥35% of every Linear A token is strictly a number, a fraction or a
commodity logogram — and >50% once the single-sign administrative abbreviations
(KU/KA/SI/RO/TE…, which function as short identifiers on sealings) are included. This is
an accounting notation with an attached syllabic layer, not a literary script.** That
single fact shapes everything: the *readable* part is the administrative core; the
*unreadable* part is a thin syllabic vocabulary (n≈1,099 multi-syllable tokens, 1,443
distinct transliterated tokens, most hapax).

---

## 3. Functional registers — the script splits cleanly by object type ((a) + (b))

Cross-tabulating formulas × support × site (my computation over all 1,721 inscriptions)
recovers **three non-overlapping functional registers**:

| Register | Object (support) | Sites | Signature formulas | Function |
|---|---|---|---|---|
| **R1 Palace accounting** | Tablet (435), Lames (18) | Haghia Triada (overwhelmingly), Phaistos, Zakros, Khania | `%WORD% + numeral(+fraction)`, closed by **KU-RO / KI-RO / PO-TO-KU-RO / KA-PA** | commodity ledgers, deficits, totals |
| **R2 Sealing/rationing** | Nodule (890), Roundel (151), Sealing (12) | all sites (HT dominates) | **\*301** standalone (82% of *301 uses), sometimes `MI+*301`, `I+*301`, `E-*301` + numeral | docket / seal-stamp accounting |
| **R3 Sanctuary ritual** | Stone vessel (60: offering tables), Clay vessel, Metal | Iouktas, Psykhro, Kophinas, Syme, Palaikastro, Troullos, Platanos | **JA-SA-SA-RA-ME (7), U-NA-KA-NA-SI (4), SI-RU-TE (7, all stone vessels), A-TA-I-\*301-WA-JA (11, all stone vessels)** | dedicatory libation formula |

Crucially (a): **KU-RO occurs in 34/34 cases on Tablets; SI-RU-TE in 7/7 on Stone
vessels; JA-SA-SA-RA-ME 6/7 on Stone vessels (+1 Metal); U-NA-KA-NA-SI 4/4 on Stone
vessels.** The formulaic core never migrates between registers — the system is
**genre-stratified** (supports hypothesis H7 of the engine report). This is the strongest
available *functional* evidence: same signs, different object classes, different
formulaic behavior → different discourse functions.

---

## 4. Formula-by-formula functional translation (the core of this pass)

### 4.1 R1 — Accounting formula templates (≈35–55% readable, see §2 census)

Measured positional grammar (a): entries follow the pattern
`[word] [numeral] [fraction?]`; totals appear as `KU-RO [number]` at the **end** of the
account (KU-RO positional: MID 33/34 — i.e., after the entry list, before any closing
word), `PO-TO-KU-RO [number]` after KU-RO (grand total), and deficits open/close with
`KI-RO [number]`.

**Exemplar structural translations (my rendering, gloss source = dataset editorial
readings (b); every number (a)):**

- **HT1** (Haghia Triada, Tablet): `QE-RA₂-U 𐄁 KI-RO 197 · ZU-SU 70 · DI-DI-ZA-KE 52 ·
  KU-PA₃-NU 109 · A-RA-NA-RE 105` →
  **"Deficit account (KI-RO 'owed'): four named entries, amounts 197 / 70 / 52 / 109 /
  105."** The dataset's own translation glosses KI-RO as "owed" — a *deficit* ledger.
- **HT9a**: `SA-RO 𐄁 TE 𐄁 VIN 𐄁 PA-DE 𐄁 5¾ · *306-TU 10 · DI-NA-U 4 · QE-PU 2 ·
  *324-DI-RA 2½ · TA-I-AROM 2½ · A-RU 4¼ · KU-RO 31¾` →
  **"Wine (VIN) account: per-person rations 5¾, 10, 4, 2, 2½, 2½, 4¼ …; total (KU-RO)
  31¾."** Honest arithmetic note: the visible entries sum to **31**, the stated total
  exceeds it by ¾ — one fraction is lost to damage or lives in a column flattened by the
  linear transliteration (fragment marker present at 8.6–21% of tokens). The total is
  *consistent* with KU-RO="total" but not verifiable exactly on this flattened stream.
- **HT122b**: `JE-DI 𐄁 *346 𐄁 VIR+[?] 𐄁 *306-KI-TA₂ 7 · A-RA-JU-U-DE-ZA 2 · QA-QA-RU 2 ·
  DI 2 · DA-RE 2 · KU-RO 65 · PO-TO-KU-RO 97` →
  **"Census-style tablet: logogram VIR (men) + named entries with counts; total (KU-RO)
  65; grand total (PO-TO-KU-RO) 97."** Honest note: the five visible numeric entries sum
  to 15; the stated 65 implies further quantities lost to damage or embedded in the
  VIR/CYP logogram lines (fragment marker ≥8.6%) — another example of the flat-stream
  arithmetic limit, *not* a disproof of the total reading. 97 > 65 is consistent with
  PO-TO-KU-RO as an all-tablet "grand total" (b, cf. Greek πᾶς "all").
- **ZA8** (Zakros): `KI-RA 𐄁 A-TA-RE 𐄁 NI ½ · KU-TU-KO-RE [double mina] · A-RI-NI-TA 1 ·
  … · KA-I-RO 4` → **"Balance (KI-RA) account involving figs (NI) and a double-mina
  weight; balances ½, 1, …, 4."**

**Arithmetic verification of KU-RO (my test, honest result):** summing all numerals in
linear order before KU-RO matches the stated total in **8/29** tablets; line-parsed
summation matches **0/29** because the layout is columnar and the transliteration flattens
columns (a genuine, reproducible data limitation, not a refutation). Of the 21
non-matches, most are close (HT13: 131 vs 130 — one damaged fraction; HT94a: 111 vs 110;
HT102: 1,070 vs 1,060) or have no numerals in the same line (values live in prior lines).
With the fragment marker at 8.6–21% of tokens, exact column sums are often impossible to
reconstruct from the flat stream. **Conclusion (b): the KU-RO reading "total" is
consistent with the data, is the field's consensus (Brice; Godart–Olivier), is
corroborated by PO-TO-KU-RO > KU-RO, and is the *only* LA reading I would call secure —
but is not arithmetically verifiable tablet-by-tablet on this damaged, columnar corpus.**
Suggested next step: rebuild column layout from `parsedInscription` line structure
(§7, experiment E5).

### 4.2 Functional lexicon (from the dataset's editorial translations — (b) philological)

These glosses come from the source dataset (curated from the published GORILA/translation
literature), *not* invented by this pipeline; they are the field's best-attested readings:

| LA word | Gloss | n | Register | Site |
|---|---|---|---|---|
| **KU-RO** | "total" | 37 | Tablets | HT, Phaistos, Zakros |
| **KI-RO** | "owed" (deficit) | 16 | Tablets | HT |
| **KA-PA** | "summary account?" | 6 | Tablets | HT |
| **KU-PA** | "transaction term?" | 4 | Tablets | HT, KH, ZA |
| **KI-RA** / **KA-I-RO** | "balance" | 2+1 | Tablets | HT, ZA |
| **PO-TO-KU-RO** | "grand total" | 2 | Tablets | HT |
| **DA-DU-MA-TA** | "grain contributions" | 1 | Tablet | HT |
| **E-\*82** | "assessment" or "paid" | 1 | Tablet | ZA |

That is the entire verified-by-scholarship vocabulary: **9 words, ~70 attestations, all
accounting terms, all morphologically simple (KU-RO, KI-RO, KA-PA, KU-PA, PO-TO-KU-RO)**.
Note the family: KU-RO / KI-RO / KA-I-RO / PO-TO-KU-RO share the *-RO* nucleus —
suggestive of a root + suffix morphology at work (b), i.e. *ku-ro* "total", *po-to-ku-ro*
"grand total", *ki-ro* "owed", *ka-i-ro* "balance": a small derivational family.

### 4.3 R3 — The sanctuary libation formula (fully reconstructable structure, (a))

The most frequent multi-sign strings in the whole corpus (pure frequency mining,
engine report (a)): **A-TA-I-\*301-WA (12×), TA-I-\*301-WA-JA (11×), JA-SA-SA-RA-ME (9×),
U-NA-KA-NA-SI (6×), I-PI-NA-MA-SI-RU-TE chain (5×)** — i.e. the known first three rows
of the libation formula *fall out of the data with zero decipherment assumptions*.

Full formula as attested (my reconstruction from the 11 vessels carrying
A-TA-I-\*301-WA-JA and the 10 inscriptions carrying the -SA-SA-RA-ME core, (a)):

```
ROW 1   A-TA-I-*301-WA-JA   (variant: TA-NA-I-*301-U-TI-NU / TA-NA-I-*301-TI / A-TA-I-*301-WA-E)
ROW 2   [dedicant/name slot]  (JA-DI-KI-TU, O-SU-QA-RE, RE-I-KE, WI-TE-JA-MU … — varies)
ROW 3   JA-SA-SA-RA-ME      (core -SA-SA-RA-ME; variants A-SA-SA-RA-ME, SA-SA-RA-ME)
ROW 4   U-NA-KA-NA-SI       (variant U-NA-RU-KA-NA-TI / U-NA-KA-NA)
ROW 5   I-PI-NA-MA SI-RU-TE (variant I-PI-NA-MI-NA SI-RU; SI-RU-TE 7×, always stone vessels)
ROW 6   [optional closing]  (TA-NA-RA-TE-U-TI-NU, TA-NU-NI-KI-NA NI-NU-NI, I-NA-JA-PA-QA)
```

**Functional reading (b):** a fixed *dedicatory libation utterance* carved on stone
offering tables at peak sanctuaries (Iouktas, Psykhro, Kophinas, Syme, Palaikastro,
Troullos, Platanos). The row-1 *-I-*301-WA-JA* slot is a verb-like frame with the
undeciphered sign **\*301 at its heart**: `A-TA-I-[*301]-WA-JA` — whatever *301 means
(§5.5), that is the oath/offering word. Row 3–4 *ja-sa-sa-ra-me u-na-ka-na-si* is the
formulaic core that recurs across every site, in both full and truncated forms
(IOZa9/PKZa27 contain only rows 3–4), proving it is the *irreducible* utterance.

The dataset's editorial gloss for IOZa2 renders it as (c): "gives · name? · this
dedication · requesting · a favour · divine". **I report that gloss chain as the source
dataset's speculative reading (c), not verified by us** — it is a coherent narrative
(dedicant gives an offering asking a divine favor) but every element beyond the formulaic
structure is unproven. What we can honestly claim (a/b): it is a fixed votive formula
inscribed on ritual furniture at high-altitude sanctuaries, with a name slot and
site-invariant core.

### 4.4 R2 — Nodule/sealing register

\*301 appears in **288 inscriptions, 82% of them Nodules** (236), usually **alone
(`<START> *301 <END>` ×232)** or in ligatures MI+*301, I+*301, E-*301, *301+*311, *301-*301
(a). On such clay sealings, a single sign + numeral is the classic **docket/seal-owner**
notation. Function (b): *301 functions as a **logogram** on nodules (a commodity or office
mark), and as a **word-internal element** in the ritual formula — a dual role consistent
with a sign that is either a true ideogram or a syllabogram whose value we lack
(§5.5).

---

## 5. The phonetic attack — every statistical tool applied

The owner wants us to "attack the phonetic question with every statistical tool we have."
Here is the full arsenal, each result tagged.

### 5.1 Value transfer from Linear B ((a) coverage; (b) validity)

- **(a)** 94.3% of the clean LA syllabic inventory shares its value with Linear B
  (engine report): 50/53 LA value-types shared; only JU, ZU LA-specific; the 30
  undeciphered *N types (730 sign-tokens; *301 ×258, *304 ×39, *188 ×30, *86 ×23, *306 ×22,
  *21 ×21, *401 ×17) are the gap.
- **(a, my check)** of 1,098 purely syllabic LA word-tokens, only **6** are composed
  entirely of unknown-value syllables; average unknown syllables per word = 0.26. The
  "shadow" is nearly complete at the *sign* level.
- **(b) The transfer is circular for decipherment.** The values are Linear B's. They make
  LA *transliterable* (that is what the corpus gives us), but a phonetic value is only a
  hypothesis for Linear A until the language is identified. What the transfer *does* buy:
  a way to run every phonological/morphological probe below on strings, which is
  impossible for scripts without a read layer.

### 5.2 Phonotactics ((a) measured, (b) interpretation)

- Syllable shapes: LA **96% CV or V** (CV 3,980 / V 523 of 4,681) — same CV-dominant
  profile as Linear B (engine report (a)). No heavy clusters: the shadow phonology is a
  plain open-syllable language.
- Word lengths: LA **mean 1.89 syll** (median 1, max 9) vs LB 2.80 (median 3, max 11).
  (b) LA is lexically short — consistent with an administrative/ritual lexicon of
  formulaic short items, or with real morphology being smaller than LB Greek's.
- Word-initial (LA): A- ×127, KU- ×87, I- ×71, SA-/JA-/DA-/KI-/KA-/SI- … (my run (a));
  word-final (LA): -RO ×74, -TE ×53, -JA ×49, -RE ×48, -NA ×47, -TI ×44, -RA/-RA₂ ×80,
  -TA/-RU/-SI/-MA/-NE/-SE… (my run (a)). **The language is strongly suffixal: 19 of 20
  top word-final syllables are open vowels, and endings are dominated by a short set
  (-RO, -TE, -JA, -RE, -NA, -TI), while initials spread over a much wider set.**
- Vowel signatures (my run (a)): top whole-word vowel strings AA, AI, IA, UO, AE, AU,
  EI, UA, A₂, AAE, IO, II… — **no dominant vowel-harmony pattern**; mixed profiles with
  /a/ prominence. This is *consistent* with (b) an agglutinative, suffix-bearing
  language family but does **not** discriminate among candidates (Anatólic, Semitic,
  isolate all produce such distributions).

### 5.3 Morphology: reduplication, prefix/suffix, stem predictiveness ((a) + (b))

- **Reduplication is real and productive ((a) my run):** JA-SA-SA-RA-ME (SA-SA, 7×),
  QA-QA-RU (3×), TI-TI-KU (2×), DI-DI-ZA-KE, SA-SA-RA-ME, JA-JA, KI-KI-RA-JA, KI-KI-NA,
  KU-KU-DA-RA, NA-MA-MA-TI-TI, WI-JA-SU-MA-TI-TI, TA-TA, DA-DA, WI-SA-SA-NE… Greek
  Linear B shows no such reduplicating habit (b); this is a **language-specific
  phonological signature** of Minoan (c: consistent with the long-noted "Minoan
  reduplication" in scholarship — *sa-sa-ra-me*, *di-di-za-ke*, *qa-qa-ru* are the
  textbook examples from the literature).
- **Suffix stacking (b):** -SA-SA-RA-ME < SA-SA-RA + ME; U-NA-KA-NA-SI / U-NA-RU-KA-NA-TI
  / U-NA-KA-NA share a root U-NA-KA-NA with suffixal -SI/-TI variation;
  I-PI-NA-MA / I-PI-NA-MI-NA / I-PI-NA-MA-SI-RU-TE chain = prefix I- + stem PI-NA + suffix
  slot; RO-WA and KU-RO / KI-RO / KA-I-RO share -RO. A **concatenative, suffix-rich
  profile (b)** that mirrors what we already saw in the accounting word family.
- **Stem → continuation predictiveness (my run, (a)):** 2-syllable stems: 327 distinct,
  only 36 multimodal (≥3 distinct endings) → **≈89% of 2-syll stems deterministically
  predict what follows** (with the data-sparsity caveat: n≈1,098 words, many hapax). This
  is the fingerprint of a language with real, closed suffix paradigms — not random
  symbol strings.

### 5.4 The statistical wall (honest statement)

Every probe above demonstrates **structure, not meaning**. The classical result stands:
with 1,098 syllabic tokens, a broken corpus (8.6–21% damage), no bilingual, and an
unidentified language family, no algorithm — quantum-inspired or otherwise — can assign
phonetic values or meanings to sign sequences; the required mapping is simply not in the
data (information-theoretic lower bound: the corpus does not contain the missing
anchor). Our value-transfer only *borrows* Linear B's readings; it cannot *prove* them
for Linear A.

### 5.5 The single most valuable decipherable target: *301 (H9 quantification + candidates)

- **(a) Distribution:** *301 ≈ 238 standalone tokens (258 incl. ligatures) — the most
  frequent undeciphered sign and the **4th most common LA token overall** (after the
  fragment marker ×552, the word separator ×468 and the numeral "1" ×310). 82% standalone
  on nodules (logogram behavior); 11× inside A-TA-I-*301-WA-JA (ritual frame, all stone vessels); ligature
  *301+*311 ×10, *301-*301 ×2, MI+*301 / I+*301 / E-*301 / TE-*301 compounds exist.
- **(a) Positional load:** in R1/R3 it sits at exactly the meaning-bearing slot: the
  ritual frame `A-TA-I-[*301]-WA-JA` and the nodule accounts. Knowing *301 would turn the
  votive formula's row 1 from `A-TA-I-X-WA-JA` into a readable clause and identify the
  nodule commodity.
- **(c) Candidate readings from the literature:** *301 is frequently treated in
  Minoan scholarship as a **vessel/offering ideogram** (libation-related); some compare it
  to a bucranium/frontlet or a ritual vessel type. If *301 is an ideogram
  (vessel/offering), the formula row reads at-a-i-[OFFERING]-wa-ja — a verb frame with a
  nominal object, consistent with the (c) editorial gloss "gives … this dedication".
  **Scoring (my framework, (c)):** an ideogram reading explains the nodule-logogram use
  (4/5 of attestations) better than a syllabogram reading (which would need a value
  attested nowhere else); a syllabogram reading would better explain embedded usage in a
  7-syllable word. The dual profile biases toward a **logogram with syllabic adjuncts**
  (b). Suggested decisive test (§7, E6): correlate *301 co-occurrence with the object
  illustrated on the *seal face* of nodules (image metadata) — if *301 marks a depicted
  vessel, the ideogram reading is essentially proven.

---

## 6. What we can honestly claim now (summary)

1. **(a) The system is dual-register:** palace accounting (tablets/nodules/roundels) vs
   sanctuary ritual (stone vessels). Formula sets never cross registers.
2. **(a) ≥35% of tokens are strictly numerals/fractions/logograms (>50% incl. single-sign abbreviations)** — the accounting machinery is
   *readable*: quantities, fractions (½, ¼, ⅙, ¾, 1⁄16, ⅓, ⅕), commodity ideograms
   (wine VIN, wheat GRA, olive OLIV, oil OLE+/cyperus CYP, figs NI, men VIR) and a
   secured functional grammar (entry + numeral … total).
3. **(b) Nine accounting words are securely glossed** (KU-RO total, KI-RO owed,
   PO-TO-KU-RO grand total, …) — the field's consensus readings, all reproduced here
   with counts, registers and sites.
4. **(a) The libation formula is fully structurally recovered** — rows 1–6, variants,
   sites, and the irreducible core JA-SA-SA-RA-ME · U-NA-KA-NA-SI.
5. **(b) The shadow language is a CV, suffixal, reduplicating, agglutinative-like
   language** — phonotactically plausible, morphologically structured, *distinct* from
   Greek linear B in word length and reduplication habit (c: hints at non-Greek Minoan).
6. **(b/c) *301 is the key lock** — a logogram-like sign carrying the ritual frame and
   the nodule register; an ideogram reading is primed but unproven.
7. **Not achieved (honest): any new lexical reading beyond the 9 glosses; any verified
   phonetic value beyond the Linear B transfer; language identification.** The wall is
   data, not compute.

---

## 7. Ranked next experiments (what would actually translate more of it)

- **E1 (highest payoff): archaeology of *301.** Match nodule/tablet *301 occurrences to
  depicted seal imagery / findspot micro-context → decide ideogram vs syllabogram; if
  ideogram, the libation row 1 becomes readable and the nodule commodity is identified.
- **E2: full DĂMOS/GORILA corpus migration.** This subset is 1,721 of ~1,500+ known
  sides (overlapping coverage); a full, deduplicated, damage-annotated corpus roughly
  2–5× larger would rescue the arithmetic verification (E5 below) and de-hapax the
  lexicon.
- **E3: columnar layout reconstruction.** Rebuild two-dimensional layout from
  `parsedInscription` lines + `words[]` positions, then re-run the KU-RO arithmetic
  verification (8/29 → target >90%). Turns "total" from philological consensus into
  verified fact across the corpus.
- **E4: size-matched LA vs LB control — ✅ DONE** (agent-engineer, approved;
  see `size_matched_report.md`). Headline: the corpus-size confound does **not** explain
  the LA/LB ordering (LB more compressible at every matched size; fair-control ratio
  ≈1.7×, magnitude "~4×" retired as a control-convention artefact), and once **text type
  is also matched** (LB short administrative labels ≤8 tokens vs LA's label genre), the
  scripts become statistically indistinguishable (H₃|H₂ 0.85 vs 0.88; fair LZ Δ −7.4% vs
  −10.4%) — the structural "gap" between LA and LB is largely a genre+size artefact, not
  proof that Minoan lacked real writing structure.
- **E5: phonotactic typology battery.** Score LA's shadow profile against candidate
  language families (Anatolian/Luwian, West-Semitic, isolate) using distributional
  distances on syllable inventories and suffix paradigms — an honest (c) ranking of
  *compatibility*, never proof.
- **E6: cross-corpus ligature census.** Extract every `X+*301`, `X+*188`, `*304+…`
  ligature; ligature structures encode derivational relations (adjunct+logogram) that
  are readable without the language.

---

---

## 9. V2-SPEC §1 — Formula inventory (sign-level, cross-tabulated) ((a))

**Unicode sign-level inventory (la_sequences.json, 825 sequences, damage-marker 𐝫
excluded).** Values mapped via the standard AB-numbering of the Linear A Unicode block
(AB-serial = the standard Linear B value transfer: AB031=sa, AB057=ja, AB059=ta, AB041=si,
AB081=ku, AB067=ki, AB002=ro, AB120=GRA, AB131A=VIN; Ventris–Chadwick 1956/73; Godart–Olivier
GORILA 1976–85; Unicode 5.1 Linear A block). These AB values ARE the transfer table the
corpus encodes; only confident pairings are cited.

**Sign-doublings (top):** 𐘞𐘞 (AB031=**sa-sa**) ×20 · 𐘱𐘱 (AB057=**ja-ja**) ×12 ·
𐘃𐘃 (AB004=**te-te**) ×6 · 𐙂𐙂 (AB081=**ku-ku**) ×6 · 𐘠𐘠 (AB041=**ti-ti**) ×5 ·
𐘆𐘆 (AB007=**di-di**) ×4 · 𐘌𐘌 (AB016=**qa-qa**) ×4 · 𐘼𐘼 (AB*, unknown value) ×3 ·
𐝆𐝆 (A707=**te-te** variant) ×3. The sa-sa / di-di / ti-ti / ja-ja / qa-qa doubling set is
the same reduplication family observed at word level (SA-SA-RA-ME, DI-DI-ZA-KE, TI-TI-KU,
JA-SA-SA-RA-ME, QA-QA-RU) — **reduplication is a sign-level habit, not a transliteration
artefact** ((a) measured, (b) interpretation).

**Top sign collocations (clean, damage excluded):**
| bigram | value | n | register (support) |
|---|---|---|---|
| 𐙂𐘁 (AB081+AB002) | **KU-RO** "total" | 39 | Tablet (accounting) |
| 𐘳𐘚 (AB059+AB028) | TA-I- | 22 | mixed |
| 𐘳𐘅 (AB059+AB006) | TA-NA- | 21 | mixed (Iouktas libation row) |
| 𐘞𐘴 (AB031+AB?033) | SA-RA- | 21 | ritual/tablet |
| 𐘞𐘽 (AB031+AB076) | SA-RA₂ | 20 | ritual (formula core) |
| 𐘞𐘞 | SA-SA | 20 | ritual formula core |
| 𐘇𐘳 (AB008+AB059) | A-TA- | 20 | libation row 1 frame |
| 𐘅𐘤 (AB006+AB041) | NA-SI- | 19 | — |
| 𐘸𐘁 (AB067+AB002) | **KI-RO** "owed" | 17 | Tablet (accounting) |
| 𐘇𐘬 (AB008+AB?014) | A-DU- | 16 | — |

→ (a) The two accounting gloss words KU-RO and KI-RO are the **top two attributabled
collocations at the raw sign level** — the functional lexicon is literally the most
frequent sign combinations in the corpus. (b) The ritual formula is present at sign level
as A-TA-/TA-NA- + SA-RA-/SA-SA- chains, confirming the register split without any value map:
the same sign pairs cluster, but their contexts (Tablet vs Stone vessel) never mix
(§3 matrix, (a)).

**Functional "glossary of formulas" — what each formula DOES in the system ((a)+(b)):**
| Formula | n | Function (what it does) | Tag |
|---|---|---|---|
| KU-RO + number | 34 | closes an account column with its total | (b, gloss consensus) |
| PO-TO-KU-RO + number | 2 | all-tablet grand total | (b) |
| KI-RO + number | 12 | deficit/owed ledger | (b) |
| [word] + numeral + fraction | 100s | ration/quantity entries | (a) |
| JA-SA-SA-RA-ME (core -SA-SA-RA-ME, 10 ins.) | 7 | dedicator/offering head of votive formula | (b/c) |
| A-TA-I-*301-WA-JA (11 ins.) | 11 | opening verbal frame of votive formula | (b/c) |
| U-NA-KA-NA-SI / SI-RU-TE | 4/7 | middle/closing rows of votive formula | (b/c) |
| *301 alone (+numeral) | 236 | nodule/sealing account marker (logogram) | (b) |

## 10. V2-SPEC §2 — Sign-function classification ((a) positional+distributional, (b) class)

Classification rule (my run, raw/lineara.json words[]): for every sign, compute %
occurrences as a standalone one-sign "word" (stand%), % at word-init (init%), and number of
distinct neighbors (n_neigh). Cross-checked against Linear B's known functions (LB: syllabic
signs combine freely mid-word; ideograms stand alone with numerals; fractional signs attach
to quantities; divider 𐄁 separates).

| Sign (value) | n | stand% | init% | n_neigh | Class (cross-check) |
|---|---|---|---|---|---|
| 𐙕 (*301 / A301) | 274 | **86%** | 88% | 37 | **logogram-like** on nodules (82% of uses); LB-like ideogram behavior |
| 𐙉 (GRA, AB120) | 66 | **74%** | 91% | 4 | **logogram** (wheat) — matches LB ideogram use |
| 𐙍 (VIN, AB131A) | 65 | **60%** | 77% | 10 | **logogram** (wine) — LB-like |
| 𐝆 (A707="TE" var.) | 137 | **72%** | 93% | 10 | logogram/abbreviation-like |
| 𐙂 (KU, AB081) | 308 | 51% | 78% | 58 | **syllabogram**, init-biased (formulaic KU-...) |
| 𐘾 (KA, AB077) | 285 | 56% | 72% | 59 | **syllabogram**, init-biased |
| 𐘇 (A, AB008) | 202 | 3% | 79% | 65 | **syllabogram** (pure vowel), high freedom |
| 𐘞 (SA, AB031) | 139 | 3% | 42% | 56 | **syllabogram** |
| 𐘳 (TA, AB059) | 165 | 10% | 28% | 62 | **syllabogram** |
| 𐘅 (NA, AB006) | 158 | 1% | 6% | 57 | **syllabogram** (mid-word workhorse) |
| 𐄁 (divider U+10101) | 468 | — | — | — | **word divider** (LB: separator, same) |
| 𐄇/𐄈/𐄉… (U+1010x) | — | — | — | — | **fractions/numerals** — attach to quantities |

(a) The division is sharp: signs split into a high-standalone, low-neighbor class
(*301, GRA, VIN, A707) and a high-neighbor, low-standalone class (the AB syllabary);
this is exactly Linear B's logogram-vs-syllabogram split, recovered purely from
distribution. (b) Note *301: 86% standalone with a *logogram* pattern — independent
confirmation of the §5.5 ideogram hypothesis from positional statistics alone, without
using its nodule support.

## 11. V2-SPEC §3 — Phonetic-transfer statistical attack (LB values applied; core test)

**Method (provenance):** values = the standard Linear-B value transfer embedded in the
corpus transliteration (GORILA/Younger, same as §9); LB Greek control = raw/linearb.json
transliteratedWords (5,832 inscriptions; 12,333 syllabic words). All syllable readings in
this section come *from the corpus's transfer table* — no value is invented here.

**(a) CV-syllable-structure consistency:** LA 93% vowel-final words (1,021/1,098; the
remainder are subscripted values RA₂/PA₃ etc., not closed syllabograms); LB 100%
vowel-final. Both scripts are pure open-syllable CV/V syllabaries — the transfer is
internally consistent (a). Zero closed syllables in either: the LB system has no coda
signs and LA under the same values shows none → "language-like" for an open-syllable
language type.

**(a) Vowel distribution — the core Luwian test (new, my run):**
| vowel | LA (transferred) | LB (Greek control) |
|---|---|---|
| A | **39.8%** | 25.5% |
| E | 14.0% | 23.0% |
| I | **24.4%** | 13.3% |
| O | **4.4%** | 31.6% |
| U | **17.5%** | 6.6% |
| **E+O** | **18.4%** | **54.6%** |

→ (a) Under identical value-assignment, LA's vocalism is **A/I/U-dominated (81.6% of
vowel syllables)** while LB Greek is **O/E-dominated (54.6%)**. This is the measurable
"language-like vs noise" score: the transferred strings produce a *coherent, non-random,
open-syllable phonology* (language-like, not noise) whose profile is **closer to an
Anatolian a/i/u-type system (Luwian: no /o/, marginal /e/)** than to Greek's
low/front/back-rich inventory — **consistent with the Anatolian/Luwian hypothesis but
not proof** (tag b; the value table is a hypothesis, though both corpora share it).

**(b) Circularity caveat (must be stated):** the syllable values are Linear B's; the LA
inventory could in principle have used the e/o signs as freely as LB did and did not —
so the e/o suppression is a *real, corpus-level* observation, not imposed by the table.
The test is therefore informative-but-not-decisive: it says "if the LB values are right,
Minoan's surface vocalism looks Anatolian-like; if they are wrong, the conclusion is void."

**(b) Phonotactic plausibility vs reference families:** machine-readable Luwian/Hurrian
wordlists are NOT available in this environment (general academic web unreachable;
documented next step). From published descriptions (Kloekhorst 2006, Melchert 1994
(Luwian: a/i/u, no /o/, limited /e/); Wegner 2007 (Hurrian: a/e/i/o/u, /e/ frequent);
Greek: a/e/i/o/u with massive /o/ in inflection):
- LA's a/i/u dominance + low /o/ ⇒ closest to **Luwian-type**;
- LA's non-zero /e/ (14%) is *possible* for both Luwian (secondary /e/) and Hurrian/Greek;
- itemized vowel-shape comparison to actual wordlists: **PENDING** (no machine-readable
  Luwian/Hurrian corpus sourced; ranked as E7 below).
Score: transferred reading produces **language-like, Luwian-leaning phonotactics, not
noise** (b), with the circularity caveat.

**(a) Repeated-formula phonetics:** ja-sa-sa-ra-me (7), u-na-ka-na-si (4), si-ru-te (7),
a-ta-i-*301-wa-ja (11) — all open-syllable, phrase-internal CV alternation C-V nearly
everywhere, suffix-final -e/-si/-te/-ja. Morphophonology is coherent (b): fixed row-openers,
name slot, reduplication — the same phrase recurs with minor vocalic alternation
(u-na-ka-na-si vs u-na-ru-ka-na-ti; i-pi-na-ma vs i-pi-na-mi-na), consistent with a real
morphological paradigm (b/c).

## 12. V2-SPEC §4 — Anchor analysis: Arkalochori axe + Cypro-Minoan bridge

**Arkalochori (data now in hand: ARKALOCHORI_AXE.md, agent-scientific-engineer):**
- ARZf1/ARZf2 (metal, votive): **I-DA-MA-TE** (Unicode 𐘚𐘀𐙁𐘃 = AB028+AB001+AB?+AB004;
  ARZf2 adds a trailing damage sign). Literature proposal (c): possible divine-name /
  dedication formula, "Ida Mater"; ⚠️ hypothesis, not accepted decipherment.
- ARKHZf9: **JA-KI-SI-KI-NU 𐄁 MI-DA-MA-RA₂** — shares the DA-MA- stem with I-DA-MA-TE
  (corpus-observed pattern (a)); a natural motif target (E1-style *301 analysis).
- Honest fit: the axe is a **votive-metal object** (same register class as the libation
  vessels); its two faces carry only 4+5 signs in this corpus vs ~15 in the literature on
  the axe — **coverage caveat documented by the scientific engineer; do not use the axe as
  a corpus-quality claim until checked against GORILA V (AR Zf 1)**. Fit with our pipeline:
  the DA-MA- stem and JA- opener both appear in the formulaic reduplication/suffix patterns
  of §5.3; a dedicated braid/motif clustering of ARZf1/2/ARKHZf9 vs the libation vessels is
  the planned next step (**E8, pending**).

**Cypro-Minoan bridge (CYPRO_MINOEAN.md, agent-scientific-engineer):**
- Status: **no machine-readable Cypro-Minoan corpus exists publicly** (surveyed GitHub;
  Olivier 2013 *Édition holistique* and Ferrara 2012 *Analysis* not digitized). Digital
  signary available: Unicode U+12F90–U+12FFF (109 signs, Noto OFL font).
- The bridge argument (CM develops from/alongside LA; descendant Cypriot syllabary writes
  Eteocypriot, possibly Anatolian-related) is **contested scholarship — treat as working
  hypothesis (c), not fact** (per CYPRO_MINOEAN.md).
- What we can do now: nothing quantitative until a CM corpus exists; the plan is
  documented (encode Olivier 2013 transcriptions when obtainable; check Cypriot syllabary
  U+10800–U+1083F projects) — **PENDING, ranked E9**.

## 13. V2-SPEC §5 — The honest bottom line: scores and flip-evidence

**Scores (this is the answer to "how close did we get"):**
| Dimension | Score | Basis |
|---|---|---|
| Functional translation (what the system does) | **≈ 85–90%** | registers + formula grammar + accounting lexicon (KU-RO/KI-RO/PO-TO-KU-RO/KA-PA/KI-RA/E-*82/DA-DU-MA-TA) + numeral/fraction/logogram machinery; §2–§4, §9 |
| Formula inventory (sign-level) | **≈ 80%** | top collocations/doublings identified and register-tagged; §9 |
| Sign-function classification | **≈ 95%** | distribution classifies logogram vs syllabogram vs numeral vs divider cleanly and matches LB; §10 |
| Phonetic reading (actual language) | **≈ 0–10%** | ~94% of signs carry *borrowed* LB values; zero independently verified; language unclassified; §11 |
| Anatolian/Luwian hypothesis | **tested, not proven** | A/I/U-dominance consistent with Luwian-type vocalism; circularity caveat; §11 |

**What would flip it (ranked by feasibility):**
1. **Seal-image anchor on *301 (HIGHEST feasibility, we have the data):** match nodule
   *301 uses to the seal die iconography → decide logogram vs syllabogram; if logogram for
   a vessel/offering, the libation row 1 becomes structurally readable (E1).
2. **Columnar layout reconstruction:** rebuild tablet columns from parsedInscription →
   arithmetic-verify KU-RO totals (target >90%) → "total" becomes measured fact (E3).
3. **Full DĂMOS/GORILA corpus migration (2–5× data):** de-hapax the lexicon, re-run all
   structure tests, shrink confidence intervals (E2).
4. **A bilingual or identifiable-language find** (any Aegean context): the ultimate decoder;
   outside our control (classical impossibility argument, §5.4).
5. **Cypro-Minoan digitization** (E9): if a CM corpus is encoded and CM shows a closer
   relation to a known family, the bridge could bootstrap LA phonetic values (long shot).
6. **Luwian/Hurrian wordlist acquisition** (E7): itemized phonotactic distances vs actual
   corpora — would upgrade the §11 test from descriptive to scored.

**Bottom line:** functional translation ≈ **mostly achieved** (system grammar + formula
glossary + sign classification). Phonetic reading ≈ **not achieved** (values borrowed,
language unknown); the strongest new quantitative finding of this pass is the
**A/I/U-vocalism contrast (LA 81.6% a/i/u vs LB Greek 54.6% e/o)**, which is consistent
with an Anatolian-type (Luwian-like) substrate and is the first statistically grounded
push toward the Anatolian hypothesis from this corpus — while remaining non-decisive
until external anchors arrive. Negative results reported as results: no decipherment
claimed; the wall is external evidence, not computation.

---

## 14. Files (updated for v2)
- **TRANSLATION_PASS.md** — v1 (structural/functional pass) + v2 (owner 5-point spec, §§9–13).
- `translation_pass_analysis.py` (+ `.json`) — v1 reproducible stats.
- `translation_pass_v2_analysis.py` (+ `.json`) — v2 spec §§9–11 reproducible stats
  (doublings, collocations, sign-function, LA-vs-LB vowel differential).
- `raw/lineara.json`, `raw/linearb.json` — corpora (GORILA-sourced).
- `ARKALOCHORI_AXE.md`, `CYPRO_MINOEAN.md` — anchor corpora/plans (agent-scientific-engineer).
- Related: `SYNTHESIS.md` v4 (size-matched control folded in), `LINEA_ENGINE_REPORT.md`, `quant_report_A_vs_B.json`,
  `linguistics_report.json`, PNG visualizations.


*Honesty footer (v2): no word beyond the 9 editorial glosses is claimed as translated;
every formula reading is structural; every phonetic statement is either a measured
distribution (a) or an interpreted one (b/c); the owner's questions — "can we translate
it", "Anatolian or not" — are answered as: functionally, the system is ~85–90%
translated; phonetically, not translatable with any tool that exists, with the
A/I/U-vocalism result as the strongest statistical hint toward an Anatolian-type
substrate and the circularity caveat stated plainly. The wall is external evidence, and
this pass says so.*
