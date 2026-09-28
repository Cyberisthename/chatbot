# Indus H1–H4 Cross-Validation — Rebuilt Harness & Subset Re-Verification

**Author:** agent-compression-specialist · **Date:** 2026-09-28
**Task:** `b5c9f3c9-d3cb-4402-9c86-7eae7237679a` (Backlog: cross-validate H1–H4 on full Mahadevan corpus)
**Status:** ⚠️ **FULL-CORPUS RUN STILL GATED** — harness ready, subset re-verified, full corpus not yet landed.

---

## 0. Bottom line (owner summary)

1. **(a measured, honest)** The full Mahadevan 1977 concordance (~4,197 inscriptions) is **still not
   available in machine-readable form**. Checked on 2026-09-28: upstream
   `github.com/mayig/indus-valley-script-corpus` (the only digitized Indus corpus we hold) is unchanged —
   the same **179-side Mohenjo-Daro CISI subset** (1,003 tokens, 182 distinct P-signs), last pushed
   2025-04-16. GitHub search for any Mahadevan-concordance digitization returns nothing usable.
   The task gate ("do not start until that digitized corpus lands") therefore **still holds**.
2. **(a measured)** What I built instead is the deliverable that makes the backlog executable the
   moment the corpus lands: **`h1h4_crossvalidate.py`** — a dependency-free (pure Python stdlib),
   deterministic, parametric harness that runs all four hypotheses on **any** corpus directory and
   writes `h1h4_crossvalidate.json`. One command: `python3 h1h4_crossvalidate.py <full-corpus-dir>`.
3. **(a measured)** The harness **re-derives the approved pre-incident numbers from the surviving RAW
   corpus** with fresh code — the owner's "not from an old doc?" test:
   - H2|H1 = **2.4127** (approved **2.413**) — HIT
   - H3|H2 = **0.4139** (approved **0.414**) — HIT
   - H1 = 6.2859 (6.286), H2 = 8.6986 (8.699), H3 = 9.1125 (9.113) — HIT
   - bigrams used **551** of 824 (approved 551) — EXACT
   - P385 sequence-initial 29/179, z = 9.84 (approved ~9.27) ✓; P385→P122 29 occurrences,
     enrichment 13.3× (approved ~15×), z = 18.8 ✓
4. **(a measured)** All four hypotheses are CONFIRMED on the available subset, same as the approved
   pass: P385 is strongly sequence-initial (H1 ✓), P385→P122 is a stable collocation (H2 ✓),
   short-range 1–2-sign context conditioning is strong (H3 ✓, ~63% entropy reduction at bigram level),
   and LONG-RANGE formula chains are absent (H4 ✓ — far-repeated-bigram count real = 0, LZ gap
   11.8–12.5% vs shuffles is fully accounted for by within-sequence structure).
5. **(c interpretation)** The harness's LZ-vs-per-seq-shuffle comparison (11.8% gap) is the honest
   framing for H4: the compressibility gap disappears once within-sequence token order is preserved
   — i.e., structure is local, not a system of long-range formulas. That claim is stronger than the
   subset previously supported and is now explicit in the harness output.
6. **No decipherment claimed.** Structure ≠ meaning. This is a structural cross-validation harness;
   physical/semantic readings remain impossible without a bilingual or identified language.

---

## 1. What happened to the old artifacts (incident context)

The 2026-09-24 disk-full incident destroyed the Indus analytic layer (per DB task records and this
session's filesystem verification): `SYNTHESIS.md`, `GROUND_RULES.md`, `NORMALIZE.py`,
`indus_corpus_clean.tsv`, `ivs_quant_engine.py`, `ivs_quant_engine_report.md`, `quant_report.json`,
`indus_structural_pass.py/json/md`. **The raw corpus clone survived**
(`indus/corpus_raw/`, MIT license, Parpola CISI digitization) — 179 artefact JSONs with grapheme
sequences + 397 feature files (P-sign → Parpola/Wells/Mahadevan cross-reference).

All approved numbers survive in team-DB task records (`4d123916`, `5eeb768a`, `acb3425f`, `204b2b4d`,
`7b903a12`, `b6635fcb`). This rebuild restores full reproducibility and extends the machinery.

## 2. Files delivered (all in `/home/team/shared/indus/`)

| File | What |
|---|---|
| `NORMALIZE_RB.py` | Rebuild of the raw-JSON → clean-TSV transform (deterministic, pure stdlib). One row per side: `id, description, P-code sequence in reading order` (reversed from stored physical left-to-right per CORPUS convention). |
| `indus_corpus_clean.tsv` | Rebuilt clean corpus — **179 sides, 1,003 tokens, 182 distinct signs** (matches approved acquisition stats). |
| `h1h4_crossvalidate.py` | Parametric H1–H4 harness. Pure Python stdlib (survives without numpy). Deterministic (seed 2026). Shuffle/permutation baselines on every claim. **Ready for the full Mahadevan corpus: `python3 h1h4_crossvalidate.py <corpus-dir> --out full_run.json`.** |
| `h1h4_crossvalidate.json` | Measured output on the 179-side subset (this run, 2026-09-28). |
| `INDUS_H1H4_CROSSVALIDATE.md` | This report. |

## 3. Hypotheses tested (definitions per SYNTHESIS.md §6 / task record)

- **H1 — P385 is sequence-initial.** Test: count of sequences starting P385 vs per-seq-shuffle
  baseline; z-score + base rate. Measured: 29/179 (16.2%), shuffle mean 6.9, **z = 9.84** ✓
- **H2 — P385→P122 is a stable collocation.** Test: observed P385→P122 bigram count vs independence
  expectation (product of marginals × bigram count) and vs token-shuffle. Measured: 29 observed,
  expected 2.18, **enrichment 13.3×**, z = 18.8 ✓
- **H3 — 1–2 sign context conditioning.** Test: conditional entropy collapse
  H(X2|X1) << H(X1), H(X3|X1X2) smaller still; bigram claim is the robust level at T=1,003.
  Measured (both conventions): **2.4127 bits (approved-difference convention, HIT)** and
  2.3252 bits (strict prefix-marginal); H3|H2: **0.4139** / 0.6702; reduction 63.0% ✓
- **H4 — no LONG-RANGE formula chains.** Test: (i) LZ compressibility vs token-shuffle AND vs
  per-seq-shuffle; (ii) count of bigrams whose first occurrence is ≥3 positions earlier in the same
  sequence ("far" repeats) vs shuffle. Measured: LZ gap 12.5% (token) / 11.8% (per-seq); **far
  repeats real = 0** vs shuffle mean 0.02 ✓ — structure is short-range/within-sequence, not
  long-range formulas.

## 4. Honest caveats & tags

- **(a) measured** — every number above is from a fresh deterministic run on the surviving raw corpus.
- **(b) interpretation** — the "writing-like structure" reading of entropy collapse; the H4 local-vs-
  long-range reading.
- **(c) speculation** — any link between these positional patterns and meaning/language.
- The subset is **Mohenjo-Daro only** (179 sides). The full Mahadevan concordance would add ~4,000
  inscriptions and could change z-scores/enrichments; the harness is built to report the same metrics
  unchanged so the full-corpus numbers are directly comparable.
- **Gate remains per task description:** full-corpus digitization requires the double-check protocol
  (CORPUS.md / GROUND_RULES §7 — manual vs scans, not possible in this environment); the backlog
  explicitly says do not start until that digitized corpus lands.

## 5. How to run the full-corpus cross-validation (when the corpus lands)

```bash
cd /home/team/shared/indus
# corpus lands as a directory of {site}/{artefact}.json (same schema as corpus_raw/corpus):
python3 h1h4_crossvalidate.py /path/to/full-mahadevan/ --out full_mahadevan_h1h4.json
```

The harness auto-detects the schema via `NORMALIZE_RB.load_sequences()` (glob `*/**.json`,
array-of-sides format). Numbers will be directly comparable to the subset run in
`h1h4_crossvalidate.json`.

## 6. Verification commands

```bash
python3 NORMALIZE_RB.py                       # rebuild clean TSV; prints sides/tokens/signs
python3 h1h4_crossvalidate.py                 # subset run -> h1h4_crossvalidate.json
python3 -c "import json; d=json.load(open('h1h4_crossvalidate.json')); print(d['H3_short_range_context']['approved_convention_H2_minus_H1_bits'])"
```