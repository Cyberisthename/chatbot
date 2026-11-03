# Voynich structure-first probe — re-derived, re-runnable

**Script:** `scripts/voynich_structure_probe.py` · **Results:** `docs/artifacts/trifecta/voynich_structure_results.json`
**Status:** every number below is measured by a fresh run. Nothing is tuned toward a previous figure.

**Hard boundary.** Structure ≠ meaning. No Voynich word is read, sounded or translated here. The manuscript has no bilingual, so no decipherment and no phonetic reading is claimed.

## Corpus: why it is fetched, not committed

Third-party transcription, not committed. Fetched on first run into `/var/tmp/trifecta_corpus`, sha256 recorded:

`musyoku/voynich-transcription` → `voynich.txt` — Glen Claston's **v101** transcription, word-separated, one line per manuscript line (`<folio.line>w.w.w-$`). sha256 `0a96fbcf38e29868…`. Parsed: **4,486 lines, 38,242 words, 11,293 distinct words, 171 distinct letters**.
This is **not** the EVA (Zandbergen/Landini) alphabet, so letter-level numbers are transcription-specific and are reported as such. Offline: `--voynich PATH` with a local copy.

## Reproduce every number

```bash
python3 scripts/voynich_structure_probe.py
# or, with the results file this report quotes:
python3 scripts/voynich_structure_probe.py --json docs/artifacts/trifecta/voynich_structure_results.json
```

Deterministic (seed 2026, 25 null draws, z threshold 3.0).

## Measured: five probes against explicit nulls

| Probe | Real | Null (mean ± sd) | z | rejects null? |
|---|---|---|---|---|
| **V1** letter mutual information I(X1;X2) | **1.2133 bits** | 0.0284 ± 0.0030 | **+398.3** | yes |
| **V2** distinct-word share of tokens | **0.2953** | 0.7462 ± 0.0014 | **−321.6** | yes |
| **V3** letter I(X1;X2) vs within-word **anagram** null | **1.2133 bits** | 0.2624 ± 0.0046 | **+205.3** | yes |
| **V4** first-letter minus last-letter entropy | **+0.7743 bits** | 0.0006 ± 0.0095 | **+81.7** | yes |
| **V5** size-matched compressibility Δ (1,000-token budget) | **−0.369 %** | 0.0 (shuffle sd 0.123 %) | **−2.98** | **no** |

Nulls: V1, V2, V4 use the **random-scribble model** — the same words at the same positions, each letter drawn i.i.d. from the corpus's own letter unigram distribution (so letter frequencies and word lengths are matched by construction and cannot explain a rejection). V3 shuffles letters *within* each word, keeping each word's letters exactly. V5 uses a global token shuffle, the same C and Δ metric used by the Indus and Linear A probes.

**Summary: 4 of 5 probes reject the random-scribble model; V5 does not.**

## What this reproduces, and what it does not

- **The approved claim was "statistically inconsistent with random-scribble hoaxes (5 independent probes)".** Re-derived honestly: **4 of 5**, not 5 of 5. The four that reject do so overwhelmingly (|z| from 82 to 398): letter order inside words, letter dependency, heavy word repetition, and a strong first-vs-last positional grammar are all incompatible with i.i.d. scribble.
- **The failing probe is a real finding, not a bug:** Voynich **word order** carries almost no repetition-driven compressibility — Δ = −0.37 % against a shuffle of the same words at the same budget, |z| = 2.98, just under threshold. Word-level redundancy in this transcription is ~0 while its letter-level redundancy is large (see the letter-level row of `TRIFECTA_STRUCTURE_PROBE.md`: Δ = −17.7 %). That is consistent with the known "words are structured, sequences of words are nearly free" character of the manuscript, and it means the *word-order* probe should not be quoted as evidence of non-scribble.
- **Scope of the null:** this tests **one explicit** scribble model (i.i.d. letters with matched frequencies and lengths). It does not test every hoax construction, and rejecting it is **not** evidence of a language, a cipher, or a meaning.
- **Honest negative restated:** a phonetic reading remains impossible without a bilingual. Nothing here moves that line.
