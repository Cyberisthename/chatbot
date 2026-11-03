# Indus structure-first probe — re-derived, re-runnable

**Script:** `scripts/indus_structure_probe.py` · **Results:** `docs/artifacts/trifecta/indus_structure_results.json`
**Corpus:** `docs/artifacts/indus/indus_corpus_clean.tsv` (committed: 179 Mohenjo-daro sides, 1,003 P-code tokens, 182 distinct signs, corpus sha16 `0db6fbd8f99df5f3`)
**Status:** every number below is measured by a fresh run of the committed script on the committed corpus. Nothing is tuned toward a previous figure.

**Hard boundary.** Structure ≠ meaning. Nothing here reads, translates, or assigns a value to a sign. The Indus script has no bilingual and no identified language, so no decipherment is claimed.

## Reproduce every number

```bash
python3 scripts/indus_structure_probe.py
```

Machine-readable form (what the numbers below were taken from):

```bash
python3 scripts/indus_structure_probe.py --json docs/artifacts/trifecta/indus_structure_results.json
```

Deterministic: seed `2026`, no network, pure stdlib. Runtime < 1 s.

## Measured

| Probe | Result | Null / baseline |
|---|---|---|
| **H1** P385 opens inscriptions | **29 / 179** (16.2%) | per-sequence shuffle: 6.89 ± 2.25, **z = 9.84** |
| **H2** P385→P122 collocation | **29** occurrences, **13.31×** enriched | independence expectation 2.18; token-shuffle 2.27 ± 1.42, **z = 18.83** |
| **H3** short-range context | H1 = **6.2859** bits, H2\|H1 = **2.3252** bits (−63.0%), H3\|H1H2 = **0.6702** bits; bigrams used **551** of 824 | — |
| **H3** (difference convention, the one the approved figures were quoted in) | H2−H1 = **2.4127** bits, H3−H2 = **0.4139** bits | — |
| **H4** no long-range formula chains | LZ **0.85543** phrases/token vs whole-corpus token shuffle 0.97879 → **−12.60 %**; vs per-sequence shuffle 0.97045 → **−11.85 %**; far-repeated bigrams (≥3 positions apart) **0** | shuffle mean 0.025 far repeats |
| **C** size-matched compressibility (budget 1,000 tokens, 25 draws) | **0.79988 ± 0.00320** | global shuffle **0.88996 ± 0.00469** → **Δ = −10.12 %** |

`Δ = 100·(C_real/C_shuffle − 1)`; negative means the real sign order carries redundancy that chance does not. This is the same metric, same code and same token budget as the Linear A and Voynich probes, so the three are comparable (`docs/artifacts/trifecta/TRIFECTA_STRUCTURE_PROBE.md`).

## What this reproduces, and what it does not

- **Reproduced, bit-for-bit against the approved Indus figures:** H1 29/179 z = 9.84; H2 enrichment 13.31× (z = 18.83); H1 6.2859, H2 8.6986, H3 9.1125; 551 bigrams used; far-repeated bigrams 0. The difference-convention conditional entropies (2.4127 / 0.4139) also reappear.
- **Not the same harness as before.** The committed `docs/artifacts/indus/h1h4_crossvalidate.py` still defaults to `corpus_raw/`, which is *not* in the repo, so it cannot run from a clean checkout. This script reads the committed TSV instead; that is the only methodological difference, and the numbers agree.
- **Coverage limit unchanged:** 179-side Mohenjo-daro subset, not the full Mahadevan concordance. Full-corpus numbers remain gated on a digitised Mahadevan corpus.
- **(b) interpretation, not measurement:** the "writing-like structure" reading of the entropy collapse, and the local-vs-long-range reading of H4. (c) Any link between these positional patterns and meaning is speculation and is not made here.
