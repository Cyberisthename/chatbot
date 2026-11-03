# Linear A structure-first probe — re-derived, re-runnable

**Script:** `scripts/lineara_structure_probe.py` · **Results:** `docs/artifacts/trifecta/lineara_structure_results.json`
**Status:** every number below is measured by a fresh run. Nothing is tuned toward a previous figure.

**Hard boundary.** Structure ≠ meaning. No value, sound or gloss is assigned to any sign here, and no decipherment is claimed — Linear A has no bilingual and no identified language.

## Corpus: why it is fetched, not committed

The Linear A and Linear B corpora are **third-party epigraphic data** and are not committed to this repo. The script downloads them on first run into a cache directory and records a sha256 for each file, so any run is identifiable:

| Corpus | Source | sha256 (this run) |
|---|---|---|
| Linear A | `mwenge/lineara.xyz` → `items_analysis/inscriptions.json` (Dāmos-style transliterations; repo carries a public-domain-style "share freely" blessing) | `66b94203bd343ed9…` |
| Linear B | `mwenge/linearb.xyz` → `LinearBInscriptions.js` (same schema; used here only as the **deciphered-language control**) | `0967d75699b81488…` |

Records parsed: Linear A **1,720** inscriptions, Linear B **5,889**. Offline use: `--lineara PATH --linearb PATH` with local copies of those two files; nothing else changes.

## Reproduce every number

```bash
python3 scripts/lineara_structure_probe.py
# or, with the results file this report quotes:
python3 scripts/lineara_structure_probe.py --json docs/artifacts/trifecta/lineara_structure_results.json
```

Deterministic (seed 2026). Needs network on first run only; then it reads the cache at `/var/tmp/trifecta_corpus`.

## Measured (sign level = word split on `-`; word level = whole transliterated word)

| Corpus / level | tokens | distinct | H1 bits | H2\|H1 bits | C (1,000-token budget, 25 draws) | shuffle null | **Δ** |
|---|---|---|---|---|---|---|---|
| Linear A — sign | 5,456 | 174 | 5.9368 | 4.4162 (−25.6%) | **0.82676 ± 0.00821** | 0.86812 | **−4.76 %** |
| Linear B — sign (control) | 54,873 | 268 | 6.1924 | 4.7621 (−23.1%) | **0.78604 ± 0.04687** | 0.88892 | **−11.57 %** |
| Linear A — word | 3,060 | 1,139 | 8.4071 | 1.6422 (−80.5%) | **0.92896 ± 0.00900** | 0.96408 | **−3.64 %** |
| Linear B — word (control) | 25,241 | 6,035 | 10.0729 | 2.6227 (−74.0%) | **0.90868 ± 0.03177** | 0.98548 | **−7.79 %** |

`Δ = 100·(C_real/C_shuffle − 1)`. Sign-level contrast: **Linear A is 6.81 percentage points less order-redundant than deciphered Linear B** under an identical metric at an identical token budget. (b) Interpretation: the undeciphered corpus carries measurably *less* order-driven redundancy than a corpus we know is language.

## What this reproduces, and what it does not

- **The ordering reproduces, the quoted magnitudes do not.** The approved pair was "LA LZ Δ −7.6 % vs LB −30 %". Re-derived here: **−4.76 % vs −11.57 %** (sign level) and **−3.64 % vs −7.79 %** (word level). Same sign, same direction, roughly half the separation. The old numbers came from a destroyed engine whose LZ normalisation and tokenisation are not recoverable, so **the −7.6 % / −30 % pair stays marked unreproduced**; these are the numbers that now have a script behind them.
- **The ≈85–90 % "functional translation" is not rebuildable as code and is not rebuilt here.** `New folder/TRANSLATION_PASS.md` line 538 states it as a **self-assessment** of the functional pass (registers + formula grammar + accounting lexicon), not as a computed statistic. It is an (interpretation)-tagged judgement over an epigraphic reading; there is no metric to re-run. The one computed figure in that document is the vocalism contrast, and it depends on a **borrowed sign→value table that is not in the repo**, so **81.6 % A/I/U also stays unreproduced** — it can be cited only as a hypothesis-conditioned contrast, never as "81.6 % translated".
- **Sign-value overlap 94.3 %**: same blocker — needs the sign→value table. Unreproduced.
- **(b) interpretation:** reading the LA/LB Δ gap as "unknown script looks less language-like than a deciphered one". It is one control corpus of different size and genre (accounting tablets), and the draw-to-draw spread is large (LB sign-level sd 0.047), so the gap is directional evidence, not a strong claim. (c) Any statement about the *language* of Linear A is outside what this script can support.
