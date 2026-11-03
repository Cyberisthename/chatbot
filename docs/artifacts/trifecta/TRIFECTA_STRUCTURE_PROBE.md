# Three scripts, one method — the trifecta, re-derived

**Script:** `scripts/trifecta_structure_probe.py` · **Results:** `docs/artifacts/trifecta/trifecta_structure_results.json`
**Inputs:** the three probes in this directory (`indus_structure_probe.py`, `lineara_structure_probe.py`, `voynich_structure_probe.py`), loaded as modules so all four rows below come from **identical code**.
**Status:** every number is measured by a fresh run. Nothing is tuned toward a previous figure.

**Hard boundary.** Structure ≠ meaning. No script is translated, no sign is assigned a value, and no decipherment is claimed for Linear A or the Indus script. The method maps the *skeleton* of an unknown writing system; it does not recover meaning.

## Reproduce every number

```bash
python3 scripts/trifecta_structure_probe.py
# or, with the results file this report quotes:
python3 scripts/trifecta_structure_probe.py --json docs/artifacts/trifecta/trifecta_structure_results.json
```

Needs the two fetched corpora in `/var/tmp/trifecta_corpus` (created automatically by the Linear A / Voynich probes; pass `--cache DIR` to move it). The Indus corpus is committed in-repo. Deterministic, seed 2026.

## The one metric, applied identically to every script

`C` = Lempel-Ziv (1976) distinct phrases per token, computed on a fixed **1,000-token budget** over 25 seeded subsamples of whole sequences, with sequence-boundary sentinels so no phrase spans two sequences. `Δ = 100·(C_real/C_shuffle − 1)` against a global shuffle of the same tokens at the same budget: **negative = the real order carries redundancy that chance does not**. Δ is the scale-free comparison — it cancels alphabet size and budget choice.

## Measured

| Script | level | role | tokens | distinct | C | shuffle | **Δ** |
|---|---|---|---|---|---|---|---|
| Indus | sign | undeciphered | 1,003 | 182 | 0.79988 ± 0.00320 | 0.88996 | **−10.12 %** |
| Linear A | sign | undeciphered | 5,456 | 174 | 0.82676 ± 0.00821 | 0.86812 | **−4.76 %** |
| **Linear B** | sign | **deciphered control** | 54,873 | 268 | 0.78604 ± 0.04687 | 0.88892 | **−11.57 %** |
| Linear A | word | undeciphered | 3,060 | 1,139 | 0.92896 ± 0.00900 | 0.96408 | −3.64 % |
| Linear B | word | deciphered control | 25,241 | 6,035 | 0.90868 ± 0.03177 | 0.98548 | −7.79 % |
| Voynich | word | undeciphered | 38,242 | 11,293 | 0.99472 ± 0.00375 | 0.99840 | **−0.37 %** |
| Voynich | letter | undeciphered | 161,419 | 171 | 0.52660 ± 0.01050 | 0.64000 | **−17.72 %** |

**Indus vs the deciphered control (sign level):** Δ −10.12 % vs −11.57 %, **gap 1.45 percentage points**; C 0.7999 vs 0.7860.

## What this says, stated as narrowly as the data allows

- **(a) measured:** at sign level, Indus order-redundancy sits **1.45 pp from the deciphered Linear B control** under an identical metric at an identical budget — i.e. on top of it. Linear A sits 6.8 pp *away* (less redundant than the control), and Voynich at word level is essentially at the shuffle floor (−0.37 %) while at letter level it is the most ordered of the four (−17.72 %).
- **(b) interpretation:** "Indus sits in the deciphered-language band" is supportable in the weak form *Indus's order-driven redundancy is at the deciphered control's level*. It is **not** supportable as a band claim: there is **one** deciphered control, so there is a **point, not a band**, and the LB sign-level draw-to-draw spread (sd 0.047 on C) is wide because 5,000-token draws from 54,873 tokens are noisy. Adding a second deciphered control is the obvious next step to make the word "band" true.
- **(c) speculation, not claimed:** anything about the *content* or *language* of the Indus script, Linear A, or the Voynich manuscript. Structure ≠ meaning; no bilingual exists for two of the three.
- **Cross-script caveat stated plainly:** "token" means a sign in Indus and Linear A/B sign rows, and a word in the word rows. The sign rows are the like-for-like comparison; the word rows are reported because they are the level at which the Voynich claim lives.

## Reproduction status of the three headline trifecta figures

| Headline as approved | Re-derived here | Verdict |
|---|---|---|
| Indus "in the deciphered-language band on size-matched compression" | Δ −10.12 % vs deciphered control −11.57 % (gap 1.45 pp), identical metric | **reproduced in the weak form**; "band" wording needs a second deciphered control |
| Voynich "statistically inconsistent with random-scribble hoaxes (5 independent probes)" | 5 probes run, **4 reject** the scribble/anagram nulls, word-order compressibility (V5) does **not** (Δ −0.37 %, \|z\| = 2.98) | **partly reproduced — 4 of 5, not 5 of 5**; V5 must not be quoted as evidence |
| Linear A "functional translation ≈85–90 %" | not rebuildable as code — a self-assessed (interpretation) figure in `New folder/TRANSLATION_PASS.md` line 538, with no metric behind it; the one computed figure there (81.6 % A/I/U vocalism) needs a borrowed sign→value table that is not in the repo | **not reproduced — no script can be written for it** |

Also unreproduced and left marked as such: the old "LA LZ Δ −7.6 % vs LB −30 %" pair (measured now: −4.76 % / −11.57 % sign level, −3.64 % / −7.79 % word level — same direction, about half the separation; the destroyed engine's normalisation is not recoverable) and the 94.3 % sign-value overlap (same missing sign→value table).
