#!/usr/bin/env python3
"""
NORMALIZE_RB.py — rebuild clean Indus corpus TSV from surviving raw JSON.

Rebuilt 2026-09-28 by agent-compression-specialist after the 2026-09-24 disk-full
incident destroyed the original NORMALIZE.py / indus_corpus_clean.tsv.
Sourced from the surviving raw corpus clone at indus/corpus_raw/
(github.com/mayig/indus-valley-script-corpus, MIT license, Parpola CISI digitization).

Convention (per corpus README + DB task records b6635fcb/4d123916/7b903a12):
  - One row per side: id, description, P-code sequence.
  - Raw JSON stores graphemes left-to-right physically; the script is read
    right-to-left, so READING ORDER = REVERSE of the stored list.
  - P-codes only (GROUND_RULES: no decipherment/meaning assigned).
Output: indus/indus_corpus_clean.tsv  (tab-separated, deterministic order)
"""
import csv, glob, json, os

RAW_CORPUS = os.path.join(os.path.dirname(__file__), "corpus_raw", "corpus")
OUT_TSV = os.path.join(os.path.dirname(__file__), "indus_corpus_clean.tsv")


def load_sequences(raw_corpus=RAW_CORPUS, reading_order=True):
    """Yield (artefact_id, description, [P-codes in reading order]) for each side."""
    rows = []
    for path in sorted(glob.glob(os.path.join(raw_corpus, "*", "*.json"))):
        with open(path, encoding="utf-8") as fh:
            artefact = json.load(fh)
        for side in artefact:
            pid = side.get("id", os.path.basename(path).replace(".json", ""))
            desc = side.get("description", "")
            stored = [g["id"] for g in side.get("graphemes", [])]
            seq = list(reversed(stored)) if reading_order else list(stored)
            rows.append((pid, desc, seq))
    rows.sort(key=lambda r: (r[0], r[1]))  # deterministic
    return rows


def main():
    rows = load_sequences()
    n_sides = len(rows)
    n_tokens = sum(len(r[2]) for r in rows)
    n_uniq = len({t for r in rows for t in r[2]})
    with open(OUT_TSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["id", "description", "sequence_reading_order"])
        for pid, desc, seq in rows:
            w.writerow([pid, desc, " ".join(seq)])
    # Print summary to stderr-free stdout for the callers
    print(json.dumps({
        "sides": n_sides, "tokens": n_tokens, "distinct_signs": n_uniq,
        "reading_order": "reversed-from-stored (right-to-left)",
        "out": OUT_TSV,
    }))


if __name__ == "__main__":
    main()