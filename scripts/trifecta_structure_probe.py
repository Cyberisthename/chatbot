#!/usr/bin/env python3
"""
trifecta_structure_probe.py — put the three structure-first probes on one scale.

Runs scripts/indus_structure_probe.py, scripts/lineara_structure_probe.py and
scripts/voynich_structure_probe.py as modules, and compares them with the one
metric they all compute identically:

  C        size-matched compressibility = LZ(1976) distinct phrases per token at
           a fixed token budget, mean over seeded subsamples.
  Delta%   = 100 * (C_real / C_shuffle - 1): how much redundancy the real order
           carries compared with a global shuffle of the same tokens at the same
           budget. Negative = more compressible than chance.

The scale-free comparison is Delta%, because it cancels the alphabet size and the
token-budget choice. Linear B (deciphered Greek) is the deciphered-language
control. With a single deciphered control there is no "band" in the strict sense,
only a point, so this script reports the gap to that point rather than claiming
a band.

HARD BOUNDARY: structure != meaning. Nothing here reads or translates any
script; no decipherment is claimed for Linear A or the Indus script.

Reproduce every number in docs/artifacts/trifecta/TRIFECTA_STRUCTURE_PROBE.md:

    python3 scripts/trifecta_structure_probe.py

Both non-committed corpora are fetched on first run into --cache
(default /var/tmp/trifecta_corpus); the Indus corpus is committed in-repo.
"""
import argparse
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BUDGET = 1000
DRAWS = 25


def load_module(name):
    path = os.path.join(HERE, name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--cache", default="/var/tmp/trifecta_corpus")
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    indus = load_module("indus_structure_probe")
    lineara = load_module("lineara_structure_probe")
    voynich = load_module("voynich_structure_probe")

    rows = []

    def add(script, level, role, seqs):
        c = indus.size_matched(seqs, args.budget, args.draws)  # identical code in all three
        rows.append({
            "script": script, "level": level, "role": role,
            "n_tokens": sum(len(s) for s in seqs),
            "n_distinct_tokens": len({t for s in seqs for t in s}),
            "c_real": c["c_real_mean"], "c_real_sd": c["c_real_sd"],
            "c_shuffle": c["c_shuffle_mean"],
            "delta_pct": c["delta_pct_vs_shuffle"],
        })

    add("indus", "sign", "undeciphered", indus.load_corpus(indus.DEFAULT_CORPUS))
    la_records = lineara.load_la(os.path.join(args.cache, "lineara_inscriptions.json"))
    lb_records = lineara.load_lb(os.path.join(args.cache, "linearb_inscriptions.js"))
    add("linear_a", "sign", "undeciphered", lineara.sign_seqs(la_records))
    add("linear_b", "sign", "deciphered control", lineara.sign_seqs(lb_records))
    add("linear_a", "word", "undeciphered", lineara.word_seqs(la_records))
    add("linear_b", "word", "deciphered control", lineara.word_seqs(lb_records))
    voy = voynich.load_voynich(os.path.join(args.cache, "voynich.txt"))
    add("voynich", "word", "undeciphered", voy)
    add("voynich", "letter", "undeciphered",
        [tuple(ch for w in s for ch in w) for s in voy])

    by = {(r["script"], r["level"]): r for r in rows}
    band = by[("linear_b", "sign")]
    ind = by[("indus", "sign")]
    out = {
        "meta": {
            "title": "Three scripts, one method: structure-first probe comparison",
            "budget_tokens": args.budget, "draws": args.draws,
            "metric": "C = LZ phrases / token at matched budget; Delta% = 100*(C_real/C_shuffle-1)",
            "deciphered_control": "Linear B (sign level)",
            "boundary": "structure != meaning; no decipherment claimed for Linear A or Indus",
        },
        "rows": rows,
        "indus_vs_deciphered_control": {
            "indus_delta_pct": ind["delta_pct"],
            "linear_b_delta_pct": band["delta_pct"],
            "gap_pp": ind["delta_pct"] - band["delta_pct"],
            "indus_c_real": ind["c_real"],
            "linear_b_c_real": band["c_real"],
            "note": "one deciphered control = one point, not a band; the gap is reported, "
                    "not a membership claim",
            "tag": "measured",
        },
    }
    text = json.dumps(out, indent=2)
    print(text)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
