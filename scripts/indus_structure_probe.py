#!/usr/bin/env python3
"""
indus_structure_probe.py — structure-first probe of the Indus sign corpus.

HARD BOUNDARY (house rule, applies to every number printed here):
  structure != meaning. Nothing in this file reads, translates, or assigns
  phonetic values to a sign. No decipherment is claimed without a bilingual or
  an identified language, and none exists for the Indus script.

What it measures (all on the COMMITTED corpus, no network, no tuning):
  H1  P385 opens inscriptions far beyond chance (count vs per-sequence shuffle).
  H2  P385->P122 is an enriched collocation (vs marginal-independence and shuffle).
  H3  Short-range context conditioning: H(X2|X1) << H(X1) and H(X3|X1X2) lower.
  H4  No long-range formula chains: LZ compressibility gap vs whole-corpus token
      shuffle is accounted for by within-sequence order; far-repeated bigrams = 0.
  C   Size-matched compressibility c = LZ phrases / N tokens at a fixed token
      budget, with a global-shuffle null. This is the cross-script comparable
      number: the same metric is computed for Linear A / Linear B / Voynich by
      scripts/lineara_structure_probe.py and scripts/voynich_structure_probe.py,
      and compared by scripts/trifecta_structure_probe.py.

Corpus: docs/artifacts/indus/indus_corpus_clean.tsv (committed, 179 Mohenjo-daro
sides, 1,003 P-code tokens). Provenance of the corpus is documented in
docs/artifacts/indus/NORMALIZE_RB.py and INDUS_H1H4_CROSSVALIDATE.md; it is a
subset of the Parpola/CISI digitization, NOT the full Mahadevan concordance.

Reproduce every number in docs/artifacts/trifecta/INDUS_STRUCTURE_PROBE.md:

    python3 scripts/indus_structure_probe.py

Options: --corpus PATH (any TSV with a header and a token column),
         --budget N (size-matched token budget, default 1000),
         --draws R (subsample draws, default 25),
         --json PATH (machine-readable output).
"""
import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics as st
import sys

SEED = 2026
DEFAULT_CORPUS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "docs", "artifacts", "indus", "indus_corpus_clean.tsv")


# ------------------------------------------------------------------ loading
def load_corpus(path):
    """Read a TSV: header row, one row per sequence, tokens in the last column."""
    seqs = []
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh, delimiter="\t"))
    header, body = rows[0], rows[1:]
    idx = header.index("sequence_reading_order") if "sequence_reading_order" in header else len(header) - 1
    for row in body:
        if len(row) <= idx:
            continue
        toks = row[idx].split()
        if toks:
            seqs.append(tuple(toks))
    return seqs


# ------------------------------------------------------------------ baselines
def token_shuffle(seqs, rng):
    """Destroy order completely: shuffle all tokens across the whole corpus."""
    flat = [t for s in seqs for t in s]
    rng.shuffle(flat)
    out, i = [], 0
    for s in seqs:
        out.append(tuple(flat[i:i + len(s)]))
        i += len(s)
    return out


def per_seq_shuffle(seqs, rng):
    """Shuffle inside each sequence: keeps sequence lengths, loses order."""
    out = []
    for s in seqs:
        t = list(s)
        rng.shuffle(t)
        out.append(tuple(t))
    return out


# ------------------------------------------------------------------ LZ / C
def lz_phrases(seq):
    """Lempel-Ziv (1976) distinct-phrase count of a token sequence."""
    i, n, count, seen = 0, len(seq), 0, set()
    while i < n:
        j = i
        while j < n and seq[i:j + 1] in seen:
            j += 1
        seen.add(seq[i:j + 1])
        count += 1
        i = max(j, i + 1)
    return count


def compressibility(seqs):
    """LZ phrases per token, with sequence-boundary sentinels so no phrase
    can span two sequences. Lower = more redundant = more compressible."""
    sentinel = "\x00_SEP\x00"
    flat = [sentinel]
    for s in seqs:
        flat.extend(s)
        flat.append(sentinel)
    n_tok = sum(len(s) for s in seqs)
    return lz_phrases(tuple(flat)) / max(1, n_tok)


def size_matched(seqs, budget, draws, seed=SEED):
    """Mean +- sd of compressibility at a fixed token budget, over `draws`
    seeded subsamples of whole sequences (last sequence truncated to hit the
    budget exactly), each with its own global-shuffle null."""
    rng = random.Random(seed)
    pool = [s for s in seqs if s]
    real, null = [], []
    for _ in range(draws):
        order = pool[:]
        rng.shuffle(order)
        take, n = [], 0
        for s in order:
            if n >= budget:
                break
            room = budget - n
            take.append(s[:room])
            n += len(take[-1])
        if n < budget:          # corpus smaller than the budget: report as is
            budget_eff = n
        else:
            budget_eff = budget
        real.append(lz_phrases(tuple(t for s in take for t in s)) / budget_eff)
        flat = [t for s in take for t in s]
        rng.shuffle(flat)
        null.append(lz_phrases(tuple(flat)) / budget_eff)
    return {
        "budget_tokens": int(budget),
        "draws": draws,
        "c_real_mean": float(st.mean(real)),
        "c_real_sd": float(st.pstdev(real)),
        "c_shuffle_mean": float(st.mean(null)),
        "c_shuffle_sd": float(st.pstdev(null)),
        # Delta% = 100*(C_real/C_shuffle - 1): negative = more redundant than chance.
        # Same definition and sign convention as the Linear A and Voynich probes.
        "delta_pct_vs_shuffle": float(100 * (st.mean(real) / st.mean(null) - 1)),
        "tag": "measured",
    }


# ------------------------------------------------------------------ entropies
def entropy_chain(seqs):
    toks = [t for s in seqs for t in s]
    big, tri = [], []
    for s in seqs:
        big += list(zip(s, s[1:]))
        tri += list(zip(s, s[1:], s[2:]))
    c1 = {}
    for t in toks:
        c1[t] = c1.get(t, 0) + 1
    n1 = len(toks)
    H1 = -sum((k / n1) * math.log(k / n1) for k in c1.values())
    c2 = {}
    for p in big:
        c2[p] = c2.get(p, 0) + 1
    n2 = len(big)
    H2 = -sum((k / n2) * math.log(k / n2) for k in c2.values())
    c1b = {}
    for a, _ in big:
        c1b[a] = c1b.get(a, 0) + 1
    H21 = H2 - sum((-c1b[a] / n2) * math.log(c1b[a] / n2) for a in c1b)
    c3 = {}
    for p in tri:
        c3[p] = c3.get(p, 0) + 1
    n3 = len(tri)
    H3 = -sum((k / n3) * math.log(k / n3) for k in c3.values())
    c2b = {}
    for a, b, _ in tri:
        c2b[(a, b)] = c2b.get((a, b), 0) + 1
    H32 = H3 - sum((-c2b[p] / n3) * math.log(c2b[p] / n3) for p in c2b)
    return H1, H2, H3, H21, H32


# ------------------------------------------------------------------ H1..H4
def h1_initial(seqs, n_perms=400, seed=SEED):
    n_initial = sum(1 for s in seqs if s and s[0] == "P385")
    rng = random.Random(seed)
    draws = [sum(1 for s in per_seq_shuffle(seqs, rng) if s and s[0] == "P385")
             for _ in range(n_perms)]
    mu, sd = st.mean(draws), st.pstdev(draws)
    return {
        "measured_initial": n_initial, "n_seqs": len(seqs),
        "base_rate": n_initial / len(seqs),
        "shuffle_mean": float(mu), "shuffle_sd": float(sd),
        "z_score": float((n_initial - mu) / sd) if sd > 0 else None,
        "perms": n_perms, "tag": "measured",
    }


def h2_collocation(seqs, n_perms=400, seed=SEED):
    def count(ss):
        return sum(1 for s in ss for a, b in zip(s, s[1:]) if a == "P385" and b == "P122")

    real = count(seqs)
    rng = random.Random(seed)
    draws = [count(token_shuffle(seqs, rng)) for _ in range(n_perms)]
    mu, sd = st.mean(draws), st.pstdev(draws)
    flat = [t for s in seqs for t in s]
    nb = sum(max(0, len(s) - 1) for s in seqs)
    exp = (flat.count("P385") / len(flat)) * (flat.count("P122") / len(flat)) * nb
    return {
        "measured": real, "expected_independent": float(exp),
        "enrichment": float(real / exp) if exp > 0 else None,
        "shuffle_mean": float(mu), "shuffle_sd": float(sd),
        "z_score": float((real - mu) / sd) if sd > 0 else None,
        "perms": n_perms, "tag": "measured",
    }


def h3_context(seqs):
    H1, H2, H3, H21, H32 = entropy_chain(seqs)
    lb = math.log(2)
    return {
        "H1_bits": H1 / lb, "H2_bits": H2 / lb, "H3_bits": H3 / lb,
        "H2_given_H1_bits": H21 / lb, "H3_given_H21_bits": H32 / lb,
        # Successive joint-entropy DIFFERENCE convention used by the pre-incident
        # engine (ivs_quant_engine.py): this is the convention the approved
        # figures 2.413 / 0.414 were quoted in.
        "H2_minus_H1_bits_difference_convention": (H2 - H1) / lb,
        "H3_minus_H2_bits_difference_convention": (H3 - H2) / lb,
        "reduction_H21_vs_H1_pct": 100 * (1 - H21 / H1) if H1 else None,
        "n_bigrams": sum(max(0, len(s) - 1) for s in seqs),
        "bigrams_used": len({(a, b) for s in seqs for a, b in zip(s, s[1:])}),
        "tag": "measured",
    }


def h4_longrange(seqs, n_perms=200, seed=SEED):
    def far_count(ss):
        cnt = 0
        for s in ss:
            first = {}
            for i in range(len(s) - 1):
                bg = (s[i], s[i + 1])
                if bg in first:
                    if i - first[bg] >= 3:
                        cnt += 1
                else:
                    first[bg] = i
        return cnt

    lz_real = compressibility(seqs)
    rng = random.Random(seed)
    lz_tok, lz_seq, far = [], [], []
    for _ in range(n_perms):
        lz_tok.append(compressibility(token_shuffle(seqs, rng)))
        sh = per_seq_shuffle(seqs, rng)
        lz_seq.append(compressibility(sh))
        far.append(far_count(sh))
    return {
        "lz_real": float(lz_real),
        "lz_token_shuffle_mean": float(st.mean(lz_tok)),
        "lz_per_seq_shuffle_mean": float(st.mean(lz_seq)),
        "lz_gap_vs_token_shuffle_pct": float(100 * (1 - lz_real / st.mean(lz_tok))),
        "lz_gap_vs_per_seq_shuffle_pct": float(100 * (1 - lz_real / st.mean(lz_seq))),
        "far_repeated_bigrams_real": int(far_count(seqs)),
        "far_repeated_bigrams_shuffle_mean": float(st.mean(far)),
        "perms": n_perms, "tag": "measured",
    }


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--budget", type=int, default=1000)
    ap.add_argument("--draws", type=int, default=25)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    seqs = load_corpus(args.corpus)
    fp = hashlib.sha256("\n".join(" ".join(s) for s in seqs).encode()).hexdigest()[:16]
    out = {
        "meta": {
            "script": "indus",
            "title": "Indus structure-first probe (re-derived, no tuning)",
            "corpus": os.path.relpath(os.path.abspath(args.corpus)),
            "corpus_sha16": fp,
            "n_sequences": len(seqs),
            "n_tokens": sum(len(s) for s in seqs),
            "n_distinct_tokens": len({t for s in seqs for t in s}),
            "boundary": "structure != meaning; no decipherment claimed",
            "seed": SEED,
        },
        "H1_P385_initial": h1_initial(seqs),
        "H2_P385_P122_collocation": h2_collocation(seqs),
        "H3_short_range_context": h3_context(seqs),
        "H4_long_range": h4_longrange(seqs),
        "C_size_matched_compressibility": size_matched(seqs, args.budget, args.draws),
    }
    text = json.dumps(out, indent=2)
    print(text)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
