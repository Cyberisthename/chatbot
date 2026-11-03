#!/usr/bin/env python3
"""
lineara_structure_probe.py — structure-first probe of the Linear A corpus,
with Linear B (deciphered Greek) as the deciphered-language control.

HARD BOUNDARY (house rule, applies to every number printed here):
  structure != meaning. This script never assigns a value or a sound to a sign
  and never translates anything. Linear A has no bilingual and no identified
  language, so no decipherment is claimed. What is measured is the shape of the
  corpus: token statistics, positional/context conditioning, and how much
  redundancy the token stream carries compared with a shuffled null.

Corpora (NOT committed to this repo — third-party data, fetched at run time):
  Linear A  mwenge/lineara.xyz  items_analysis/inscriptions.json
            Damos-style transliterations; the repo carries a public-domain-style
            blessing ("share freely"), the underlying inscriptions are the
            GORILA/Damos epigraphic record.
  Linear B  mwenge/linearb.xyz  LinearBInscriptions.js (same schema, JS Map)
  Both are downloaded into --cache (default /var/tmp/trifecta_corpus) and the
  sha256 of each file is printed and recorded so a run is identifiable.
  If the network is unavailable, point --lineara / --linearb at local copies of
  those two files and nothing else changes.

Metrics (identical code to scripts/indus_structure_probe.py and
scripts/voynich_structure_probe.py, so the numbers are comparable):
  tokens      word level ("QE-RA2-U") and sign level (word split on '-').
  H1, H2|H1   unigram entropy and first-order conditional entropy, bits/token.
  C           size-matched compressibility: LZ(1976) distinct phrases per token
              at a fixed token budget, over seeded subsamples; with a
              global-shuffle null.  Delta% = 100*(C_real/C_shuffle - 1):
              more negative = the real order carries more redundancy than
              chance. The same C is what "size-matched compression" means for
              the Indus and Voynich probes.

Reproduce every number in docs/artifacts/trifecta/LINEARA_STRUCTURE_PROBE.md:

    python3 scripts/lineara_structure_probe.py
"""
import argparse
import hashlib
import json
import math
import os
import random
import statistics as st
import sys
import urllib.request

SEED = 2026
URL_LA = "https://raw.githubusercontent.com/mwenge/lineara.xyz/master/items_analysis/inscriptions.json"
URL_LB = "https://raw.githubusercontent.com/mwenge/linearb.xyz/master/LinearBInscriptions.js"
DEFAULT_CACHE = "/var/tmp/trifecta_corpus"

# A word token: transliteration letters, optional subscript digits, optional '-'
WORD_OK = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
              "0123456789₀₁₂₃₄₅₆₇₈₉*-+")
SEPARATORS = {"𐄁", "𐄂", "\n", "[", "]", "(", ")", "…", "vacat", ""}


# ------------------------------------------------------------------ fetching
def fetch(url, path):
    if not os.path.exists(path) or os.path.getsize(path) < 1000:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print("# downloading %s -> %s" % (url, path), file=sys.stderr)
        with urllib.request.urlopen(url, timeout=120) as r, open(path, "wb") as fh:
            fh.write(r.read())
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def load_la(path):
    """Linear A: JSON array of [id, {transliteratedWords: [...]}, ...]."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return [(pid, rec.get("transliteratedWords") or []) for pid, rec in data]


def load_lb(path):
    """Linear B: JS file declaring `var inscriptions = new Map([[id, {...}], ...]);`
    The literal between `new Map(` and the final `)` is valid JSON."""
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    i = src.index("new Map(") + len("new Map(")
    j = src.rindex(")")
    data = json.loads(src[i:j])
    return [(pid, rec.get("transliteratedWords") or []) for pid, rec in data]


# ------------------------------------------------------------------ tokens
def is_word(tok):
    return bool(tok) and tok not in SEPARATORS and all(c in WORD_OK for c in tok) \
        and any(c.isalpha() for c in tok)


def word_seqs(records):
    out = []
    for _, words in records:
        seq = tuple(w for w in words if is_word(w))
        if seq:
            out.append(seq)
    return out


def sign_seqs(records):
    """Sign level: split each word on '-' (Linear A/B syllabic words are
    hyphenated sign by sign), keep the syllabic pieces."""
    out = []
    for _, words in records:
        seq = []
        for w in words:
            if not is_word(w):
                continue
            seq.extend(p for p in w.split("-") if p and any(c.isalpha() for c in p))
        if seq:
            out.append(tuple(seq))
    return out


# ------------------------------------------------------------------ metrics
def lz_phrases(seq):
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
    sentinel = "\x00_SEP\x00"
    flat = [sentinel]
    for s in seqs:
        flat.extend(s)
        flat.append(sentinel)
    return lz_phrases(tuple(flat)) / max(1, sum(len(s) for s in seqs))


def size_matched(seqs, budget, draws, seed=SEED):
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
            take.append(s[:budget - n])
            n += len(take[-1])
        budget_eff = n if n < budget else budget
        real.append(lz_phrases(tuple(t for s in take for t in s)) / budget_eff)
        flat = [t for s in take for t in s]
        rng.shuffle(flat)
        null.append(lz_phrases(tuple(flat)) / budget_eff)
    c_real, c_null = st.mean(real), st.mean(null)
    return {
        "budget_tokens": int(budget), "draws": draws,
        "c_real_mean": float(c_real), "c_real_sd": float(st.pstdev(real)),
        "c_shuffle_mean": float(c_null), "c_shuffle_sd": float(st.pstdev(null)),
        "delta_pct_vs_shuffle": float(100 * (c_real / c_null - 1)),
        "tag": "measured",
    }


def entropies(seqs):
    toks = [t for s in seqs for t in s]
    c1, c2, prev = {}, {}, {}
    for t in toks:
        c1[t] = c1.get(t, 0) + 1
    for s in seqs:
        for a, b in zip(s, s[1:]):
            c2[(a, b)] = c2.get((a, b), 0) + 1
            prev[a] = prev.get(a, 0) + 1
    n1 = len(toks)
    n2 = sum(c2.values())
    H1 = -sum((k / n1) * math.log(k / n1) for k in c1.values())
    H2 = -sum((k / n2) * math.log(k / n2) for k in c2.values())
    H21 = H2 - sum((-prev[a] / n2) * math.log(prev[a] / n2) for a in prev)
    lb = math.log(2)
    return {
        "n_tokens": n1,
        "H1_bits": H1 / lb,
        "H2_given_H1_bits": H21 / lb,
        "reduction_H21_vs_H1_pct": 100 * (1 - H21 / H1) if H1 else None,
        "type_token_ratio": len(c1) / n1,
        "tag": "measured",
    }


def describe(name, seqs, budget, draws):
    e = entropies(seqs)
    c = size_matched(seqs, budget, draws)
    e["n_sequences"] = len(seqs)
    e["n_distinct_tokens"] = len({t for s in seqs for t in s})
    e["mean_seq_len"] = st.mean(len(s) for s in seqs)
    return {"name": name, "tokens": e, "compressibility": c}


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--lineara", default=None, help="local copy of inscriptions.json")
    ap.add_argument("--linearb", default=None, help="local copy of LinearBInscriptions.js")
    ap.add_argument("--budget", type=int, default=1000)
    ap.add_argument("--draws", type=int, default=25)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    la_path = args.lineara or os.path.join(args.cache, "lineara_inscriptions.json")
    lb_path = args.linearb or os.path.join(args.cache, "linearb_inscriptions.js")
    la_sha = fetch(URL_LA, la_path) if not args.lineara else _sha(la_path)
    lb_sha = fetch(URL_LB, lb_path) if not args.linearb else _sha(lb_path)

    la = load_la(la_path)
    lb = load_lb(lb_path)

    out = {
        "meta": {
            "script": "lineara",
            "title": "Linear A structure-first probe (re-derived, no tuning), Linear B as deciphered control",
            "linear_a_source": URL_LA, "linear_a_sha256": la_sha,
            "linear_b_source": URL_LB, "linear_b_sha256": lb_sha,
            "boundary": "structure != meaning; no decipherment claimed (no bilingual exists for Linear A)",
            "seed": SEED,
            "linear_a_records": len(la), "linear_b_records": len(lb),
        },
        "linear_a_word_level": describe("linear_a_words", word_seqs(la), args.budget, args.draws),
        "linear_a_sign_level": describe("linear_a_signs", sign_seqs(la), args.budget, args.draws),
        "linear_b_word_level": describe("linear_b_words", word_seqs(lb), args.budget, args.draws),
        "linear_b_sign_level": describe("linear_b_signs", sign_seqs(lb), args.budget, args.draws),
    }
    la_c = out["linear_a_sign_level"]["compressibility"]["delta_pct_vs_shuffle"]
    lb_c = out["linear_b_sign_level"]["compressibility"]["delta_pct_vs_shuffle"]
    out["contrast"] = {
        "delta_sign_level_linear_a_pct": la_c,
        "delta_sign_level_linear_b_pct": lb_c,
        "linear_a_less_redundant_than_linear_b_by_pp": float(lb_c - la_c),
        "interpretation": "deciphered Linear B carries more order-driven redundancy than "
                          "undeciphered Linear A under the same metric at the same token budget",
        "tag": "measured",
    }
    text = json.dumps(out, indent=2)
    print(text)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    return 0


def _sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


if __name__ == "__main__":
    sys.exit(main())
