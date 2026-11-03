#!/usr/bin/env python3
"""
voynich_structure_probe.py — structure-first probe of the Voynich manuscript
transcription against a random-scribble null.

HARD BOUNDARY (house rule, applies to every number printed here):
  structure != meaning. Nothing here reads Voynich words or assigns them sounds
  or meanings. The manuscript has no bilingual, so no decipherment is claimed
  and no phonetic reading is offered. The question asked is narrow and
  answerable: does the transcription behave like text with systematic internal
  order, or like random scribble?

Corpus (NOT committed — third-party transcription, fetched at run time):
  musyoku/voynich-transcription  voynich.txt  (Glen Claston's v101 transcription,
  word-separated, one line per manuscript line: `<folio.line>w.w.w-$`).
  Downloaded into --cache (default /var/tmp/trifecta_corpus); the sha256 of the
  file is printed and recorded so a run is identifiable. --voynich PATH points
  at a local copy instead. This is NOT the EVA (Zandbergen/Landini) alphabet, so
  letter-level numbers are transcription-specific and are reported as such.

Five probes, each against a seeded null, each reported with a z-score:
  V1  letter-level mutual information I(X1;X2) within words.
  V2  word repetition: distinct-word share of tokens.
  V3  letter-level I(X1;X2) against a within-word ANAGRAM null (same letters per
      word, order destroyed) - tests word-internal order specifically.
  V4  positional grammar: first-letter minus last-letter entropy, in bits.
  V5  size-matched LZ compressibility vs a global token shuffle (the same C and
      Delta% used by the Indus and Linear A probes, so the three scripts are
      comparable).
  V1, V2, V4 use the RANDOM-SCRIBBLE null: the same words at the same positions,
  each letter drawn i.i.d. from the corpus letter unigram distribution. V3 uses
  the anagram null, V5 a token-shuffle null. A probe "rejects its null" at
  |z|>=3.

Honest scope: this tests ONE explicit scribble model (i.i.d. letters with the
manuscript's own letter frequencies and word lengths). It does not test every
hoax construction, and rejecting this null is not evidence of a language.

Reproduce every number in docs/artifacts/trifecta/VOYNICH_STRUCTURE_PROBE.md:

    python3 scripts/voynich_structure_probe.py
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
URL_VOY = "https://raw.githubusercontent.com/musyoku/voynich-transcription/master/voynich.txt"
DEFAULT_CACHE = "/var/tmp/trifecta_corpus"
Z_THRESHOLD = 3.0


# ------------------------------------------------------------------ fetching
def fetch(url, path):
    if not os.path.exists(path) or os.path.getsize(path) < 1000:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print("# downloading %s -> %s" % (url, path), file=sys.stderr)
        with urllib.request.urlopen(url, timeout=120) as r, open(path, "wb") as fh:
            fh.write(r.read())
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def load_voynich(path):
    """One sequence of words per manuscript line."""
    seqs = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("<") and ">" in line:
                line = line.split(">", 1)[1]
            line = line.rstrip("-=").strip()
            words = tuple(w for w in line.split(".") if w)
            if words:
                seqs.append(words)
    return seqs


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


def entropy(counts):
    n = sum(counts.values())
    return -sum((c / n) * math.log(c / n) for c in counts.values())


def letter_mutual_information(seqs):
    """I(X1;X2) over within-word adjacent letter pairs, in bits."""
    h1, h2, h21 = {}, {}, {}
    for s in seqs:
        for w in s:
            for a, b in zip(w, w[1:]):
                h1[a] = h1.get(a, 0) + 1
                h2[(a, b)] = h2.get((a, b), 0) + 1
    for (a, _b), c in h2.items():
        h21[a] = h21.get(a, 0) + c
    H1, H2 = entropy(h1), entropy(h2)
    H21 = H2 - entropy(h21)
    return (H1 - H21) / math.log(2)


def distinct_word_share(seqs):
    toks = [w for s in seqs for w in s]
    return len(set(toks)) / len(toks)


def positional_entropy(seqs):
    first, last = {}, {}
    for s in seqs:
        for w in s:
            if not w:
                continue
            first[w[0]] = first.get(w[0], 0) + 1
            last[w[-1]] = last.get(w[-1], 0) + 1
    return entropy(first) / math.log(2), entropy(last) / math.log(2)


def anagram_null(seqs, rng):
    """Destroy letter order inside every word, keeping each word's letters."""
    out = []
    for s in seqs:
        words = []
        for w in s:
            chars = list(w)
            rng.shuffle(chars)
            words.append("".join(chars))
        out.append(tuple(words))
    return out


def scribble_null(seqs, rng):
    """Random-scribble model: same words in the same positions, each letter
    drawn i.i.d. from the corpus letter unigram distribution."""
    letters, weights = [], []
    for s in seqs:
        for w in s:
            for ch in w:
                letters.append(ch)
    counts = {}
    for ch in letters:
        counts[ch] = counts.get(ch, 0) + 1
    pop = list(counts)
    wts = [counts[c] for c in pop]
    return [tuple("".join(rng.choices(pop, weights=wts, k=len(w))) for w in s) for s in seqs]


def probe(name, real, null_vals, unit):
    mu, sd = st.mean(null_vals), st.pstdev(null_vals)
    z = (real - mu) / sd if sd > 0 else None
    return {
        "probe": name, "unit": unit,
        "real": float(real), "null_mean": float(mu), "null_sd": float(sd),
        "z_score": float(z) if z is not None else None,
        "rejects_scribble_null": bool(z is not None and abs(z) >= Z_THRESHOLD),
        "tag": "measured",
    }


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--voynich", default=None, help="local copy of voynich.txt")
    ap.add_argument("--perms", type=int, default=25, help="scribble-null draws")
    ap.add_argument("--budget", type=int, default=1000)
    ap.add_argument("--draws", type=int, default=25)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    path = args.voynich or os.path.join(args.cache, "voynich.txt")
    sha = fetch(URL_VOY, path) if not args.voynich else _sha(path)
    seqs = load_voynich(path)

    rng = random.Random(SEED)
    mi_r, dws_r, gap_r, anag_r = [], [], [], []
    for _ in range(args.perms):
        null = scribble_null(seqs, rng)
        mi_r.append(letter_mutual_information(null))
        dws_r.append(distinct_word_share(null))
        f, l = positional_entropy(null)
        gap_r.append(f - l)
        anag_r.append(letter_mutual_information(anagram_null(seqs, rng)))

    f_real, l_real = positional_entropy(seqs)
    probes = [
        probe("V1 letter mutual information I(X1;X2)", letter_mutual_information(seqs), mi_r, "bits"),
        probe("V2 distinct-word share of tokens", distinct_word_share(seqs), dws_r, "ratio"),
        probe("V3 letter MI vs within-word anagram null", letter_mutual_information(seqs),
              anag_r, "bits"),
        probe("V4 first-minus-last letter entropy gap", f_real - l_real, gap_r, "bits"),
    ]
    c = size_matched(seqs, args.budget, args.draws)
    c_z = c["delta_pct_vs_shuffle"] / max(1e-12, c["c_shuffle_sd"] / c["c_shuffle_mean"] * 100)
    probes.append({
        "probe": "V5 size-matched compressibility delta", "unit": "% vs shuffle",
        "real": c["delta_pct_vs_shuffle"], "null_mean": 0.0,
        "null_sd": float(c["c_shuffle_sd"] / c["c_shuffle_mean"] * 100),
        "z_score": float(c_z), "rejects_scribble_null": bool(abs(c_z) >= Z_THRESHOLD),
        "tag": "measured",
    })

    n_reject = sum(1 for p in probes if p["rejects_scribble_null"])
    out = {
        "meta": {
            "script": "voynich",
            "title": "Voynich structure-first probe vs a random-scribble null (re-derived, no tuning)",
            "source": URL_VOY, "sha256": sha,
            "transcription": "Glen Claston v101 (word-separated); NOT EVA, so letter-level "
                             "numbers are transcription-specific",
            "n_lines": len(seqs),
            "n_words": sum(len(s) for s in seqs),
            "n_distinct_words": len({w for s in seqs for w in s}),
            "n_distinct_letters": len({ch for s in seqs for w in s for ch in w}),
            "boundary": "structure != meaning; no decipherment and no phonetic reading claimed",
            "null_model": "i.i.d. letters drawn from the corpus letter unigram, same word lengths "
                          "and positions (V1-V4); global token shuffle (V5)",
            "z_threshold": Z_THRESHOLD, "perms": args.perms, "seed": SEED,
        },
        "probes": probes,
        "summary": {
            "probes_run": len(probes),
            "probes_rejecting_scribble_null": n_reject,
            "verdict": ("inconsistent with this random-scribble model on %d of %d probes"
                        % (n_reject, len(probes))),
            "tag": "measured",
        },
        "C_size_matched_compressibility": c,
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
