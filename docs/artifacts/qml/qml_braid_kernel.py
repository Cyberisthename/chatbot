#!/usr/bin/env python3
"""
qml_braid_kernel.py — QLM Layer: FBSC braid-semantic language kernel (frontier research).

agent-theoretical-physicist · 2026-09-20 (regenerated 2026-09-24)
pure Python + numpy/scipy · deterministic · no third-party quantum SDKs.

WHAT IT IS: a braid-attention kernel where (1) each sign/syllable token gets an FBSC 3-seed
(either arbitrary "hash", structure-derived from frequency+positional coordinates, or
distributional co-occurrence via SVD/LSA); (2) the token becomes an ordered braid word →
an exchange-with-phase braid unitary (the Burau representation here); (3) a word/formula is
the braid COMPOSITION of its tokens (ordered, non-commutative); (4) "semantic distance" =
1 − state-overlap between composed braid states. Boundary markers get dedicated boundary
seeds to segment streams into phrase braids.

REGENERATION HONESTY: the machinery below is reproducible. The measured ARI numbers from the
approved rounds (+0.108 struct seed, +0.152 co-occurrence seed, +0.190 Burau invariant) were
computed on the REAL Linear A top-60 formula dataset with register labels. That dataset did
NOT survive the 2026-09-24 disk-full event (not in the repo, not in /var/tmp backups). Those
numbers are therefore PRESERVED AS RECORDED in QLM_FRAMEWORK.md / QLM_ROUND2.md / the results
JSONs, and are NOT re-measurable here. This script runs a deterministic synthetic self-test to
prove the machinery runs and produces stable output, but it does not reproduce the Linear-A
ARI values — that would be fabrication to claim otherwise.

Run:  python3 qml_braid_kernel.py  ->  qml_self_test.json (synthetic, deterministic)
"""
import json
import hashlib
import numpy as np

RNG_SEED = 2026


def _stable_hash(obj):
    """Deterministic hash (hashlib), independent of PYTHONHASHSEED."""
    return int(hashlib.sha256(repr(obj).encode('utf-8')).hexdigest(), 16)
N_STRANDS = 4
WORD_LEN = 6
T_GRID = [-1.0, 2.0, 0.5 + 0.5j]

# Boundary markers (Linear A / undeciphered-script conventions, as recorded)
BOUNDARY_SEEDS = {
    "word_sep": "\U00010101",   # U+10101 word separator
    "damage": "\U0001076B",     # U+1076B damage marker
    "prefix": "\U00010607",     # U+10607 prefix sign
}


# --- FBSC 3-seed schemes ----------------------------------------------------
def seed_hash(token):
    """Arbitrary (null) seed: a fixed 3-vector derived from the token's bytes."""
    h = _stable_hash(token) % (2 ** 32)
    return np.array([(h & 0xFF) / 255.0, ((h >> 8) & 0xFF) / 255.0,
                     ((h >> 16) & 0xFF) / 255.0]) + 1e-3


def seed_struct(token, freq, position, vocab_size):
    """Structure-derived seed from frequency + positional coordinate (normalized)."""
    return np.array([freq, position, vocab_size / 100.0])


def cooccurrence_seeds(formulas, dim=3):
    """Distributional (co-occurrence) seeds via SVD/LSA of the within-word sign
    co-occurrence matrix. formulas: list of token lists (each formula is a sequence)."""
    vocab = sorted({tok for f in formulas for tok in f})
    idx = {t: i for i, t in enumerate(vocab)}
    n = len(vocab)
    C = np.zeros((n, n))
    for f in formulas:
        for a in f:
            for b in f:
                if a != b:
                    C[idx[a], idx[b]] += 1
    if n < 2:
        return {t: np.ones(dim) / dim for t in vocab}
    U, s, _ = np.linalg.svd(C, full_matrices=False)
    k = min(dim, len(s))
    emb = U[:, :k] * s[:k]
    if emb.shape[1] < dim:
        emb = np.pad(emb, ((0, 0), (0, dim - emb.shape[1])))
    # row-normalize to a comparable seed box
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    emb = emb / norms
    return {t: emb[idx[t]] for t in vocab}


# --- Braid machinery --------------------------------------------------------
def burau_sigma(n, i, t, inverse=False):
    M = np.eye(n, dtype=np.complex128)
    if inverse:
        block = np.array([[0.0, 1.0], [1.0 / t, 1.0 - 1.0 / t]], dtype=np.complex128)
    else:
        block = np.array([[1.0 - t, t], [1.0, 0.0]], dtype=np.complex128)
    M[i - 1:i + 1, i - 1:i + 1] = block
    return M


def burau_matrix(word, n, t):
    M = np.eye(n, dtype=np.complex128)
    for (i, sign) in word:
        M = M @ burau_sigma(n, i, t, inverse=(sign < 0))
    return M


def seed_to_word(seed, n=N_STRANDS, L=WORD_LEN, salt=0):
    """Deterministic ordered braid word from a 3-seed."""
    h = _stable_hash((tuple(np.round(seed, 6)), salt)) % (2 ** 32)
    rng = np.random.RandomState(h)
    return [(int(rng.randint(1, n - 1)), int(1 if rng.rand() < 0.5 else -1))
            for _ in range(L)]


def braid_state(word, n=N_STRANDS, t=2.0):
    """Composed braid state = Burau matrix applied to a reference vector |1,0,...,0>."""
    M = burau_matrix(word, n, t)
    ref = np.zeros(n, dtype=np.complex128)
    ref[0] = 1.0
    return M @ ref


def state_overlap_distance(a, b):
    """1 − |<a|b>| / (||a|| ||b||)."""
    num = abs(np.vdot(a, b))
    den = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    return float(1.0 - num / den)


def burau_trace_vector(word, n, t_values):
    return [np.trace(burau_matrix(word, n, t)) for t in t_values]


def burau_invariant_distance(v1, v2):
    return float(np.mean([abs(a - b) for a, b in zip(v1, v2)]))


def jaccard_distance(set_a, set_b):
    a, b = set(set_a), set(set_b)
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return float(1.0 - inter / union) if union else 1.0


def bigram_jaccard_distance(seq_a, seq_b):
    def bigrams(s):
        return set(zip(s, s[1:]))
    return jaccard_distance(bigrams(seq_a), bigrams(seq_b))


def formula_to_word(tokens, seeds):
    """Compose a formula's tokens into ONE ordered braid word (non-commutative order)."""
    word = []
    for i, tok in enumerate(tokens):
        word.extend(seed_to_word(seeds[tok], salt=i))
    return word


def clustering_ari(labels, distances):
    """Adjusted Rand Index between ground-truth labels and a distance-threshold clustering.
    Uses a simple 2-cluster threshold split; returns a real ARI (sklearn-free)."""
    from scipy.cluster.hierarchy import linkage, fcluster
    n = len(labels)
    if n < 2:
        return 0.0
    # distances is already a condensed (upper-triangle) vector
    Z = linkage(np.array(distances), method='average')
    k = len(set(labels)) if labels else 2
    pred = fcluster(Z, k, criterion='maxclust')
    return float(adjusted_rand_index(list(labels), list(pred)))


def adjusted_rand_index(a, b):
    """Pair-counting ARI (no sklearn)."""
    a, b = list(a), list(b)
    n = len(a)
    # contingency table over unique labels
    ua = sorted(set(a)); ub = sorted(set(b))
    ct = {}
    for x, y in zip(a, b):
        ct[(x, y)] = ct.get((x, y), 0) + 1
    sum_a = {x: sum(ct.get((x, y), 0) for y in ub) for x in ua}
    sum_b = {y: sum(ct.get((x, y), 0) for x in ua) for y in ub}
    idx = sum(v * (v - 1) / 2 for v in ct.values())
    sum_a2 = sum(v * (v - 1) / 2 for v in sum_a.values())
    sum_b2 = sum(v * (v - 1) / 2 for v in sum_b.values())
    total = n * (n - 1) / 2
    expected = sum_a2 * sum_b2 / total if total else 0.0
    max_i = (sum_a2 + sum_b2) / 2
    if max_i - expected == 0:
        return 0.0
    return (idx - expected) / (max_i - expected)


# --- Synthetic self-test (data-agnostic; the real Linear A dataset is lost) ---
def synthetic_dataset(rng):
    """Generate a small deterministic dataset of formula-like token sequences with two
    register groups that share suffix tokens (mimicking the Linear A accounting-vs-ritual
    structure, WITHOUT claiming to be real data)."""
    # two "families" sharing a suffix token
    fam_a = [["KU", "RO"], ["KI", "RO"], ["KU", "RO", "JA"], ["KI", "RO", "JA"]]
    fam_b = [["JA", "SA", "ME"], ["A", "TA", "ME"], ["JA", "SA", "ME", "RA"], ["SI", "RU", "ME"]]
    formulas = fam_a + fam_b
    labels = ["A"] * len(fam_a) + ["B"] * len(fam_b)
    return formulas, labels


def main():
    rng = np.random.RandomState(RNG_SEED)
    formulas, labels = synthetic_dataset(rng)
    vocab = sorted({t for f in formulas for t in f})
    freq = {t: sum(f.count(t) for f in formulas) / len(formulas) for t in vocab}

    # 1) hash seeds (null control)
    hash_seeds = {t: seed_hash(t) for t in vocab}
    # 2) structure seeds (frequency + positional)
    struct_seeds = {t: seed_struct(t, freq[t], vocab.index(t), len(vocab)) for t in vocab}
    # 3) co-occurrence seeds (SVD/LSA)
    cooc_seeds = cooccurrence_seeds(formulas, dim=3)

    def distances(seeds, mode):
        D = []
        for i in range(len(formulas)):
            wi = formula_to_word(formulas[i], seeds)
            for j in range(i + 1, len(formulas)):
                if mode == "state":
                    D.append(state_overlap_distance(
                        braid_state(wi), braid_state(formula_to_word(formulas[j], seeds))))
                else:  # burau
                    D.append(burau_invariant_distance(
                        burau_trace_vector(wi, N_STRANDS, T_GRID),
                        burau_trace_vector(formula_to_word(formulas[j], seeds), N_STRANDS, T_GRID)))
        return D

    out = {
        "meta": {
            "note": ("SYNTHETIC self-test. The real Linear A top-60 dataset was lost in the "
                     "2026-09-24 disk-full event; recorded ARI values (struct +0.108, cooc "
                     "+0.152, Burau +0.190) are preserved in QLM_FRAMEWORK.md/QLM_ROUND2.md "
                     "and are NOT re-measurable here."),
            "n_strands": N_STRANDS, "word_len": WORD_LEN, "rng_seed": RNG_SEED,
        },
        "synthetic_ari": {
            "hash_seed_state": round(clustering_ari(labels, distances(hash_seeds, "state")), 4),
            "struct_seed_state": round(clustering_ari(labels, distances(struct_seeds, "state")), 4),
            "cooc_seed_state": round(clustering_ari(labels, distances(cooc_seeds, "state")), 4),
            "burau_cooc": round(clustering_ari(labels, distances(cooc_seeds, "burau")), 4),
            "bigram_jaccard": round(clustering_ari(labels, [
                bigram_jaccard_distance(formulas[i], formulas[j])
                for i in range(len(formulas)) for j in range(i + 1, len(formulas))]), 4),
        },
    }
    with open('qml_self_test.json', 'w') as f:
        json.dump(out, f, indent=2)
    print('=== qml_braid_kernel — synthetic self-test (a) ===')
    print('machinery runs deterministically on synthetic data; real Linear-A ARI is NOT')
    print('re-measurable (dataset lost). Recorded values preserved in the report files.\n')
    print(json.dumps(out["synthetic_ari"], indent=2))
    print('\nwrote qml_self_test.json')


if __name__ == '__main__':
    main()
