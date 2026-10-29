#!/usr/bin/env python3
"""Pre-registered structure-only FlyWire challenge-distribution graph probe.

External data are intentionally not bundled. See docs/external_data/flywire/.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
import zlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

URL = "https://storage.googleapis.com/flywire-data/challenges/mfas/connectome_graph.csv.gz"
EXPECTED_HEADER = ["Source Node ID", "Target Node ID", "Edge Weight"]
SEEDS = [104729, 130363, 155921, 196613, 225307]
MASK64 = (1 << 64) - 1


def splitmix64(x: int) -> int:
    """The standard SplitMix64 unsigned-integer mixer."""
    z = (x + 0x9E3779B97F4A7C15) & MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return (z ^ (z >> 31)) & MASK64


def selected(node_id: int) -> bool:
    return splitmix64(node_id & MASK64) % 10 == 0


def parse_slice(path: Path):
    """Stream CSV and construct the registered induced binary graph slice."""
    selected_nodes: set[int] = set()
    pairs: list[tuple[int, int]] = []
    selection_cache: dict[int, bool] = {}
    raw_rows = 0
    raw_weight_sum = 0
    self_loops = 0

    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header != EXPECTED_HEADER:
            raise ValueError(f"unexpected header: {header!r}; expected {EXPECTED_HEADER!r}")
        for row_no, row in enumerate(reader, start=2):
            if len(row) != 3:
                raise ValueError(f"row {row_no}: expected 3 columns, got {len(row)}")
            # Python int preserves the source's signed-64-bit identifiers exactly.
            source, target, weight = int(row[0]), int(row[1]), int(row[2])
            if not (-(1 << 63) <= source < (1 << 63) and -(1 << 63) <= target < (1 << 63)):
                raise ValueError(f"row {row_no}: node identifier outside int64")
            raw_rows += 1
            raw_weight_sum += weight
            s_keep = selection_cache.setdefault(source, selected(source))
            t_keep = selection_cache.setdefault(target, selected(target))
            if s_keep:
                selected_nodes.add(source)
            if t_keep:
                selected_nodes.add(target)
            if s_keep and t_keep:
                if source == target:
                    self_loops += 1
                else:
                    pairs.append((source, target))

    unique_pairs = set(pairs)
    node_ids = np.array(sorted(selected_nodes), dtype=np.int64)
    index = {int(node): i for i, node in enumerate(node_ids)}
    # Sorting makes the initial edge arrangement deterministic; controls have their own seeds.
    pair_indices = sorted((index[s], index[t]) for s, t in unique_pairs)
    u = np.array([p[0] for p in pair_indices], dtype=np.int32)
    v = np.array([p[1] for p in pair_indices], dtype=np.int32)
    return node_ids, u, v, {
        "raw_rows": raw_rows,
        "raw_weight_sum": raw_weight_sum,
        "selected_node_count": int(len(node_ids)),
        "retained_nonloop_rows_before_deduplication": len(pairs),
        "retained_self_loop_rows_excluded": self_loops,
        "duplicate_ordered_pairs_collapsed": len(pairs) - len(unique_pairs),
    }


def degree_diagnostics(u: np.ndarray, v: np.ndarray, n: int) -> dict:
    out_degree = np.bincount(u, minlength=n).astype(np.int64)
    in_degree = np.bincount(v, minlength=n).astype(np.int64)

    def describe(values: np.ndarray) -> dict:
        _, counts = np.unique(values, return_counts=True)
        p = counts / counts.sum()
        return {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "max": int(values.max(initial=0)),
            "shannon_entropy_bits": float(-(p * np.log2(p)).sum()),
        }

    return {"in_degree": describe(in_degree), "out_degree": describe(out_degree)}


def motif_and_reciprocity(u: np.ndarray, v: np.ndarray, n: int) -> tuple[int, int, float]:
    """Return (directed 3-cycles, feed-forward motifs, reciprocity fraction)."""
    out_neighbors = [set() for _ in range(n)]
    in_neighbors = [set() for _ in range(n)]
    for a, b in zip(u.tolist(), v.tolist()):
        out_neighbors[a].add(b)
        in_neighbors[b].add(a)

    reciprocal = sum(1 for a, b in zip(u.tolist(), v.tolist()) if a in out_neighbors[b])

    # For arc a→b, close b→c→a. Each directed 3-cycle contributes three closures.
    cycle_closures = 0
    # For arc y→z, choose a common predecessor x: x→y, x→z, y→z.
    feed_forward = 0
    for a, b in zip(u.tolist(), v.tolist()):
        left, right = out_neighbors[b], in_neighbors[a]
        if len(left) > len(right):
            left, right = right, left
        cycle_closures += sum(1 for c in left if c in right)

        left, right = in_neighbors[a], in_neighbors[b]
        if len(left) > len(right):
            left, right = right, left
        feed_forward += sum(1 for x in left if x in right)

    cycles = cycle_closures // 3
    if cycle_closures % 3:
        raise AssertionError("directed 3-cycle closure count was not divisible by three")
    return cycles, feed_forward, reciprocal / len(u)


def lz_ratio(u: np.ndarray, v: np.ndarray, node_ids: np.ndarray) -> tuple[float, int, int]:
    """Registered degree-first canonical sequence proxy, compressed with DEFLATE/LZ77."""
    n = len(node_ids)
    out_degree = np.bincount(u, minlength=n)
    in_degree = np.bincount(v, minlength=n)
    order = sorted(range(n), key=lambda x: (int(out_degree[x]), int(in_degree[x]), int(node_ids[x])))
    rank = np.empty(n, dtype=np.uint32)
    rank[np.asarray(order, dtype=np.int64)] = np.arange(n, dtype=np.uint32)
    targets: list[int] = []
    by_source: list[list[int]] = [[] for _ in range(n)]
    for a, b in zip(u.tolist(), v.tolist()):
        by_source[a].append(int(rank[b]))
    for source in order:
        targets.extend(sorted(by_source[source]))
    raw = np.asarray(targets, dtype="<u4").tobytes()
    compressed = zlib.compress(raw, level=9)
    return len(compressed) / len(raw), len(raw), len(compressed)


def compute_metrics(u: np.ndarray, v: np.ndarray, node_ids: np.ndarray) -> dict:
    n, m = len(node_ids), len(u)
    if m == 0:
        raise ValueError("registered induced slice has no arcs")
    cycles, feed_forward, reciprocity = motif_and_reciprocity(u, v, n)
    ratio, raw_bytes, compressed_bytes = lz_ratio(u, v, node_ids)
    return {
        "reciprocity_fraction": reciprocity,
        "directed_3_cycle_count": cycles,
        "directed_3_cycle_rate_per_arc": cycles / m,
        "feed_forward_motif_count": feed_forward,
        "feed_forward_motif_rate_per_arc": feed_forward / m,
        "canonical_edge_sequence_lz_ratio": ratio,
        "canonical_edge_sequence_raw_bytes": raw_bytes,
        "canonical_edge_sequence_compressed_bytes": compressed_bytes,
        "degree_diagnostics": degree_diagnostics(u, v, n),
    }


def degree_preserving_switches(u0: np.ndarray, v0: np.ndarray, n: int, seed: int):
    """Simple directed double-edge switches; exactly preserve every in/out degree."""
    u, v = u0.copy(), v0.copy()
    rng = np.random.default_rng(seed)
    m = len(u)
    keys = {int(a) * n + int(b) for a, b in zip(u.tolist(), v.tolist())}
    accepted = attempts = 0
    target = 20 * m
    max_attempts = target * 500
    while accepted < target:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(f"switch chain stalled after {accepted}/{target} accepted swaps")
        i, j = rng.integers(0, m, size=2)
        if i == j:
            continue
        a, b, c, d = int(u[i]), int(v[i]), int(u[j]), int(v[j])
        # These also rule out a no-op proposal and cross-proposal collision.
        if a == c or b == d or a == d or c == b:
            continue
        new1, new2 = a * n + d, c * n + b
        if new1 == new2 or new1 in keys or new2 in keys:
            continue
        old1, old2 = a * n + b, c * n + d
        keys.remove(old1)
        keys.remove(old2)
        keys.add(new1)
        keys.add(new2)
        v[i], v[j] = d, b
        accepted += 1
    if not np.array_equal(np.bincount(u, minlength=n), np.bincount(u0, minlength=n)):
        raise AssertionError("out-degree preservation failed")
    if not np.array_equal(np.bincount(v, minlength=n), np.bincount(v0, minlength=n)):
        raise AssertionError("in-degree preservation failed")
    return u, v, {"accepted_swaps": accepted, "attempts": attempts}


def matched_density_random(n: int, m: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Uniform simple directed G(n,m) without self loops, via Floyd's sampler."""
    population = n * (n - 1)
    if m > population:
        raise ValueError("too many edges for a simple directed graph")
    rng = np.random.default_rng(seed)
    chosen: set[int] = set()
    for j in range(population - m, population):
        t = int(rng.integers(0, j + 1))
        chosen.add(j if t in chosen else t)
    codes = np.fromiter(chosen, dtype=np.int64, count=m)
    u = codes // (n - 1)
    v = codes % (n - 1)
    v += (v >= u)
    return u.astype(np.int32), v.astype(np.int32)


def summarize(real: dict, controls: list[dict]) -> dict:
    names = [
        "reciprocity_fraction",
        "directed_3_cycle_rate_per_arc",
        "feed_forward_motif_rate_per_arc",
        "canonical_edge_sequence_lz_ratio",
    ]
    result = {}
    for name in names:
        values = np.array([item["metrics"][name] for item in controls], dtype=float)
        observed = float(real[name])
        mean = float(values.mean())
        sd = float(values.std(ddof=1))
        outside = observed < float(values.min()) or observed > float(values.max())
        z = None if sd == 0 else (observed - mean) / sd
        separated = bool(outside and z is not None and abs(z) >= 3.0)
        result[name] = {
            "observed": observed,
            "control_values": values.tolist(),
            "control_mean": mean,
            "control_sample_sd": sd,
            "observed_minus_control_mean": observed - mean,
            "standardized_difference": z,
            "outside_all_control_draws": outside,
            "decision": "separated" if separated else "not_separated",
        }
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    node_ids, u, v, parsing = parse_slice(args.data)
    observed = compute_metrics(u, v, node_ids)
    n, m = len(node_ids), len(u)

    degree_controls = []
    density_controls = []
    for seed in SEEDS:
        cu, cv, switch_info = degree_preserving_switches(u, v, n, seed)
        degree_controls.append({"seed": seed, "switch_info": switch_info, "metrics": compute_metrics(cu, cv, node_ids)})
        ru, rv = matched_density_random(n, m, seed)
        density_controls.append({"seed": seed, "metrics": compute_metrics(ru, rv, node_ids)})
        print(f"completed seed {seed}", flush=True)

    output = {
        "experiment": "FlyWire challenge-connectome structural-order probe",
        "scope": "Structure-only graph analysis; no claim about neural function, behaviour, meaning, cognition, consciousness, or quantum/FBSC mechanisms.",
        "input": {
            "url": URL,
            "local_filename": args.data.name,
            "sha256": sha256(args.data),
            "retrieval_date_utc": datetime.now(timezone.utc).date().isoformat(),
            "header": EXPECTED_HEADER,
            "node_id_parsing": "Python int validated to signed int64 range; never parsed as float.",
            "edge_interpretation": "directed, weighted input; binary simple directed topology analysis",
        },
        "slice": {
            "definition": "induced nodes satisfying splitmix64(uint64(node_id)) mod 10 == 0",
            "node_count": n,
            "arc_count": m,
            "density": m / (n * (n - 1)),
            **parsing,
        },
        "observed": observed,
        "ensembles": {
            "degree_preserving_directed_switch": {
                "description": "20*m accepted simple directed double-edge switches per seed; exact in/out degree preservation",
                "seeds": SEEDS,
                "runs": degree_controls,
                "comparison": summarize(observed, degree_controls),
            },
            "matched_density_directed_gnm": {
                "description": "uniform simple directed G(n,m) without self loops on the same node set",
                "seeds": SEEDS,
                "runs": density_controls,
                "comparison": summarize(observed, density_controls),
            },
        },
        "non_transfer": "The Burau braid invariant is not used because it is defined on braid words, not arbitrary graphs.",
        "decision_rule": "separated iff observed is outside all five controls and |observed-control_mean|/sample_sd >= 3; otherwise not_separated.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "nodes": n, "arcs": m,
        "degree_preserving": output["ensembles"]["degree_preserving_directed_switch"]["comparison"],
        "matched_density": output["ensembles"]["matched_density_directed_gnm"]["comparison"],
    }, indent=2))


if __name__ == "__main__":
    main()
