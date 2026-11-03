# Pre-registration: FlyWire challenge-connectome structural-order probe

**Registered:** 2026-10-29, before downloading or inspecting the data file.

## Question

Does a deterministic induced slice of the FlyWire challenge-connectome distribution carry graph structure that is absent after degree-preserving directed randomization? This is a graph-structure test only. It is not a test of neural function, behaviour, meaning, cognition, consciousness, or any quantum/FBSC claim.

## Exact input and deterministic slice

* Input URL: `https://storage.googleapis.com/flywire-data/challenges/mfas/connectome_graph.csv.gz`
* Expected CSV columns: `Source Node ID,Target Node ID,Edge Weight`.
* Input interpretation: directed, weighted edges. Node identifiers will be parsed as signed 64-bit integers, never floats.
* Slice: an **induced 10% node slice**. A node is retained iff `splitmix64(uint64(node_id)) mod 10 == 0`, where SplitMix64 is the standard unsigned 64-bit integer mixer implemented verbatim in the analysis script. An edge is retained only when both endpoints are retained. Self loops, if present, are excluded from topology metrics and reported.
* The topology analysis is **binarized**: an ordered source--target pair is one edge regardless of its synapse-count weight. This is deliberate: the requested degree-preserving edge-switch null is defined for simple directed topology, while raw synapse weights would require an additional, separately specified strength-preserving null. The raw weight column is parsed and only used for input validation / descriptive totals, not as a topology metric.
* Duplicate ordered pairs, if any, are collapsed to a single arc and their occurrence is reported.

## Pre-specified metrics

All metrics use the same loop-free, binary, induced directed graph.

1. **Reciprocity:** fraction of arcs whose reverse arc exists.
2. **Directed 3-cycle rate:** number of directed 3-cycles divided by number of arcs. A directed 3-cycle is counted once per cyclic vertex triple.
3. **Feed-forward motif rate:** count of ordered triples `(x,y,z)` with `x→y`, `x→z`, and `y→z`, divided by number of arcs. This directed three-node motif is counted by role assignment.
4. **Canonical edge-sequence LZ ratio:** concatenate `uint32` target ranks within source-rank order after ranking nodes by `(out-degree, in-degree, int64 node id)`; compress those bytes using Python `zlib` level 9 and report compressed / uncompressed bytes. The node ranking is recomputed per graph using the same degree-first rule. This is an ordering-dependent, lossless structural coding proxy, not an estimate of semantic information.
5. **In- and out-degree distribution diagnostics:** mean, standard deviation, maximum, and Shannon entropy (base 2) of each degree sequence. These are reported to validate controls. Under the directed degree-preserving null they should be identical (up to representation), so they are not interpreted as a separation result there.

No Burau braid invariant is used: it is defined on braid words, not arbitrary graphs, and therefore does not transfer to this probe.

## Control ensembles

Each ensemble has exactly five independent seeded realizations, with seeds:
`104729, 130363, 155921, 196613, 225307`.

1. **Directed degree-preserving edge-switch null.** Starting from the observed simple directed graph, repeatedly choose two arcs `(a,b)` and `(c,d)` and propose `(a,d),(c,b)`; retain only proposals that introduce neither a self loop nor a duplicate, and that change two arcs. The chain runs for `20 × m` accepted swaps per seed, where `m` is the observed number of arcs. Thus in- and out-degree sequences are exactly preserved.
2. **Matched-density directed random graph.** Draw `m` distinct ordered non-self pairs uniformly without replacement from the `n(n-1)` possible ordered pairs on the same `n` nodes. It preserves node count and edge density, but not degree sequences.

## Decision rule

For each of metrics 1--4 and each ensemble separately, call the observed graph **separated** only if both conditions hold: (a) its value is strictly outside the inclusive min--max range of all five control draws, and (b) its absolute standardized difference from the five-control mean is at least 3 control standard deviations (sample SD, `ddof=1`). Otherwise call it **not separated**. Exact degree preservation diagnostics are reported as expected invariants, not promoted to a discovery. Every result will list all five control values, their mean and sample SD, and the observed-control difference.

## Reproduction boundary

The data file is third-party external data and will not be committed. The analysis will record the retrieval date, SHA-256 of the downloaded gzip, and source/licence statement in `docs/external_data/flywire/README.md`. If no explicit licence is supplied with the source location, the README will say exactly: **"no explicit licence found"**.
