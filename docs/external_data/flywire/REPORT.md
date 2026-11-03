# FlyWire challenge-connectome: structural-order probe

## Verdict

**On the pre-registered deterministic induced 10% node slice, the FlyWire challenge-connectome distribution is structurally separated from both control ensembles on every four registered topology/coding metrics.** In particular, the observed binary directed graph has much greater reciprocity, directed 3-cycle rate, and feed-forward-motif rate than degree-preserving directed edge-switch controls, and its registered canonical edge sequence compresses more strongly.

This is a result about **graph structure only**. It does not demonstrate neural function, behaviour, meaning, cognition, consciousness, sentience, a mind, or an uploaded brain. It does not validate, use, or extend FBSC or any quantum claim. As with undeciphered-script work, structure is not meaning.

## Data and registered slice

The input is FlyWire's **CHALLENGE DISTRIBUTION** for algorithm benchmarking, derived from the proofread connectome—not a claim to have analyzed an unrestricted copy of the whole fly brain. It is a third-party directed, weighted edge list. The registered analysis binarizes it because the degree-preserving null is a simple-directed-topology null; raw synapse weights were not tested against a strength-preserving null.

* Gzip SHA-256: `5e99efb931e41dfbc4994dbd6c2ad1c2df5c188bf4c6c88ccd0f8c02c642a04d`
* Rows parsed: 5,657,719; raw weight sum: 41,912,141.
* Slice: retain nodes satisfying `splitmix64(uint64(node_id)) mod 10 == 0`, then retain only edges whose two endpoints remain.
* Binary directed slice: **13,550 nodes**, **55,558 arcs**, density **0.000302622**. No self loops or duplicate ordered pairs occurred in the retained slice.
* The physical CSV has the header spelling `Source Node  ID` (two spaces); IDs were validated as int64 and never parsed as floating point.

The immutable pre-registration is [`PREREGISTRATION.md`](PREREGISTRATION.md). Its commit (`00324ca`, 2026-10-29T12:34:09Z) was pushed before downloading the gzip. A post-download parser-only correction records the source's two-space header spelling; it made no metric, slice, control, or decision-rule change.

## Controls and decision rule

Each ensemble used five independent seeds: 104729, 130363, 155921, 196613, and 225307.

1. **Directed degree-preserving edge switches:** 20 × 55,558 = **1,111,160 accepted switches** per seed, preserving every node's in-degree and out-degree exactly. Attempts ranged from 1,133,555 to 1,134,109.
2. **Matched-density directed G(n,m):** uniform simple directed random graphs on the same 13,550 nodes with the same 55,558 arcs; degree sequences are not preserved.

A metric is called *separated* only if the observed value is outside all five control values and at least 3 sample control SD from the control mean. The JSON has each individual draw and exact calculations.

## Results

Rates are normalized per observed arc; LZ is compressed/raw bytes (lower means more compressible). Values are `observed; degree-preserving mean ± sample SD; matched-density mean ± sample SD`.

| Registered metric | Observed | Degree-preserving | Matched density | Decision |
|---|---:|---:|---:|---|
| Reciprocity fraction | 0.201699 | 0.006401 ± 0.000374 | 0.000266 ± 0.000055 | separated from both |
| Directed 3-cycle rate / arc | 0.138216 | 0.028352 ± 0.001671 | 0.000414 ± 0.000087 | separated from both |
| Feed-forward motif rate / arc | 0.715289 | 0.118942 ± 0.005041 | 0.001321 ± 0.000126 | separated from both |
| Canonical edge-sequence LZ ratio | 0.504333 | 0.536969 ± 0.000454 | 0.583686 ± 0.000294 | separated from both |

Effect sizes against the degree-preserving mean are: reciprocity **+0.195299** (522.77 SD), 3-cycle rate **+0.109864** (65.77 SD), feed-forward rate **+0.596346** (118.30 SD), and LZ ratio **−0.032636** (−71.91 SD). All observed values lie outside the five-draw ranges. Against matched density the corresponding standardized differences are +3689.63, +1579.31, +5649.39, and −269.64 SD.

No registered topology/coding metric was *not separated* in this slice. The registered degree-distribution summaries are deliberately **not** a discovery metric under the degree-preserving ensemble: they are expected invariants and were checked exactly by assertion. Observed mean in- and out-degree are both 4.100221; maximum in-degree is 872 and maximum out-degree is 777. Matched-density controls intentionally do not retain that degree shape.

The observed counts behind the two motif rates are 7,679 directed 3-cycles and 39,740 role-assigned feed-forward motifs. The registered LZ stream contained 222,232 raw bytes and 112,079 compressed bytes.

## Non-transfer finding

The Burau braid invariant was **not** applied. It is defined for braid words, not arbitrary directed graphs, so transferring it to this connectome would not be mathematically justified.

## What this does not show

* It does not establish a property of every FlyWire release, every brain region, every fly, or the unrestricted full connectome; the result is for this exact challenge-distribution file and registered slice.
* It does not say why these structural differences exist, whether they are biological mechanisms, or whether they affect computation or behaviour.
* It does not analyze synaptic weights beyond parsing and reporting their raw sum.
* It does not establish causal, functional, semantic, cognitive, consciousness, sentience, or brain-upload claims.
* It does not show a quantum effect or a connection to FBSC.
* It does not transfer the source data into the owned core. The source is third-party; see [`README.md`](README.md), including the exact statement **"no explicit licence found"**.

## Exact reproduction

The raw gzip is external and intentionally not committed. From the repository root, with a downloaded copy at `/path/to/connectome_graph.csv.gz`:

```bash
python3 -m venv .venv
.venv/bin/pip install --no-cache-dir numpy scipy
.venv/bin/python scripts/flywire_structure_probe.py \
  --data /path/to/connectome_graph.csv.gz \
  --output docs/external_data/flywire/RESULTS.json
```

The expected input URL, SHA-256, control seeds, metrics, and decision rule are in `PREREGISTRATION.md` and `RESULTS.json`.
