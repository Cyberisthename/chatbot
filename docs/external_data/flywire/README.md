# FlyWire external-data boundary

## Source and custody

* **Source:** `https://storage.googleapis.com/flywire-data/challenges/mfas/connectome_graph.csv.gz`
* **Retrieved:** 2026-10-29 UTC.
* **Downloaded gzip SHA-256:** `5e99efb931e41dfbc4994dbd6c2ad1c2df5c188bf4c6c88ccd0f8c02c642a04d`.
* **Source HTTP metadata observed at retrieval:** public `200`; stored content length `39056663`; last modified `2024-06-11T20:48:28Z`.
* **Format actually observed:** CSV header `Source Node  ID,Target Node ID,Edge Weight` (there are two spaces between `Node` and `ID` in the first field). Node identifiers are parsed as signed 64-bit integers, never floats.

This is third-party external data. It is a FlyWire **CHALLENGE DISTRIBUTION** packaged for algorithm benchmarking and derived from the proofread connectome; it is not represented here as an unrestricted copy of the whole fly brain. The gzip file is deliberately excluded from git and is **not part of the owned core**.

## Licence

**no explicit licence found**

The retrieved object header and the CSV header contain no licence statement. This repository does not infer, invent, or grant a licence for the source data. Use of the file remains subject to the source provider's applicable terms.

## Scope boundary

`PREREGISTRATION.md`, `RESULTS.json`, `REPORT.md`, and `scripts/flywire_structure_probe.py` contain an independently reproducible, structure-only graph analysis. They make no claim about neural function, behaviour, meaning, cognition, consciousness, or an uploaded brain. They do not attach the data to FBSC or quantum claims.
