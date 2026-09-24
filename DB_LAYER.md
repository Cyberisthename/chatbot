# JARVIS DB Layer — run/state/experiment provenance (repo-local)

Short owner-facing guide to the storage layer behind
"connect jarvis to a database, train after".

## What it is

`src/quantum_llm/db_store.py` is a **provenance store** for JARVIS runs,
optimized states, and experiments. Per the owner's repo-local pivot
(2026-09-22), it is a **repo-local SQLite file** (`jarvis.db` in the repo
root): no external database, no env vars, no credentials, no network. The
FBSC core stays deterministic and runs identically regardless of storage —
this layer only records *what ran and what came out of it*.

> Honesty boundary: the DB records provenance — it is NOT a training loop and
> JARVIS does not "learn" from the database. The training hook (runs driving
> the next CMA-ES/evolutionary search) is a separate task; see the PR
> description for the training-design note.

## Configuration — none (default); explicit overrides optional

There is nothing to configure. The store always points at the repo-local
`jarvis.db` (created + migrated automatically on first use). No env vars, no
credentials, no network.

Explicit overrides (accepted for tests/dev only — **never the default, never
env-driven**):

- `sqlite:///path` or `sqlite:///:memory:` — any SQLite path (stdlib `sqlite3`).
- `postgres://...` / `postgresql://...` — Postgres remains a supported
  override (the option is not removed, it is just never the default). Requires
  `pip install "psycopg2-binary>=2.9"` (lazy import; the module stays
  importable without it) and a reachable server.
- Any other scheme raises a clear `ValueError`.
- A DB write failure can never block the deterministic FBSC core: the
  optimizer hook wraps every call and never raises.

## Schema (created idempotently via `init_db()`)

- **runs** — one row per optimizer execution:
  `run_id` (sha256 hash of seed+objective_version+timestamp), `started_at`,
  `seed_triple` (JSON `[s1,s2,s3]`), `objective`, `objective_version`, `mse`,
  `metrics` (JSON report), `source`, `created_at`.
- **states** — verified optimized seeds:
  `seed_triple`, `verifier` (e.g. `seedopt_fresh_instance`), `mse`,
  `metrics` (verified metrics JSON), `source_file`, `source_run`, timestamps.
- **experiments** — provenance for anyon/validation/training work:
  `id`, `kind` (`anyon`|`validation`|`training`, CHECK-constrained),
  `artifact_path`, `verdict` (JSON), timestamps.

## Wiring

`variational_seed_optimizer.py` calls `_record_run_db(report)` after each run:
it opens the default repo-local store (no env vars needed) and inserts a
**runs** row (seed triple, objective version, reconstruction MSE, full
metrics JSON). Any failure prints one line and optimization continues — the
optimizer never depends on the DB.

## CLI

```bash
# init tables (idempotent) + ingest the seed-optimizer reports + counts
python3 -m quantum_llm.db_store --init \
    --sync-seedopt artifacts/seed_optimizer --counts
# or explicitly list what is stored
python3 -m quantum_llm.db_store --list-runs --list-states --list-experiments
```

Flags: `--init`, `--sync-seedopt <file|dir>`, `--list-runs`, `--list-states`,
`--list-experiments`, `--counts`, `--url <explicit sqlite:// or postgres:// override>`.

## import path

From the repo root: `from src.quantum_llm.db_store import get_store` (or
`PYTHONPATH=src python3 -m quantum_llm.db_store ...`).

## Tests

`tests/test_db_store.py` (Python `unittest`, no external DB needed — sqlite
in-memory + a `DbStore` built on explicit `sqlite:///:memory:` URLs, plus
repo-local-default, env-ignored, and explicit-override coverage).
Run with `python3 -m unittest tests.test_db_store -v`.