# JARVIS DB Layer — run/state/experiment provenance (DB-optional)

Short owner-facing guide to the storage layer added for
"connect jarvis to a database, train after".

## What it is

`src/quantum_llm/db_store.py` is a **provenance store** for JARVIS runs,
optimized states, and experiments. It is deliberately **DB-optional**: the
FBSC core stays deterministic and runs identically whether or not a database
is configured.

> Honesty boundary: the DB records *what ran and what came out of it* — it is
> NOT a training loop and JARVIS does not "learn" from the database. The
> training hook (runs driving the next CMA-ES/evolutionary search) is a
> separate task; see the PR description for the training-design note.

## Configuration

| Env var | Meaning |
|---|---|
| `DATABASE_URL` | `postgres://...` / `postgresql://...` (Tiger Cloud / Postgres) or `sqlite:///path` (local dev) |

- `DATABASE_URL` absent  → store is DISABLED; writes are safe no-ops,
  `strict=True` raises a clear `DbNotConfiguredError`.
- Postgres requires `pip install "psycopg2-binary>=2.9"` (lazy import — the
  module stays importable without it). SQLite uses the Python stdlib.
- No connection is opened until the first actual operation (lazy init), so
  nothing touches the network until a secret lands.

The Tiger-specific keys (`TIGER_PUBLIC_KEY`, `TIGER_SECRET_KEY`,
`TIGER_PROJECT_ID`) are for Tiger's HTTP API if you use it; this layer consumes
a standard Postgres `DATABASE_URL`, so those keys are optional.

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

`variational_seed_optimizer.py` now calls `_record_run_db(report)` after each
run: when `DATABASE_URL` is set it inserts a **runs** row (seed triple,
objective version, reconstruction MSE, full metrics JSON). When the DB is
absent or errors, the hook prints one line and optimization continues — the
optimizer never depends on the DB.

## CLI

```bash
# init tables + ingest the four existing seed-optimizer reports
DATABASE_URL=postgres://... python3 -m quantum_llm.db_store --init \
    --sync-seedopt artifacts/seed_optimizer --counts
# or locally with sqlite
python3 -m quantum_llm.db_store --url sqlite:////tmp/jarvis.db \
    --init --sync-seedopt artifacts/seed_optimizer --list-runs --list-states
```

Flags: `--init`, `--sync-seedopt <file|dir>`, `--list-runs`, `--list-states`,
`--list-experiments`, `--counts`, `--url <override>`.

## import path

From the repo root: `from src.quantum_llm.db_store import get_store` (or
`PYTHONPATH=src python3 -m quantum_llm.db_store ...`).

## Tests

`tests/test_db_store.py` (Python `unittest`, no external DB needed — sqlite
in-memory + a `DbStore` built on an explicit `sqlite:///:memory:` URL).
Run with `python3 -m unittest tests.test_db_store -v`.