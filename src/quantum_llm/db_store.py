"""
JARVIS database storage layer — run/state/experiment provenance (repo-local).

Owned original code; stdlib-only core. **Repo-local pivot (owner direction
2026-09-22):** there is no external database — the store defaults to a
repo-local ``jarvis.db`` SQLite file, needs no env vars, no credentials and
no network. An explicit ``sqlite://...`` URL is accepted only as an override
for tests/dev. Backends beyond SQLite were removed (no Postgres / psycopg2).

Store contents:
- runs:       one row per optimizer execution (seed triple, run_id hash,
              timestamp, objective version, reconstruction MSE, metrics JSON)
- states:     optimized seeds + verified metrics (e.g. synced from
              ``seed_optimizer_report.json`` / seedopt_results-style files)
- experiments: provenance rows for anyon / validation / training experiments
              (id, kind, artifact path, verdict JSON)

Graceful degradation
--------------------
If ``jarvis.db`` is missing it is created and migrated via the init SQL on
first use (``sqlite3`` creates the file; ``init_db()`` creates the tables).
The store is always configured; a DB write failure never blocks the
deterministic FBSC core (the optimizer hook is wrapped and never raises).

Honesty boundary
----------------
This layer stores *provenance of runs and experiments* — it is NOT a training
loop and JARVIS does not "learn" from the DB. See DB_LAYER.md; the training
hook (runs driving the next CMA-ES/evolutionary search) is a separate task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

log = logging.getLogger("quantum_llm.db_store")

# Repo-local default: <repo root>/jarvis.db (module lives in src/quantum_llm/,
# so parents[2] is the repo root). No env vars, no credentials, no network.
DEFAULT_DB_PATH = str(Path(__file__).resolve().parents[2] / "jarvis.db")
DEFAULT_DB_URL = f"sqlite:///{DEFAULT_DB_PATH}"

SCHEMA_VERSION = 1
OBJECTIVE_VERSION_PREFIX = "v1"  # bump when the objective-space encoding changes

# Kinds allowed in the experiments table.
EXPERIMENT_KINDS = ("anyon", "validation", "training")

INIT_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id             TEXT PRIMARY KEY,
    started_at         TEXT NOT NULL,              -- ISO-8601 UTC
    seed_triple        TEXT NOT NULL,              -- JSON [s1, s2, s3]
    objective          TEXT NOT NULL,
    objective_version  TEXT NOT NULL,
    mse                REAL,                       -- reconstruction MSE (0.0 = core-exact)
    metrics            TEXT NOT NULL,              -- JSON blob (report/params/etc.)
    source             TEXT,                       -- 'variational_seed_optimizer' | 'seedopt_sync' | ...
    created_at         TEXT NOT NULL               -- ISO-8601 UTC
);
CREATE INDEX IF NOT EXISTS idx_runs_objective_version ON runs (objective_version);
CREATE INDEX IF NOT EXISTS idx_runs_started_at ON runs (started_at DESC);

CREATE TABLE IF NOT EXISTS states (
    state_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    seed_triple  TEXT NOT NULL,                    -- JSON [s1, s2, s3]
    verifier     TEXT NOT NULL,                    -- e.g. 'seedopt_fresh_instance'
    mse          REAL,
    metrics      TEXT NOT NULL,                    -- JSON blob (verified metrics)
    source_file  TEXT,                             -- provenance path
    source_run   TEXT,                             -- runs.run_id if derived from a run
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    UNIQUE (seed_triple, verifier, source_file)
);

CREATE TABLE IF NOT EXISTS experiments (
    id            TEXT PRIMARY KEY,
    kind          TEXT NOT NULL CHECK (kind IN ('anyon', 'validation', 'training')),
    artifact_path TEXT,
    verdict       TEXT NOT NULL,                   -- JSON blob
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_experiments_kind ON experiments (kind);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical_seed(seed_triple: Iterable[float]) -> str:
    """Canonical, stable JSON encoding of the 3-seed for hashing/reuse."""
    vals = [round(float(v), 12) for v in seed_triple]
    return json.dumps(vals, separators=(",", ":"))


def _short_hash(text: str, length: int = 8) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def _json_default(o: Any) -> Any:
    """JSON fallback that demotes numpy scalars/arrays without importing numpy.

    Keeps the DB layer stdlib-only while remaining safe for reports that
    contain np.float64/np.ndarray values straight out of the optimizer.
    """
    item = getattr(o, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    tolist = getattr(o, "tolist", None)
    if callable(tolist):
        try:
            return tolist()
        except Exception:
            pass
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _dumps(obj: Any) -> str:
    return json.dumps(obj, default=_json_default, separators=(",", ":"))


def objective_version(objective: str, params: Optional[Dict[str, Any]] = None) -> str:
    """Stable version string describing the objective space explored.

    Encodes the objective name plus the hyper-parameters that change the
    search (weights, qubit count). Two runs with the same objective_version
    are directly comparable. The prefix is bumped only when the encoding
    itself changes, not per run.
    """
    params = dict(params or {})
    weights = params.get("weights") or {}
    if not isinstance(weights, dict):
        weights = {}
    w_canon = json.dumps({k: float(v) for k, v in sorted(weights.items())},
                         sort_keys=True, separators=(",", ":"))
    nq = int(params.get("n_effective_qubits", 0))
    payload = f"{OBJECTIVE_VERSION_PREFIX}:{objective}:{w_canon}:nq{nq}"
    return f"{OBJECTIVE_VERSION_PREFIX}:{objective}:{_short_hash(payload)}"


# Module-global alias so methods whose local variables are named
# ``objective_version`` can still call the function without shadowing.
_objective_version_fn = objective_version


def _trim_report(report: Dict[str, Any]) -> Dict[str, Any]:
    """Trim bulky convergence traces out of the metrics JSON kept in the DB.

    The full report stays on disk next to the run; the DB row keeps the
    decision-relevant payload without 60 generations of per-generation history.
    """
    trimmed = {k: v for k, v in report.items()
               if k not in ("convergence", "best_per_generation")}
    return trimmed


class DbStore:
    """Lazy, repo-local SQLite store. Always configured (defaults to
    ``<repo root>/jarvis.db``); an explicit ``sqlite://`` URL overrides the
    default for tests/dev. No env vars, no credentials, no network.
    """

    def __init__(self, database_url: Optional[str] = None):
        self._url = database_url if database_url is not None else DEFAULT_DB_URL
        self._conn: Any = None
        self._backend: Optional[str] = None  # 'sqlite' | 'unsupported'
        self._last_insert_id: Optional[int] = None  # captured in _execute()

    # ------------------------------------------------------------------ state
    @property
    def configured(self) -> bool:
        # Always True: the store defaults to repo-local jarvis.db and is
        # created+migrated on first use. Kept for the optimizer hook's guard.
        return True

    @property
    def backend(self) -> Optional[str]:
        if self._backend is None:
            if self._url.startswith("sqlite://"):
                self._backend = "sqlite"
            else:
                self._backend = "unsupported"
        return self._backend

    @property
    def database_url(self) -> Optional[str]:
        return self._url

    # ---------------------------------------------------------------- connect
    def _connect(self):
        """Lazy connect: no connection is ever opened until first use."""
        if self._conn is not None:
            return self._conn
        if not self._url.startswith("sqlite://"):
            raise ValueError(
                f"Unsupported DB URL for JARVIS DB store: {self._url!r}. "
                f"The repo-local pivot supports sqlite:// URLs only "
                f"(default: {DEFAULT_DB_URL}).")
        raw = self._url[len("sqlite://"):]
        # Normalize: sqlite://, sqlite:///:memory: and sqlite://:memory:
        # all mean an in-memory database in sqlite3 terms.
        if raw in ("", "/:memory:", ":memory:"):
            path = ":memory:"
        else:
            path = raw
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self.init_db()
        return self._conn

    def _q(self, sql: str) -> str:
        """Map %s placeholders to ? for sqlite (dialect portability)."""
        return sql.replace("%s", "?")

    def _execute(self, sql: str, params: tuple = ()):
        conn = self._connect()
        cur = conn.cursor()
        try:
            cur.execute(self._q(sql), params)
            conn.commit()
            self._last_insert_id = getattr(cur, "lastrowid", None)
        finally:
            cur.close()
        return conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------ init
    def init_db(self, _force: bool = False) -> None:
        """Create tables (idempotent). Safe to call repeatedly."""
        conn = self._connect()
        cur = conn.cursor()
        try:
            for stmt in [s.strip() for s in INIT_SQL.split(";") if s.strip()]:
                cur.execute(self._q(stmt))
            conn.commit()
        finally:
            cur.close()

    @staticmethod
    def init_sql() -> str:
        """Expose the DDL for review/tooling."""
        return INIT_SQL

    # ------------------------------------------------------------------ runs
    def record_run(
        self,
        seed_triple: Iterable[float],
        objective: str,
        objective_version: Optional[str] = None,
        mse: Optional[float] = None,
        metrics: Optional[Dict[str, Any]] = None,
        source: str = "unknown",
        started_at: Optional[str] = None,
    ) -> Optional[str]:
        """Insert one optimizer run and return its run_id.

        The store is always configured (repo-local jarvis.db); a DB write
        failure propagates, but the optimizer hook wraps this and never lets
        the deterministic FBSC core depend on the DB.
        """
        started_at = started_at or _now_iso()
        objective_version = objective_version or _objective_version_fn(objective)
        seed_json = _canonical_seed(seed_triple)
        run_id = _short_hash(f"{seed_json}|{objective_version}|{started_at}", 16)
        metrics_json = _dumps(metrics or {})
        created = _now_iso()
        self._execute(
            "INSERT INTO runs (run_id, started_at, seed_triple, objective, "
            " objective_version, mse, metrics, source, created_at) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) "
            "ON CONFLICT(run_id) DO NOTHING",
            (run_id, started_at, seed_json, objective,
             objective_version, mse, metrics_json, source, created),
        )
        return run_id

    def list_runs(self, limit: int = 50,
                  objective_version: Optional[str] = None) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM runs"
        params: list = []
        if objective_version:
            sql += " WHERE objective_version = %s"
            params.append(objective_version)
        sql += " ORDER BY started_at DESC LIMIT %s"
        params.append(int(limit))
        return self._rows(sql, params)

    # ---------------------------------------------------------------- states
    def record_state(
        self,
        seed_triple: Iterable[float],
        metrics: Dict[str, Any],
        verifier: str = "seedopt_fresh_instance",
        mse: Optional[float] = None,
        source_file: Optional[str] = None,
        source_run: Optional[str] = None,
    ) -> Optional[int]:
        """Upsert one verified state and return its state_id."""
        seed_json = _canonical_seed(seed_triple)
        metrics_json = _dumps(metrics or {})
        if mse is None:
            mse = _extract_mse(metrics)
        now = _now_iso()
        rows = self._rows(
            "SELECT state_id FROM states WHERE seed_triple=%s AND verifier=%s "
            "AND source_file IS %s",
            (seed_json, verifier, source_file),
        )
        if rows:
            state_id = rows[0]["state_id"]
            self._execute(
                "UPDATE states SET metrics=%s, mse=%s, source_run=%s, "
                "updated_at=%s WHERE state_id=%s",
                (metrics_json, mse, source_run, now, state_id),
            )
            return int(state_id)
        self._execute(
            "INSERT INTO states (seed_triple, verifier, mse, metrics, "
            " source_file, source_run, created_at, updated_at) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
            (seed_json, verifier, mse, metrics_json, source_file,
             source_run, now, now),
        )
        return self._last_id()

    def list_states(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self._rows(
            "SELECT * FROM states ORDER BY updated_at DESC LIMIT %s", (int(limit),))

    # ------------------------------------------------------------ experiments
    def record_experiment(
        self,
        experiment_id: str,
        kind: str,
        verdict: Dict[str, Any],
        artifact_path: Optional[str] = None,
    ) -> bool:
        """Insert/update one experiment provenance row. Returns True if written."""
        if kind not in EXPERIMENT_KINDS:
            raise ValueError(
                f"experiment kind must be one of {EXPERIMENT_KINDS}, got {kind!r}")
        verdict_json = _dumps(verdict or {})
        now = _now_iso()
        rows = self._rows("SELECT id FROM experiments WHERE id=%s", (experiment_id,))
        if rows:
            self._execute(
                "UPDATE experiments SET kind=%s, verdict=%s, artifact_path=%s, "
                "updated_at=%s WHERE id=%s",
                (kind, verdict_json, artifact_path, now, experiment_id),
            )
        else:
            self._execute(
                "INSERT INTO experiments (id, kind, artifact_path, verdict, "
                " created_at, updated_at) VALUES (%s,%s,%s,%s,%s,%s)",
                (experiment_id, kind, artifact_path, verdict_json, now, now),
            )
        return True

    def list_experiments(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self._rows(
            "SELECT * FROM experiments ORDER BY created_at DESC LIMIT %s",
            (int(limit),))

    # ---------------------------------------------------------------- helpers
    def _rows(self, sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
        conn = self._connect()
        cur = conn.cursor()
        try:
            cur.execute(self._q(sql), params)
            rows = [dict(r) for r in cur.fetchall()]
        finally:
            cur.close()
        return rows

    def _last_id(self) -> Optional[int]:
        rid = self._last_insert_id
        return int(rid) if rid is not None else None

    def counts(self) -> Dict[str, int]:
        result = {"runs": 0, "states": 0, "experiments": 0}
        for table in result:
            rows = self._rows(f"SELECT COUNT(*) AS n FROM {table}")
            if rows:
                result[table] = int(rows[0]["n"])
        return result

    # -------------------------------------------------------------- sync CLI
    def sync_seedopt(self, path: str) -> List[str]:
        """Ingest seed_optimizer_report.json(s) into states (+ run rows).

        ``path`` may be a single report file or a directory scanned for
        ``seed_optimizer_report.json``. Each report becomes:
        - one states row (optimized seed + verified_fresh_instance metrics)
        - one runs row (source='seedopt_sync', started_at from file mtime)
        Returns the list of files ingested.
        """
        ingested: List[str] = []
        p = Path(path)
        files: List[Path] = []
        if p.is_file():
            files = [p]
        elif p.is_dir():
            files = sorted(p.rglob("seed_optimizer_report.json"))
        else:
            raise FileNotFoundError(f"sync_seedopt: no such path: {path}")
        for f in files:
            try:
                report = json.loads(f.read_text("utf-8"))
            except (OSError, ValueError) as exc:
                log.warning("sync_seedopt: skipping %s (%s)", f, exc)
                continue
            seed = report.get("optimized_seed")
            if not seed or len(seed) != 3:
                log.warning("sync_seedopt: %s has no 3-element optimized_seed", f)
                continue
            objective = report.get("objective", "combined")
            params = report.get("params", {})
            mse = float((report.get("reconstruction") or {}).get(
                "reconstruction_mse", 0.0))
            metrics = {
                "after": report.get("after", {}),
                "verified_fresh_instance": report.get("verified_fresh_instance", {}),
                "improvement": report.get("improvement", {}),
                "improvement_pct": report.get("improvement_pct", {}),
                "parameters": params,
                "reconstruction": report.get("reconstruction", {}),
                "verify_ok": report.get("verify_ok", None),
            }
            ov = objective_version(objective, params)
            started_at = _ts_from_mtime(f)
            run_id = self.record_run(
                seed_triple=seed, objective=objective,
                objective_version=ov, mse=mse,
                metrics=metrics, source="seedopt_sync", started_at=started_at,
            )
            self.record_state(
                seed_triple=seed, metrics=metrics,
                verifier="seedopt_fresh_instance", mse=mse,
                source_file=str(f), source_run=run_id,
            )
            ingested.append(str(f))
        return ingested


# --------------------------------------------------------------------- export
def get_store(database_url: Optional[str] = None) -> DbStore:
    """Singleton-ish accessor. Defaults to the repo-local jarvis.db store;
    ``database_url`` overrides (sqlite://... URLs only) for tests/dev."""
    if not hasattr(get_store, "_singleton") or get_store._singleton is None:
        get_store._singleton = DbStore(database_url)  # type: ignore[attr-defined]
    return get_store._singleton


def _extract_mse(metrics: Dict[str, Any]) -> Optional[float]:
    """Best-effort MSE extraction from a metrics blob."""
    try:
        if "mse" in metrics:
            return float(metrics["mse"])
        rec = metrics.get("reconstruction")
        if isinstance(rec, dict) and "reconstruction_mse" in rec:
            return float(rec["reconstruction_mse"])
    except (TypeError, ValueError):
        return None
    return None


def _ts_from_mtime(path: Path) -> str:
    try:
        ts = path.stat().st_mtime
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")
    except OSError:
        return _now_iso()


# ----------------------------------------------------------------------- CLI
def _cli(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="db_store",
        description="JARVIS run/state/experiment provenance store (repo-local).")
    ap.add_argument("--url", default=None,
                    help=f"sqlite:// URL override (default: {DEFAULT_DB_URL})")
    ap.add_argument("--init", action="store_true",
                    help="create tables (idempotent)")
    ap.add_argument("--sync-seedopt", metavar="PATH",
                    help="ingest a seed_optimizer_report.json or a directory of them")
    ap.add_argument("--list-runs", action="store_true")
    ap.add_argument("--list-states", action="store_true")
    ap.add_argument("--list-experiments", action="store_true")
    ap.add_argument("--counts", action="store_true")
    args = ap.parse_args(argv)

    store = DbStore(args.url)

    if args.init:
        store.init_db()
        print(f"DB initialised ({store.backend}) at {store.database_url}: "
              f"tables runs/states/experiments.")
    if args.sync_seedopt:
        files = store.sync_seedopt(args.sync_seedopt)
        print(f"sync_seedopt: ingested {len(files)} report(s).")
        for f in files:
            print(f"  - {f}")
    if args.list_runs:
        for r in store.list_runs():
            print(f"run {r['run_id']} | {r['started_at']} | {r['objective']} "
                  f"{r['objective_version']} | mse={r['mse']} | {r['source']}")
    if args.list_states:
        for s in store.list_states():
            print(f"state {s['state_id']} | seed={s['seed_triple']} | "
                  f"verifier={s['verifier']} | mse={s['mse']}")
    if args.list_experiments:
        for e in store.list_experiments():
            print(f"experiment {e['id']} | kind={e['kind']} | "
                  f"artifact={e['artifact_path']}")
    if args.counts:
        print("counts:", json.dumps(store.counts()))
    return 0


if __name__ == "__main__":
    sys.exit(_cli())