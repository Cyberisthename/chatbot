#!/usr/bin/env python3
"""Unit tests for src/quantum_llm/db_store.py (JARVIS DB provenance layer).

Run:  python3 -m unittest tests.test_db_store -v
Uses sqlite in-memory only — no Postgres / DATABASE_URL needed.
"""
import json
import os
import tempfile
import unittest
from pathlib import Path

from src.quantum_llm.db_store import (
    DbNotConfiguredError,
    DbStore,
    EXPERIMENT_KINDS,
    objective_version,
)

DB_URL = "sqlite:///:memory:"


class TestObjectiveVersion(unittest.TestCase):
    def test_stable_across_dict_key_order(self):
        params_a = {"weights": {"bio_resonance": 0.5, "braid_order": 0.5},
                    "n_effective_qubits": 128}
        params_b = {"n_effective_qubits": 128,
                    "weights": {"braid_order": 0.5, "bio_resonance": 0.5}}
        self.assertEqual(objective_version("combined", params_a),
                         objective_version("combined", params_b))

    def test_differs_when_objective_changes(self):
        params = {"weights": {}, "n_effective_qubits": 128}
        self.assertNotEqual(objective_version("bio_resonance", params),
                            objective_version("braid_order", params))

    def test_differs_when_weights_change(self):
        a = objective_version("combined", {"weights": {"x": 0.5}})
        b = objective_version("combined", {"weights": {"x": 0.7}})
        self.assertNotEqual(a, b)

    def test_prefix(self):
        self.assertTrue(objective_version("combined").startswith("v1:combined:"))


class TestDisabledMode(unittest.TestCase):
    """No DATABASE_URL -> everything degrades gracefully."""

    def setUp(self):
        self._url = os.environ.pop("DATABASE_URL", None)
        self.store = DbStore(database_url=None)

    def tearDown(self):
        if self._url is not None:
            os.environ["DATABASE_URL"] = self._url

    def test_not_configured(self):
        self.assertFalse(self.store.configured)
        self.assertIsNone(self.store.backend)

    def test_record_run_noop(self):
        self.assertIsNone(self.store.record_run(
            [0.5, 1.0, 2.0], "combined", strict=False))

    def test_record_run_strict_raises_clear_error(self):
        with self.assertRaises(DbNotConfiguredError) as ctx:
            self.store.record_run([0.5, 1.0, 2.0], "combined", strict=True)
        self.assertIn("DATABASE_URL", str(ctx.exception))

    def test_record_state_noop(self):
        self.assertIsNone(self.store.record_state([0.5, 1.0, 2.0], {}, strict=False))

    def test_record_experiment_noop(self):
        self.assertFalse(self.store.record_experiment("e1", "anyon", {}, strict=False))

    def test_lists_empty(self):
        self.assertEqual(self.store.list_runs(), [])
        self.assertEqual(self.store.list_states(), [])
        self.assertEqual(self.store.list_experiments(), [])
        self.assertEqual(self.store.counts(),
                         {"runs": 0, "states": 0, "experiments": 0})


class TestSqliteStore(unittest.TestCase):
    def setUp(self):
        self.store = DbStore(DB_URL)

    def test_init_db_idempotent(self):
        self.store.init_db()
        self.store.init_db()  # second call must not raise
        self.assertEqual(self.store.counts(),
                         {"runs": 0, "states": 0, "experiments": 0})

    def test_record_run_and_list(self):
        run_id = self.store.record_run(
            [0.57721, 1.618034, 2.71828], "combined",
            mse=0.0, metrics={"after": {"combined": 0.28}},
            source="variational_seed_optimizer", started_at="2026-09-20T00:00:00Z")
        self.assertIsNotNone(run_id)
        runs = self.store.list_runs()
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["run_id"], run_id)
        self.assertEqual(json.loads(runs[0]["seed_triple"]),
                         [0.57721, 1.618034, 2.71828])
        self.assertEqual(runs[0]["mse"], 0.0)
        self.assertEqual(runs[0]["objective"], "combined")
        self.assertTrue(runs[0]["objective_version"].startswith("v1:combined:"))
        self.assertEqual(json.loads(runs[0]["metrics"])["after"]["combined"], 0.28)

    def test_run_id_deterministic(self):
        kw = dict(seed_triple=[0.5, 1.0, 2.0], objective="combined",
                  started_at="2026-09-20T00:00:00Z")
        id1 = self.store.record_run(**kw)
        id2 = DbStore(DB_URL).record_run(**kw)  # fresh store, same inputs
        self.assertEqual(id1, id2)

    def test_record_state_upsert(self):
        s1 = self.store.record_state(
            [0.5, 1.0, 2.0], {"after": {"combined": 0.3}},
            verifier="seedopt_fresh_instance", mse=0.0,
            source_file="artifacts/seed_optimizer/seed_optimizer_report.json")
        s2 = self.store.record_state(
            [0.5, 1.0, 2.0], {"after": {"combined": 0.4}},
            verifier="seedopt_fresh_instance", mse=1e-9,
            source_file="artifacts/seed_optimizer/seed_optimizer_report.json")
        self.assertEqual(s1, s2)  # same row, updated in place
        states = self.store.list_states()
        self.assertEqual(len(states), 1)
        self.assertEqual(json.loads(states[0]["metrics"])["after"]["combined"], 0.4)

    def test_record_experiment_kinds(self):
        for kind in EXPERIMENT_KINDS:
            self.assertTrue(self.store.record_experiment(
                f"exp-{kind}", kind, {"verdict": "ok"}))
        self.assertEqual(len(self.store.list_experiments()), 3)

    def test_record_experiment_invalid_kind(self):
        with self.assertRaises(ValueError):
            self.store.record_experiment("bad", "not-a-kind", {})

    def test_record_experiment_upsert(self):
        self.assertTrue(self.store.record_experiment(
            "exp-1", "validation", {"v": 1}, artifact_path="a/b.json"))
        self.assertTrue(self.store.record_experiment(
            "exp-1", "validation", {"v": 2}, artifact_path="a/b.json"))
        experiments = self.store.list_experiments()
        self.assertEqual(len(experiments), 1)
        self.assertEqual(json.loads(experiments[0]["verdict"]), {"v": 2})

    def test_sync_seedopt_single_file(self):
        report = {
            "objective": "combined",
            "optimized_seed": [0.60944, 3.19825, 0.82779],
            "after": {"combined": 0.2794},
            "verified_fresh_instance": {"combined": 0.2794},
            "improvement": {"combined": 0.22},
            "improvement_pct": {"combined": 467.0},
            "params": {"n_effective_qubits": 128, "weights": {"x": 0.5}},
            "reconstruction": {"reconstruction_mse": 0.0},
            "verify_ok": True,
        }
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "seed_optimizer_report.json"
            p.write_text(json.dumps(report))
            ingested = self.store.sync_seedopt(td)
        self.assertEqual(len(ingested), 1)
        states = self.store.list_states()
        self.assertEqual(len(states), 1)
        self.assertEqual(json.loads(states[0]["seed_triple"]),
                         [0.60944, 3.19825, 0.82779])
        runs = self.store.list_runs()
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["source"], "seedopt_sync")
        self.assertEqual(states[0]["source_run"], runs[0]["run_id"])

    def test_sync_seedopt_directory_multi(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            (base / "a").mkdir()
            (base / "b").mkdir()
            for i, sub in enumerate(["a", "b"]):
                (base / sub / "seed_optimizer_report.json").write_text(json.dumps({
                    "objective": "combined",
                    "optimized_seed": [0.1 + i, 1.0, 2.0],
                    "after": {"combined": 0.1 + i},
                    "verified_fresh_instance": {"combined": 0.1 + i},
                    "params": {"n_effective_qubits": 128},
                    "reconstruction": {"reconstruction_mse": 0.0},
                }))
            ingested = self.store.sync_seedopt(td)
        self.assertEqual(len(ingested), 2)
        self.assertEqual(len(self.store.list_states()), 2)
        self.assertEqual(len(self.store.list_runs()), 2)

    def test_sync_seedopt_bad_json_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "seed_optimizer_report.json"
            p.write_text("{not json")
            ingested = self.store.sync_seedopt(td)
        self.assertEqual(ingested, [])
        self.assertEqual(self.store.counts()["states"], 0)

    def test_unsupported_scheme_raises(self):
        bad = DbStore("mysql://user:pass@host/db")
        with self.assertRaises(ValueError):
            bad.record_run([0.5, 1.0, 2.0], "combined", strict=True)


if __name__ == "__main__":
    unittest.main()