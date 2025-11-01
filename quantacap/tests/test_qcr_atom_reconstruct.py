import os
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from quantacap.experiments.qcr_atom_reconstruct import (
    make_grid,
    seed_qubit_weights,
    density_from_qubits,
    qcr_iterate,
    save_isosurface_mask,
    save_constant_json,
    run_qcr_atom,
)


def test_make_grid():
    X, Y, Z = make_grid(R=1.0, N=8)
    assert X.shape == (8, 8, 8)
    assert Y.shape == (8, 8, 8)
    assert Z.shape == (8, 8, 8)
    assert X.min() == pytest.approx(-1.0)
    assert X.max() == pytest.approx(1.0)


def test_seed_qubit_weights():
    weights = seed_qubit_weights(n_qubits=4, seed=42)
    assert len(weights) == 4
    assert all(-1.0 <= w <= 1.0 for w in weights)


def test_density_from_qubits():
    X, Y, Z = make_grid(R=1.0, N=8)
    zexp = [0.5, -0.5, 0.0, 0.3]
    density = density_from_qubits(zexp, X, Y, Z)
    assert density.shape == (8, 8, 8)
    assert density.min() >= 0.0
    assert density.max() <= 1.0


def test_qcr_iterate():
    X, Y, Z = make_grid(R=1.0, N=8)
    zexp = [0.5, -0.5, 0.0, 0.3]
    density, conv_hist, frames = qcr_iterate(X, Y, Z, zexp, iters=5, smooth=0.15)
    assert density.shape == (8, 8, 8)
    assert len(conv_hist) > 0
    assert len(conv_hist) <= 5


def test_save_isosurface_mask():
    with tempfile.TemporaryDirectory() as tmpdir:
        density = np.random.random((8, 8, 8))
        out_path = os.path.join(tmpdir, "test_mask.npy")
        mask = save_isosurface_mask(density, iso=0.5, out_path=out_path)
        assert os.path.exists(out_path)
        loaded = np.load(out_path)
        assert loaded.shape == (8, 8, 8)
        assert loaded.dtype == bool


def test_save_constant_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "constant.json")
        conv_hist = [1.0, 0.5, 0.1]
        zexp = [0.5, -0.5]
        const = save_constant_json(
            R=1.0, N=16, iso=0.3, conv_hist=conv_hist, zexp=zexp, out_path=out_path
        )
        assert os.path.exists(out_path)
        with open(out_path) as f:
            loaded = json.load(f)
        assert loaded["name"] == "QCR-ATOM-V1"
        assert loaded["grid"]["R"] == 1.0
        assert loaded["grid"]["N"] == 16
        assert loaded["isosurface"]["iso"] == 0.3
        assert loaded["qubits"]["z_expectations"] == zexp


def test_run_qcr_atom_small():
    with tempfile.TemporaryDirectory() as tmpdir:
        orig_dir = os.getcwd()
        try:
            os.chdir(tmpdir)
            summary = run_qcr_atom(N=8, R=1.0, iters=3, iso=0.35, n_qubits=4, seed=42)
            assert summary["experiment"] == "qcr_atom_reconstruct"
            assert os.path.exists("artifacts/qcr/atom_density.npy")
            assert os.path.exists("artifacts/qcr/atom_isomask.npy")
            assert os.path.exists("artifacts/qcr/atom_constant.json")
            assert os.path.exists("artifacts/qcr/summary.json")
        finally:
            os.chdir(orig_dir)
