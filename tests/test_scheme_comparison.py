"""The scheme comparison's shared pieces: ``metrics.py`` and ``record.py``."""

import csv
import importlib
import json
import sys
from pathlib import Path

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
from conftest import calcium

from simcardemsx.backends import GeneratedActivation

ROOT = Path(__file__).parent.parent
EXAMPLES = ROOT / "numerical_experiments"


@pytest.fixture(scope="module")
def modules():
    sys.path.insert(0, str(EXAMPLES))
    try:
        return (
            importlib.import_module("scheme_comparison.metrics"),
            importlib.import_module("scheme_comparison.record"),
        )
    finally:
        sys.path.remove(str(EXAMPLES))


@pytest.fixture(scope="module")
def metrics(modules):
    return modules[0]


@pytest.fixture(scope="module")
def record(modules):
    return modules[1]


def test_weighted_rms_and_mean(metrics):
    x = np.array([1.0, 3.0])
    w = np.array([1.0, 3.0])
    assert metrics.weighted_mean(x, w) == pytest.approx(2.5)
    assert metrics.weighted_rms(x, w) == pytest.approx(np.sqrt((1 + 27) / 4))


def test_reversal_fraction_counts_sign_changes_above_the_floor(metrics):
    w = np.ones(4)
    d = np.array([1.0, -1.0, 1.0, -1.0])
    assert metrics.reversal_fraction(d, -d, w, 0.0) == 1.0
    assert metrics.reversal_fraction(d, d, w, 0.0) == 0.0
    assert metrics.reversal_fraction(d, -d, w, 2.0) == 0.0
    # Unequal weights: only the first point reverses, and it carries 1/4 of the weight.
    d_prev = np.array([1.0, 1.0])
    d_curr = np.array([-1.0, 1.0])
    assert metrics.reversal_fraction(d_prev, d_curr, np.array([1.0, 3.0]), 0.0) == 0.25


def test_onset_ignores_a_single_turning_step(metrics):
    assert metrics.onset_index([0, 0, 1, 0, 0]) is None
    assert metrics.onset_index([0, 1, 1, 1, 0]) == 1


def test_self_convergence_order_recovers_the_order(metrics):
    for p in (1.0, 2.0):
        dts = (1.0, 0.25, 0.1)
        X = [3 * d**p for d in dts]
        got = metrics.self_convergence_order(dts, abs(X[0] - X[1]), abs(X[1] - X[2]))
        assert got == pytest.approx(p, abs=1e-8)
    got = metrics.self_convergence_order((2.0, 1.0, 0.5), 6.0, 1.5)
    assert got == pytest.approx(np.log2(4.0), abs=1e-8)


def test_reference_orders_and_richardson(metrics):
    X_star, C = 2.0, 5.0
    dts = [1.0, 0.5, 0.25]
    errors = [C * d for d in dts]
    np.testing.assert_allclose(metrics.reference_orders(dts, errors), [1.0, 1.0], atol=1e-12)
    coarse = np.array([X_star + C * 1.0, 1.0 + C])
    fine = np.array([X_star + C * 0.5, 1.0 + C * 0.5])
    got = metrics.richardson(fine, coarse, 2.0)
    np.testing.assert_allclose(got, [X_star, 1.0], atol=1e-12)


def test_naive_threshold_matches_d2(metrics):
    for dt, want in ((2.0, 75.0), (1.0, 200.0), (0.5, 600.0)):
        got = metrics.naive_threshold_kPa(dt, Kp_kPa=0.0, eta_Pa_s=100.0, rho_kg_m3=1e3, h_m=0.01)
        assert got == pytest.approx(want)


REGIME = {"Kp_kPa": 1.0, "eta_Pa_s": 100.0, "rho_kg_m3": 1e3, "h_m": 0.01}


def _run_info(t_end: float) -> dict:
    return {
        "geometry": "cube",
        "split": "caisplit",
        "scheme": "monolithic",
        "dt_mech_ms": 0.5,
        "t_end_ms": t_end,
        "regime": REGIME,
    }


def _backend(split_modules, **kw):
    from test_generated_activation import _f0, _prescribed

    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    u = _prescribed(mesh, 1.0)
    backend = GeneratedActivation(
        split_modules["caisplit"].mechanics,
        mesh,
        _f0(mesh),
        quadrature_degree=2,
        **kw,
    )
    backend.register(u)
    return backend, u


def _advance(backend, u, n: int, dt: float, rec) -> None:
    from test_generated_activation import _set_stretch

    for k in range(n):
        t = (k + 1) * dt
        backend.t.value = k * dt
        backend.dt.value = dt
        backend.inputs["cai"].x.array[:] = calcium(t)
        _set_stretch(u, 1.0 - 0.01 * (k + 1))
        backend.post_solve()
        rec.step(t, 2)


def test_recorder_writes_steps_snapshots_and_run_info(record, metrics, split_modules, tmp_path):
    backend, u = _backend(split_modules)
    rec = record.Recorder(backend, tmp_path, run_info=_run_info(2.0), snapshot_every_ms=1.0)
    _advance(backend, u, 4, 0.5, rec)
    rec.finish(failure=None, t_fail_ms=None, timings={"mech_s": 1.0})

    with open(tmp_path / "steps.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 5
    assert [float(r["t_ms"]) for r in rows] == pytest.approx([0, 0.5, 1, 1.5, 2])
    assert list(rows[0])[:8] == [
        "t_ms",
        "newton_iterations",
        "lmbda_mean",
        "lmbda_min",
        "lmbda_max",
        "Ta_mean_kPa",
        "Ka_max_kPa",
        "KaDl_max_kPa",
    ]
    assert "reversal_fraction_1e-06" in rows[0]
    assert "reversal_fraction_0" in rows[0]
    assert float(rows[0]["KaDl_max_kPa"]) == 0.0
    assert float(rows[2]["lmbda_mean"]) == pytest.approx(0.98)
    # Monotone shortening: no reversals.
    assert all(float(r["reversal_fraction_0"]) == 0.0 for r in rows)

    snaps = np.load(tmp_path / "snapshots.npz")
    np.testing.assert_allclose(snaps["t_ms"], [0.0, 1.0, 2.0])
    assert snaps["weights"].sum() == pytest.approx(1.0)
    for name in ("lmbda", "tension_kPa", "stiffness_kPa"):
        assert snaps[name].shape == (3, snaps["weights"].size)
    np.testing.assert_allclose(snaps["lmbda"][2], 0.96)

    info = json.loads((tmp_path / "run.json").read_text())
    assert info["reached_t_end"] is True
    assert info["failure"] is None
    assert info["newton"]["total"] == 8
    assert info["newton"]["steps"] == 4
    assert info["newton"]["max"] == 2
    assert info["timings"] == {"mech_s": 1.0}
    assert info["regime"] == REGIME
    assert "utc" in info and "git_commit" in info


def test_recorder_writes_a_failed_run(record, split_modules, tmp_path):
    backend, u = _backend(split_modules)
    rec = record.Recorder(backend, tmp_path, run_info=_run_info(4.0), snapshot_every_ms=None)
    _advance(backend, u, 2, 0.5, rec)
    rec.finish(failure="RuntimeError('x')", t_fail_ms=1.5, timings={})
    assert (tmp_path / "steps.csv").exists()
    assert not (tmp_path / "snapshots.npz").exists()
    info = json.loads((tmp_path / "run.json").read_text())
    assert info["reached_t_end"] is False
    assert info["failure"] == "RuntimeError('x')"
    assert info["t_fail_ms"] == 1.5


def test_recorder_refuses_a_non_quadrature_backend(record, split_modules, tmp_path):
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    f0 = dolfinx.fem.Constant(mesh, np.array([1.0, 0.0, 0.0]))
    backend = GeneratedActivation(
        split_modules["caisplit"].mechanics,
        mesh,
        f0,
        quadrature_degree=2,
        element=("DG", 1),
    )
    with pytest.raises(ValueError, match="quadrature"):
        record.Recorder(backend, tmp_path, run_info=_run_info(1.0))
