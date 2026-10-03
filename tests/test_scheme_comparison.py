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
    # "Above 0.5" is strict.
    assert metrics.onset_index([0.5, 0.5, 0.5]) is None


def test_self_convergence_order_recovers_the_order(metrics):
    for p in (1.0, 2.0):
        dts = (1.0, 0.25, 0.1)
        X = [3 * d**p for d in dts]
        got = metrics.self_convergence_order(dts, abs(X[0] - X[1]), abs(X[1] - X[2]))
        assert got == pytest.approx(p, abs=1e-8)
    got = metrics.self_convergence_order((2.0, 1.0, 0.5), 6.0, 1.5)
    assert got == pytest.approx(np.log2(4.0), abs=1e-8)


def test_self_convergence_order_is_nan_without_an_order(metrics):
    dts = (1.0, 0.5, 0.25)
    assert np.isnan(metrics.self_convergence_order(dts, 1.0, 0.0))
    assert np.isnan(metrics.self_convergence_order(dts, float("nan"), 1.0))
    assert np.isnan(metrics.self_convergence_order(dts, 1.0, float("inf")))
    # A ratio of 1e6 needs an order far above 5.
    assert np.isnan(metrics.self_convergence_order(dts, 1e6, 1.0))


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
        "regime": dict(REGIME),
    }


def _backend(split_modules, **kw):
    from conftest import _f0
    from test_generated_activation import _prescribed

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
    # A shortening step with active stiffness: a magnitude, so positive.
    assert max(float(r["Ka_max_kPa"]) for r in rows) > 0
    assert all(float(r["KaDl_max_kPa"]) >= 0 for r in rows)
    assert any(float(r["KaDl_max_kPa"]) > 0 for r in rows[1:])
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


def test_recorder_reports_reversals(record, split_modules, tmp_path):
    """Stretch alternating 1 -> 1 - a -> 1 ..., a = 1e-5, dt = 0.5 ms.

    |dλ|/dt = 2e-5 per ms: above the floors 0, 1e-7, 1e-6, 1e-5 and below 1e-4.
    """
    from test_generated_activation import _set_stretch

    a, dt = 1e-5, 0.5
    backend, u = _backend(split_modules)
    rec = record.Recorder(backend, tmp_path, run_info=_run_info(3.0))
    for k in range(6):
        backend.t.value = k * dt
        backend.dt.value = dt
        backend.inputs["cai"].x.array[:] = calcium((k + 1) * dt)
        _set_stretch(u, 1.0 - a * ((k + 1) % 2))
        backend.post_solve()
        rec.step((k + 1) * dt, 2)
    rec.finish(failure=None, t_fail_ms=None, timings={})
    with open(tmp_path / "steps.csv") as f:
        rows = list(csv.DictReader(f))
    # Row 0 is the initial state and row 1 the first step, which has no previous increment.
    assert float(rows[1]["reversal_fraction_0"]) == 0.0
    for r in rows[2:]:
        assert float(r["reversal_fraction_0"]) == 1.0
        assert float(r["reversal_fraction_1e-05"]) == 1.0
        assert float(r["reversal_fraction_0.0001"]) == 0.0


def test_recorder_refuses_incomplete_run_info(record, split_modules, tmp_path):
    backend, _ = _backend(split_modules)
    info = _run_info(1.0)
    del info["geometry"], info["regime"]["h_m"], info["regime"]["Kp_kPa"]
    with pytest.raises(ValueError, match=r"geometry.*regime\.Kp_kPa.*regime\.h_m"):
        record.Recorder(backend, tmp_path, run_info=info)
    info = _run_info(1.0)
    del info["regime"]
    with pytest.raises(ValueError, match="regime"):
        record.Recorder(backend, tmp_path, run_info=info)


def test_recorder_records_the_commit_at_start(record, split_modules, tmp_path, monkeypatch):
    """A run's provenance is the code it started with, not the code at its end."""
    backend, u = _backend(split_modules)
    monkeypatch.setattr(record, "_git_commit", lambda: "commit-at-start")
    monkeypatch.setattr(record, "_git_dirty", lambda: False)
    rec = record.Recorder(backend, tmp_path, run_info=_run_info(1.0))
    monkeypatch.setattr(record, "_git_commit", lambda: "commit-at-finish")
    monkeypatch.setattr(record, "_git_dirty", lambda: True)
    _advance(backend, u, 2, 0.5, rec)
    rec.finish(failure=None, t_fail_ms=None, timings={})
    info = json.loads((tmp_path / "run.json").read_text())
    assert info["git_commit"] == "commit-at-start"
    assert info["git_dirty"] is False


def test_git_provenance_of_a_scratch_repository(record, tmp_path, monkeypatch):
    """``_git_dirty`` looks at the tracked files only; both are None outside a repository."""
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> None:
        subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
            cwd=repo,
            check=True,
            capture_output=True,
        )

    git("init", "-q")
    (repo / "tracked.txt").write_text("a\n")
    git("add", "tracked.txt")
    git("commit", "-q", "-m", "initial")
    sha = record._git_commit(repo)
    assert sha is not None and len(sha) == 40
    assert record._git_dirty(repo) is False
    (repo / "untracked.txt").write_text("b\n")
    assert record._git_dirty(repo) is False
    (repo / "tracked.txt").write_text("changed\n")
    assert record._git_dirty(repo) is True

    outside = tmp_path / "not-a-repo"
    outside.mkdir()
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    assert record._git_commit(outside) is None
    assert record._git_dirty(outside) is None


def test_finish_after_artifacts_writes_run_json_last(record, split_modules, tmp_path):
    """``run.json`` marks a finished run, so it comes after the example's own files.

    An artifact that fails does not stop the others or ``run.json``; with no exception
    from the loop, its error is raised after ``run.json`` is written.
    """
    backend, u = _backend(split_modules)
    rec = record.Recorder(backend, tmp_path, run_info=_run_info(1.0))
    _advance(backend, u, 2, 0.5, rec)
    written = []

    def artifact(name: str):
        def write() -> None:
            assert not (tmp_path / "run.json").exists()
            (tmp_path / name).write_text(name)
            written.append(name)

        return write

    def broken() -> None:
        raise OSError("disk full")

    artifacts = [("a.txt", artifact("a.txt")), ("broken", broken), ("b.txt", artifact("b.txt"))]
    with pytest.raises(OSError, match="disk full"):
        record.finish_after_artifacts(
            rec,
            artifacts,
            failure=None,
            t_fail_ms=None,
            timings={},
        )
    assert written == ["a.txt", "b.txt"]
    info = json.loads((tmp_path / "run.json").read_text())
    assert info["reached_t_end"] is True


def test_finish_after_artifacts_keeps_the_loop_exception(record, split_modules, tmp_path):
    """When the loop raised, that exception propagates, not an artifact's."""
    backend, u = _backend(split_modules)
    rec = record.Recorder(backend, tmp_path, run_info=_run_info(2.0))
    _advance(backend, u, 2, 0.5, rec)

    def broken() -> None:
        raise OSError("disk full")

    failure = None
    with pytest.raises(RuntimeError, match="Newton failed"):
        try:
            raise RuntimeError("Newton failed")
        except BaseException as exc:
            failure = repr(exc)
            raise
        finally:
            record.finish_after_artifacts(
                rec,
                [("broken", broken)],
                failure=failure,
                t_fail_ms=1.5,
                timings={},
            )
    info = json.loads((tmp_path / "run.json").read_text())
    assert info["failure"] == "RuntimeError('Newton failed')"
    assert info["reached_t_end"] is False


def test_pending_runs_skips_completed_runs(tmp_path):
    sys.path.insert(0, str(EXAMPLES))
    try:
        run_module = importlib.import_module("scheme_comparison.run")
    finally:
        sys.path.remove(str(EXAMPLES))
    assert len(run_module.MATRIX["slab"]) == 30
    assert len(run_module.MATRIX["biv"]) == 12
    assert run_module.MATRIX["slab"][0].scheme == "monolithic"
    assert run_module.MATRIX["biv"][0].scheme == "monolithic"
    assert run_module.BATCH_ORDER == ("monolithic", "stabilized", "segregated")

    runs = run_module.MATRIX["slab"][:8]
    assert len(set(runs)) == len(runs)
    done, partial, failed, interrupted, exited, truncated, done_empty, in_solve = runs
    run_json = {
        done: {"failure": None, "reached_t_end": True},
        # A failed run is a result (the naive scheme is expected to fail): done.
        failed: {"failure": "RuntimeError('Newton failed')", "reached_t_end": False},
        # An interrupted run is not: it is run again.
        interrupted: {"failure": "KeyboardInterrupt()", "reached_t_end": False},
        exited: {"failure": "SystemExit(1)", "reached_t_end": False},
        done_empty: {},
        # An interrupt inside the SNES solve, as petsc4py re-raises it.
        in_solve: {"failure": "Error(101) <- KeyboardInterrupt()", "reached_t_end": False},
    }
    for run, info in run_json.items():
        run.outdir(tmp_path).mkdir(parents=True)
        (run.outdir(tmp_path) / "run.json").write_text(json.dumps(info))
    partial.outdir(tmp_path).mkdir(parents=True)
    (partial.outdir(tmp_path) / "stdout.log").write_text("")
    # A run.json cut short (the process died while writing it) is not a finished run.
    truncated.outdir(tmp_path).mkdir(parents=True)
    (truncated.outdir(tmp_path) / "run.json").write_text('{"failure": nu')

    pending = run_module.pending_runs(tmp_path, runs)
    assert pending == [partial, interrupted, exited, truncated, in_solve]


def test_failure_of_keeps_the_interrupt_behind_a_solver_error(record):
    """petsc4py turns a ``KeyboardInterrupt`` in a callback into ``PETSc.Error(101)``,
    with the interrupt as its ``__cause__``; the recorded failure must still show it."""

    def solve() -> None:
        try:
            raise KeyboardInterrupt
        except KeyboardInterrupt as exc:
            raise RuntimeError("error code 101") from exc

    with pytest.raises(RuntimeError) as info:
        solve()
    assert record.failure_of(info.value) == "RuntimeError('error code 101') <- KeyboardInterrupt()"
    assert record.failure_of(ValueError("x")) == "ValueError('x')"
