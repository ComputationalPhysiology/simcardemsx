"""The results module: results.bp, the CSV log, the output-folder rules, settings."""

from __future__ import annotations

import json
from pathlib import Path

from mpi4py import MPI

import basix.ufl
import dolfinx
import io4dolfinx
import numpy as np
import pytest

from simcardemsx import checkpoint
from simcardemsx.results import (
    ARTIFACTS,
    RESULTS,
    CsvLog,
    ResultsWriter,
    prepare_output,
    read_resolved_settings,
    read_result_times,
    read_results,
    stride,
    write_resolved_settings,
)

COMM = MPI.COMM_WORLD


def _mesh():
    return dolfinx.mesh.create_unit_square(COMM, 3, 3)


def _p1(mesh):
    return dolfinx.fem.functionspace(mesh, ("Lagrange", 1))


def test_results_writer_never_duplicates_a_timestamp(tmp_path):
    mesh = _mesh()
    f = dolfinx.fem.Function(_p1(mesh))
    writer = ResultsWriter(tmp_path)
    for t in (0.0, 1.0, 1.0):
        f.x.array[:] = t
        writer.write(t, {"u": f})
    again = ResultsWriter(tmp_path)
    again.resume(["u"])
    for t in (1.0, 2.0):
        f.x.array[:] = t
        again.write(t, {"u": f})
    raw = io4dolfinx.read_timestamps(tmp_path / RESULTS, COMM, "u")
    assert list(raw) == [0.0, 1.0, 2.0]


def test_read_results_round_trips_bit_for_bit(tmp_path):
    mesh = _mesh()
    p1 = dolfinx.fem.Function(_p1(mesh))
    q = dolfinx.fem.functionspace(
        mesh,
        basix.ufl.quadrature_element(mesh.basix_cell(), (), degree=2),
    )
    quad = dolfinx.fem.Function(q)
    rng = np.random.default_rng(0)
    writer = ResultsWriter(tmp_path)
    saved = {}
    for t in (0.0, 0.5, 2.0):
        p1.x.array[:] = rng.random(p1.x.array.size)
        quad.x.array[:] = rng.random(quad.x.array.size)
        saved[t] = (p1.x.array.copy(), quad.x.array.copy())
        writer.write(t, {"p1": p1, "quad": quad})
    assert list(read_result_times(tmp_path, "p1")) == [0.0, 0.5, 2.0]
    out = read_results(tmp_path, {"p1": p1, "quad": quad})
    assert sorted(out["p1"]) == [0.0, 0.5, 2.0]
    for t, (a, b) in saved.items():
        np.testing.assert_array_equal(out["p1"][t], a)
        np.testing.assert_array_equal(out["quad"][t], b)


def test_csv_log_round_trips_bit_for_bit(tmp_path):
    log = CsvLog(tmp_path / "log.csv", ["t_ms", "x"])
    log.start()
    rows = [{"t_ms": 0.1 + 0.2, "x": 1 / 3}, {"t_ms": 1.0, "x": -1e-300}]
    for r in rows:
        log.append(r)
    assert log.read() == rows


def _filled(tmp_path):
    log = CsvLog(tmp_path / "log.csv", ["t_ms", "x"])
    log.start()
    for t in range(6):
        log.append({"t_ms": float(t), "x": t * 0.1})
    return log


def test_csv_log_resume_truncates_after_t(tmp_path):
    log = _filled(tmp_path)
    kept = CsvLog(tmp_path / "log.csv", ["t_ms", "x"]).resume(2.0)
    assert len(kept) == 3
    log.append({"t_ms": 3.0, "x": 0.0})
    assert len(log.read()) == 4
    assert not list(tmp_path.glob("*.tmp*"))


def test_csv_log_resume_refuses_another_header(tmp_path):
    _filled(tmp_path)
    with pytest.raises(ValueError, match="header"):
        CsvLog(tmp_path / "log.csv", ["t_ms", "y"]).resume(2.0)


def test_csv_log_first_field_must_be_t_ms(tmp_path):
    with pytest.raises(ValueError):
        CsvLog(tmp_path / "log.csv", ["x", "t_ms"])


def test_stride_refuses_a_non_multiple():
    with pytest.raises(ValueError):
        stride(1.5, 1.0)
    with pytest.raises(ValueError):
        stride(0.0, 1.0)
    with pytest.raises(ValueError):
        stride(-1.0, 1.0)
    assert stride(2.0, 0.05) == 40


def test_prepare_output_creates_an_empty_folder(tmp_path):
    folder = tmp_path / "new"
    assert prepare_output(folder, restart=False, overwrite=False, physics={}) == "create"
    assert folder.is_dir()


def test_prepare_output_refuses_results_without_flags(tmp_path):
    (tmp_path / RESULTS).mkdir()
    with pytest.raises(ValueError, match="--overwrite.*--restart|--restart.*--overwrite"):
        prepare_output(tmp_path, restart=False, overwrite=False, physics={})
    assert (tmp_path / RESULTS).exists()


def test_prepare_output_refuses_a_stale_restart_bp_and_overwrite_removes_it(tmp_path):
    (tmp_path / checkpoint.RESTART).mkdir()
    assert checkpoint.RESTART in ARTIFACTS
    with pytest.raises(ValueError):
        prepare_output(tmp_path, restart=False, overwrite=False, physics={})
    assert prepare_output(tmp_path, restart=False, overwrite=True, physics={}) == "wipe"
    assert not (tmp_path / checkpoint.RESTART).exists()


def test_prepare_output_overwrite_keeps_other_files(tmp_path):
    (tmp_path / RESULTS).mkdir()
    (tmp_path / "log.csv").write_text("t_ms\n")
    (tmp_path / "notes.txt").write_text("keep")
    assert prepare_output(tmp_path, restart=False, overwrite=True, physics={}) == "wipe"
    assert (tmp_path / "notes.txt").read_text() == "keep"
    assert not (tmp_path / RESULTS).exists()
    assert not (tmp_path / "log.csv").exists()


def test_prepare_output_glob_artifacts_are_detected_and_wiped(tmp_path):
    artifacts = (*ARTIFACTS, "restart_recorder_*.npz")
    (tmp_path / "restart_recorder_1.5.npz").write_text("x")
    (tmp_path / "keep.npz").write_text("x")
    with pytest.raises(ValueError):
        prepare_output(tmp_path, restart=False, overwrite=False, physics={}, artifacts=artifacts)
    prepare_output(tmp_path, restart=False, overwrite=True, physics={}, artifacts=artifacts)
    assert not (tmp_path / "restart_recorder_1.5.npz").exists()
    assert (tmp_path / "keep.npz").exists()


def test_prepare_output_restart_and_overwrite_conflict(tmp_path):
    with pytest.raises(ValueError):
        prepare_output(tmp_path, restart=True, overwrite=True, physics={})


def test_prepare_output_restart_checks_the_physics(tmp_path):
    with pytest.raises(FileNotFoundError):
        prepare_output(tmp_path, restart=True, overwrite=False, physics={"a": 1})
    meta = {checkpoint.NAMESPACE: {"physics_hash": checkpoint.physics_hash({"a": 1})}}
    (tmp_path / checkpoint.RESTART_META).write_text(json.dumps(meta))
    assert prepare_output(tmp_path, restart=True, overwrite=False, physics={"a": 1}) == "restart"
    with pytest.raises(ValueError):
        prepare_output(tmp_path, restart=True, overwrite=False, physics={"a": 2})


def test_resolved_settings_round_trip(tmp_path):
    settings = {"a": 1, "path": Path("/x/y"), "nested": {"p": Path("z"), "v": [1.5, 2.5]}}
    write_resolved_settings(tmp_path / "config.resolved.toml", settings)
    assert read_resolved_settings(tmp_path / "config.resolved.toml") == {
        "a": 1,
        "path": "/x/y",
        "nested": {"p": "z", "v": [1.5, 2.5]},
    }


def test_prepare_output_corrupt_restart_json_raises_value_error(tmp_path):
    (tmp_path / checkpoint.RESTART_META).write_text("{not json")
    (tmp_path / "keep.txt").write_text("x")
    before = sorted(p.name for p in tmp_path.iterdir())
    with pytest.raises(ValueError):
        prepare_output(tmp_path, restart=True, overwrite=False, physics={})
    assert sorted(p.name for p in tmp_path.iterdir()) == before


def test_results_writer_resume_skips_a_name_without_times(tmp_path):
    mesh = _mesh()
    f = dolfinx.fem.Function(_p1(mesh))
    ResultsWriter(tmp_path).write(1.0, {"u": f})
    writer = ResultsWriter(tmp_path)
    writer.resume(["u", "never_written"])
    assert writer._last == {"u": 1.0}
    ResultsWriter(tmp_path / "empty").resume(["u"])  # no results.bp: sets nothing


def test_results_writer_resume_propagates_a_real_read_failure(tmp_path):
    (tmp_path / RESULTS).write_text("not a bp folder")
    with pytest.raises(Exception):  # noqa: B017, PT011
        ResultsWriter(tmp_path).resume(["u"])
