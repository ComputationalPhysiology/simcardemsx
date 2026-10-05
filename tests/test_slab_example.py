"""The slab example (``numerical_experiments/strong_coupling_zetasplit/main.py``) with a
crossbridge contraction model: its output, its restart, and its ``post.py``.

The example runs as a subprocess from its own directory, as it is meant to be run: its
paths (``../odefiles``, ``generated_odes/``) are relative to that directory, and it sets
the root logger and dolfinx's log to DEBUG for the whole process.

Gate R4 is :func:`test_slab_restart_matches_the_uninterrupted_run`: 4 ms straight
against 2 ms followed by ``--restart`` to 4 ms, compared bit for bit.
"""

import csv
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import io4dolfinx
import numpy as np
import pytest

from simcardemsx.results import RESULTS, read_resolved_settings, read_results

EXAMPLES = Path(__file__).parent.parent / "numerical_experiments"
EXAMPLE_DIR = EXAMPLES / "strong_coupling_zetasplit"
CAISPLIT = "../odefiles/ToRORd_dynCl_endo_caisplit.ode"
#: The crossbridge path every run below takes, at a mechanics step of 1 ms.
CROSSBRIDGE = (
    "--crossbridge",
    "Land2017",
    "--scheme",
    "stabilized",
    "--odefile",
    CAISPLIT,
    "--dt-mech",
    "1",
)
#: What ``post.py`` writes into ``post/``.
POST_FILES = (
    "fields.bp",
    "traces.csv",
    "ep_volume_averages.png",
    "ep_point_traces.png",
    "mech_volume_averages.png",
    "mech_point_traces.png",
    "log.png",
)
POST_DATA_FILES = ("fields.bp", "traces.csv")
RESULT_NAMES = ("v", "cai", "u", "lmbda", "tension_kPa", "stiffness_kPa")
#: Every process here runs with one hash seed. gotranx orders the generated EP module's
#: states by iterating sets of names, so the order depends on ``PYTHONHASHSEED``, and a
#: restart in a process that generated another order is refused ("the same names in
#: another order"). ``test_ode_model.py``'s strict xfail tracks that; until gotranx
#: generates the same code in every process, the restarts below would otherwise fail
#: about one time in two, for that reason alone.
ENV = {**os.environ, "PYTHONHASHSEED": "0"}


def _run_example(*args: str, output_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "main.py", *args, "--output-dir", str(output_dir)],
        cwd=EXAMPLE_DIR,
        capture_output=True,
        text=True,
        timeout=900,
        env=ENV,
    )


def _run_slab(*args: str, output_dir: Path) -> subprocess.CompletedProcess:
    """The crossbridge slab with ``args`` appended (``--t-end``, ``--restart``, ...)."""
    return _run_example(*CROSSBRIDGE, *args, output_dir=output_dir)


def _run_post(output_dir: Path, *python_args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *python_args, "post.py", "--output-dir", str(output_dir)],
        cwd=EXAMPLE_DIR,
        capture_output=True,
        text=True,
        timeout=900,
        env=ENV,
    )


def _post() -> ModuleType:
    """``strong_coupling_zetasplit.post``, imported without running anything."""
    sys.path.insert(0, str(EXAMPLES))
    try:
        return importlib.import_module("strong_coupling_zetasplit.post")
    finally:
        sys.path.remove(str(EXAMPLES))


def _files(folder: Path) -> dict[str, bytes]:
    """Every file under ``folder``, by its path relative to it, with its bytes."""
    return {
        str(path.relative_to(folder)): path.read_bytes()
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


def _raw_times(folder: Path, name: str) -> list[float]:
    """Every time ``results.bp`` holds ``name`` at, duplicates included."""
    from mpi4py import MPI

    times = io4dolfinx.read_timestamps(
        filename=folder / RESULTS,
        comm=MPI.COMM_WORLD,
        function_name=name,
    )
    return [float(t) for t in times]


def _results(folder: Path) -> dict[str, dict[float, np.ndarray]]:
    settings = read_resolved_settings(folder / "config.resolved.toml")
    return read_results(folder, _post().result_functions(settings))


@pytest.mark.slow
def test_slab_example_runs_a_crossbridge_model(tmp_path):
    """``--crossbridge Land2017`` on the Ca_i split, stabilized, reaches t_end, records the
    crossbridge backend in ``run.json``, and contracts: the Recorder's ``steps.csv`` has a
    row per mechanics step, and the tension is positive by the last one."""
    output_dir = tmp_path / "out"
    result = _run_slab("--t-end", "2", output_dir=output_dir)
    assert result.returncode == 0, result.stderr[-3000:]

    run = json.loads((output_dir / "run.json").read_text())
    assert run["reached_t_end"] is True
    assert run["failure"] is None
    assert run["backend"] == "crossbridge:Land2017"
    assert run["split"] == "caisplit"
    assert run["scheme"] == "stabilized"
    assert run["restart"] is False
    assert len(run["history"]) == 1

    with (output_dir / "steps.csv").open() as f:
        steps = list(csv.DictReader(f))
    assert [float(step["t_ms"]) for step in steps] == [0.0, 1.0, 2.0]
    assert float(steps[-1]["Ta_mean_kPa"]) > 0.0

    # One log.csv row per mechanics step and one at t = 0, and the CLIs' files.
    with (output_dir / "log.csv").open() as f:
        log = list(csv.reader(f))
    assert log[0] == ["t_ms", "newton_iterations", "lmbda_mean", "Ta_mean_kPa"]
    assert [float(row[0]) for row in log[1:]] == [0.0, 1.0, 2.0]
    for name in ("config.resolved.toml", RESULTS, "restart.bp", "restart.json", "timings.json"):
        assert (output_dir / name).exists(), name
    for name in RESULT_NAMES:
        assert _raw_times(output_dir, name) == [0.0, 1.0, 2.0], name
    # DataCollector's files are gone.
    assert not (output_dir / "lmbda_prev_mean.txt").exists()
    assert not (output_dir / "disp.bp").exists()


def test_slab_example_refuses_monolithic_with_crossbridge(tmp_path):
    """``--crossbridge`` with the default ``--scheme monolithic`` exits with the reason
    before anything is built or written."""
    output_dir = tmp_path / "out"
    result = _run_example("--crossbridge", "Land2017", "--odefile", CAISPLIT, output_dir=output_dir)

    assert result.returncode != 0
    assert "--scheme monolithic is not available with --crossbridge" in result.stderr
    assert not output_dir.exists()


@pytest.mark.slow
def test_slab_restart_matches_the_uninterrupted_run(tmp_path):
    """Gate R4: 4 ms straight, and 2 ms followed by ``--restart`` to 4 ms, give the same
    ``results.bp`` (every name, every time, bit for bit, and no time twice), the same
    ``log.csv`` and ``steps.csv`` byte for byte, and the same ``post/traces.csv``."""
    straight, restarted = tmp_path / "a", tmp_path / "b"
    result = _run_slab("--t-end", "4", output_dir=straight)
    assert result.returncode == 0, result.stderr[-3000:]
    result = _run_slab("--t-end", "2", output_dir=restarted)
    assert result.returncode == 0, result.stderr[-3000:]
    result = _run_slab("--t-end", "4", "--restart", output_dir=restarted)
    assert result.returncode == 0, result.stderr[-3000:]

    for name in RESULT_NAMES:
        assert _raw_times(restarted, name) == _raw_times(straight, name), name
    a, b = _results(straight), _results(restarted)
    assert sorted(a) == sorted(b) == sorted(RESULT_NAMES)
    for name in RESULT_NAMES:
        assert list(a[name]) == list(b[name]), name
        for t in a[name]:
            assert np.array_equal(a[name][t], b[name][t]), (name, t)
    assert a["u"][4.0].any(), "the slab did not move: the comparison is vacuous"

    for name in ("log.csv", "steps.csv"):
        assert (straight / name).read_bytes() == (restarted / name).read_bytes(), name

    run = json.loads((restarted / "run.json").read_text())
    assert run["restart"] is True
    assert run["reached_t_end"] is True
    assert len(run["history"]) == 2

    post = _post()
    post.main(["--output-dir", str(straight)])
    post.main(["--output-dir", str(restarted)])
    assert (straight / "post" / "traces.csv").read_bytes() == (
        restarted / "post" / "traces.csv"
    ).read_bytes()


@pytest.mark.slow
def test_slab_post_regenerates_the_plots(tmp_path):
    """After a 2 ms run, ``post.py`` exits 0 and writes every file of ``post/``, with a
    ``traces.csv`` row per saved time. Without matplotlib it still writes the data files
    and skips the plots with a warning."""
    output_dir = tmp_path / "out"
    result = _run_slab("--t-end", "2", output_dir=output_dir)
    assert result.returncode == 0, result.stderr[-3000:]

    # matplotlib made unimportable for the whole process.
    no_matplotlib = "import sys; sys.modules['matplotlib'] = None; import runpy; " + (
        "sys.argv = sys.argv[1:]; runpy.run_path('post.py', run_name='__main__')"
    )
    result = _run_post(output_dir, "-c", no_matplotlib)
    assert result.returncode == 0, result.stderr[-3000:]
    assert "the plots are skipped" in result.stderr
    for name in POST_DATA_FILES:
        assert (output_dir / "post" / name).exists(), name
    assert not list((output_dir / "post").glob("*.png"))

    result = _run_post(output_dir)
    assert result.returncode == 0, result.stderr[-3000:]
    for name in POST_FILES:
        assert (output_dir / "post" / name).exists(), name

    with (output_dir / "post" / "traces.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert [float(row["t_ms"]) for row in rows] == [0.0, 1.0, 2.0]
    with (output_dir / "log.csv").open() as f:
        log = list(csv.DictReader(f))
    # The volume averages are the run's own mesh means.
    for row, logged in zip(rows, log):
        assert float(row["lmbda_mean"]) == pytest.approx(float(logged["lmbda_mean"]), rel=1e-12)
        assert float(row["Ta_mean_kPa"]) == pytest.approx(float(logged["Ta_mean_kPa"]), rel=1e-12)
    assert float(rows[-1]["Ta_mean_kPa"]) > 0.0


@pytest.mark.slow
def test_slab_restart_at_or_past_t_end_does_nothing(tmp_path):
    """``--restart`` with ``--t-end`` at or before the checkpoint exits 0 without
    stepping: ``results.bp``'s times and ``log.csv`` are unchanged."""
    output_dir = tmp_path / "out"
    result = _run_slab("--t-end", "2", output_dir=output_dir)
    assert result.returncode == 0, result.stderr[-3000:]
    times = {name: _raw_times(output_dir, name) for name in RESULT_NAMES}
    log = (output_dir / "log.csv").read_bytes()
    steps = (output_dir / "steps.csv").read_bytes()

    for t_end in ("2", "1"):
        result = _run_slab("--t-end", t_end, "--restart", output_dir=output_dir)
        assert result.returncode == 0, result.stderr[-3000:]
        assert {name: _raw_times(output_dir, name) for name in RESULT_NAMES} == times
        assert (output_dir / "log.csv").read_bytes() == log
        assert (output_dir / "steps.csv").read_bytes() == steps
        run = json.loads((output_dir / "run.json").read_text())
        assert run["restart"] is True
        assert run["failure"] is None


@pytest.mark.slow
def test_slab_restart_refuses_other_physics(tmp_path):
    """``--restart`` with another ``--scheme`` exits non-zero, says the physics differ,
    and changes no file."""
    output_dir = tmp_path / "out"
    result = _run_slab("--t-end", "2", output_dir=output_dir)
    assert result.returncode == 0, result.stderr[-3000:]
    before = _files(output_dir)

    result = _run_example(
        *CROSSBRIDGE,
        "--t-end",
        "4",
        "--restart",
        "--scheme",
        "segregated",
        output_dir=output_dir,
    )
    assert result.returncode != 0
    assert "physics" in result.stderr
    assert _files(output_dir) == before


@pytest.mark.slow
def test_slab_refuses_existing_results_without_flags(tmp_path):
    """A second plain run into a folder with results is refused and changes no file;
    ``--overwrite`` replaces the results and keeps a file of the user's."""
    output_dir = tmp_path / "out"
    result = _run_slab("--t-end", "1", output_dir=output_dir)
    assert result.returncode == 0, result.stderr[-3000:]
    before = _files(output_dir)

    result = _run_slab("--t-end", "1", output_dir=output_dir)
    assert result.returncode != 0
    assert "--overwrite" in result.stderr
    assert _files(output_dir) == before

    (output_dir / "notes.txt").write_text("mine\n")
    result = _run_slab("--t-end", "2", "--overwrite", output_dir=output_dir)
    assert result.returncode == 0, result.stderr[-3000:]
    assert (output_dir / "notes.txt").read_text() == "mine\n"
    assert _raw_times(output_dir, "u") == [0.0, 1.0, 2.0]
    run = json.loads((output_dir / "run.json").read_text())
    assert run["reached_t_end"] is True
    assert len(run["history"]) == 1
