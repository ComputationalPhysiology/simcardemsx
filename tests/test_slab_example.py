"""The slab example (``numerical_experiments/strong_coupling_zetasplit/main.py``) with a
crossbridge contraction model.

The example runs as a subprocess from its own directory, as it is meant to be run: its
paths (``../odefiles``, ``meshes/``, ``generated_odes/``) are relative to that directory,
and it sets the root logger and dolfinx's log to DEBUG for the whole process.
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).parent.parent / "numerical_experiments" / "strong_coupling_zetasplit"
CAISPLIT = "../odefiles/ToRORd_dynCl_endo_caisplit.ode"


def _run_example(*args: str, output_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "main.py", *args, "--output-dir", str(output_dir)],
        cwd=EXAMPLE_DIR,
        capture_output=True,
        text=True,
        timeout=900,
    )


@pytest.mark.slow
def test_slab_example_runs_a_crossbridge_model(tmp_path):
    """``--crossbridge Land2017`` on the Ca_i split, stabilized, reaches t_end, records the
    crossbridge backend in ``run.json``, and contracts: the Recorder's ``steps.csv`` has a
    row per mechanics step, and the tension is positive by the last one."""
    output_dir = tmp_path / "out"
    result = _run_example(
        "--crossbridge",
        "Land2017",
        "--odefile",
        CAISPLIT,
        "--scheme",
        "stabilized",
        "--dt-mech",
        "1",
        "--t-end",
        "2",
        output_dir=output_dir,
    )
    assert result.returncode == 0, result.stderr[-3000:]

    run = json.loads((output_dir / "run.json").read_text())
    assert run["reached_t_end"] is True
    assert run["failure"] is None
    assert run["backend"] == "crossbridge:Land2017"
    assert run["split"] == "caisplit"
    assert run["scheme"] == "stabilized"

    with (output_dir / "steps.csv").open() as f:
        steps = list(csv.DictReader(f))
    assert [float(step["t_ms"]) for step in steps] == [0.0, 1.0, 2.0]
    assert float(steps[-1]["Ta_mean_kPa"]) > 0.0


def test_slab_example_refuses_monolithic_with_crossbridge(tmp_path):
    """``--crossbridge`` with the default ``--scheme monolithic`` exits with the reason
    before anything is built or written."""
    output_dir = tmp_path / "out"
    result = _run_example("--crossbridge", "Land2017", "--odefile", CAISPLIT, output_dir=output_dir)

    assert result.returncode != 0
    assert "--scheme monolithic is not available with --crossbridge" in result.stderr
    assert not output_dir.exists()
