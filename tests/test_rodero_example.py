"""The ``rodero_05`` example (``numerical_experiments/rodero_05``) without its case data:
what its physics hash holds, its refusals before anything is built, and its summary.

Importing ``rodero_05.main`` and ``rodero_05.post`` builds nothing, and none of these
tests reads physcardems' case: the physics hash reads only the case's file names and
contents, so a small fake case directory with the same layout stands in for it, and the
refusals are checked with the builders replaced by ones that fail if called.
"""

import dataclasses
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest

from simcardemsx.checkpoint import physics_hash
from simcardemsx.results import CsvLog

EXAMPLES = Path(__file__).parent.parent / "numerical_experiments"
sys.path.insert(0, str(EXAMPLES))
try:
    rodero = importlib.import_module("rodero_05.main")
    post = importlib.import_module("rodero_05.post")
    case_module = importlib.import_module("rodero_05.case")
finally:
    sys.path.remove(str(EXAMPLES))

#: The options that define the physics; every other option must be in ``NOT_PHYSICS``.
PHYSICS_OPTIONS = {"case_dir", "tref"}


def _fake_case(folder: Path) -> Path:
    """A directory holding a file at every path ``case.required_files`` names (a
    ``.bp`` directory with one file in it), with no real data."""
    (folder / "steady_state_pcl800_fake").mkdir(parents=True)
    for path in case_module.required_files(folder):
        if path.suffix == ".bp":
            path.mkdir(parents=True)
            (path / "data.0").write_bytes(b"geometry")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")
    return folder


@pytest.fixture
def fake_case(tmp_path) -> Path:
    return _fake_case(tmp_path / "case")


def _physics(fake_case: Path, *argv: str) -> dict:
    return rodero.physics(rodero.parse_args(["--case-dir", str(fake_case), *argv]))


def test_every_option_is_physics_or_not(fake_case):
    """Each option ``parse_args`` defines is either one of :data:`PHYSICS_OPTIONS`, and in
    the physics, or named in ``NOT_PHYSICS``, and not in them. ``physics`` keeps every
    option ``NOT_PHYSICS`` does not name, so an option added without a decision lands in
    the hash: this test fails until it is placed here or in ``NOT_PHYSICS``."""
    options = set(vars(rodero.parse_args([])))
    physics = _physics(fake_case)
    not_physics = set(rodero.NOT_PHYSICS)

    assert not_physics <= options, f"NOT_PHYSICS names no option: {sorted(not_physics - options)}"
    assert not (not_physics & set(physics)), sorted(not_physics & set(physics))
    assert options & set(physics) == PHYSICS_OPTIONS, (
        f"options in the physics: {sorted(options & set(physics))}; an output option belongs "
        "in NOT_PHYSICS, a physics option in PHYSICS_OPTIONS"
    )
    unplaced = options - not_physics - PHYSICS_OPTIONS
    assert not unplaced, f"options neither physics nor in NOT_PHYSICS: {sorted(unplaced)}"


def test_physics_leaves_out_the_solver_options(fake_case):
    """physcardems' solver options are in the resolved settings, not in the physics: a
    restart may change them, as with the CLIs."""
    args = rodero.parse_args(["--case-dir", str(fake_case)])
    settings = rodero.settings(args)
    assert settings["solver"]["preconditioner_lag"] == rodero.PRECONDITIONER_LAG
    assert settings["solver"]["petsc_options"]["ksp_type"] == "gmres"
    assert "solver" not in rodero.physics(args)


def test_tref_changes_the_hash(fake_case):
    assert physics_hash(_physics(fake_case, "--tref", "5")) != physics_hash(_physics(fake_case))
    # Land's Tref is 120 kPa times --tref, in the hashed Land parameters.
    assert _physics(fake_case, "--tref", "5")["land"]["Tref"] == pytest.approx(600.0)


def test_the_case_directory_changes_the_hash(fake_case, tmp_path):
    """Another directory, or a file the run reads changed in place, is other physics."""
    before = physics_hash(_physics(fake_case))
    other = _fake_case(tmp_path / "other")
    assert physics_hash(_physics(other)) != before

    (case_module.required_files(fake_case)[2]).write_text('{"LV": 30}')  # markers.json
    assert physics_hash(_physics(fake_case)) != before


def test_the_same_case_by_another_path_keeps_the_hash(fake_case, monkeypatch):
    """The case directory is hashed by its resolved path, so a relative path to it, or
    the same path with ``..`` in it, is the same physics."""
    absolute = physics_hash(_physics(fake_case))
    monkeypatch.chdir(fake_case.parent)
    assert physics_hash(_physics(Path(fake_case.name))) == absolute
    assert physics_hash(_physics(fake_case / ".." / fake_case.name)) == absolute


@pytest.mark.parametrize(
    "argv",
    [
        ("--t-end", "4"),
        ("--save-every", "4"),
        ("--save-every-ep", "2"),
        ("--checkpoint-every", "10"),
        ("--output-dir", "/elsewhere"),
        ("--restart",),
        ("--overwrite",),
    ],
    ids=lambda argv: argv[0],
)
def test_output_options_leave_the_hash(fake_case, argv):
    assert physics_hash(_physics(fake_case, *argv)) == physics_hash(_physics(fake_case))


def test_physics_holds_what_case_py_copies_from_physcardems(fake_case):
    """The Land parameters as computed, the cycles as passed to pulse (SI), the tags,
    the valve-plug stiffness and the reference scaling."""
    physics = _physics(fake_case)
    assert physics["land"] == case_module.land_parameters(case_module.TREF_SCALE)
    assert physics["cycle"] == {
        name: dataclasses.asdict(params) for name, params in case_module.cycle_parameters().items()
    }
    assert physics["case"]["tags"] == case_module.TAGS
    assert physics["case"]["valve_stiffness_scale"] == 3.0
    assert physics["case"]["reference_scale"] == 0.9
    assert physics["mechanics"]["quadrature_degree"] == case_module.QUADRATURE_DEGREE
    assert (physics["dt_mech"], physics["dt_ep"]) == (2.0, 0.05)


def _material_in_base_units() -> dict:
    return {
        model: {name: float(value.to_base_units()) for name, value in values.items()}
        for model, values in rodero.material_parameters().items()
    }


def test_physics_holds_the_material_pulse_is_given(fake_case):
    """The material in the physics is :func:`material_parameters` in SI base units, and
    :func:`material` passes pulse exactly those values, each modulus times the case's
    stiffness scale, on a stand-in case of one cube's six cells."""
    material = _physics(fake_case)["mechanics"]["material"]
    assert {k: v for k, v in material.items() if k in rodero.material_parameters()} == (
        _material_in_base_units()
    )
    assert material["HolzapfelOgden"]["a"] == 610.0  # Pa
    assert material["HolzapfelOgden"]["b_f"] == 35.31
    assert material["Compressible2"]["kappa"] == 1e6  # Pa
    assert material["Viscous"]["eta"] == 100.0  # Pa s
    assert material["moduli_scaled_by_stiffness"] == list(rodero.MODULI)

    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    scale = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("DG", 0)))
    scale.x.array[:] = [1.0, 3.0, 1.0, 1.0, 3.0, 1.0]
    f0 = dolfinx.fem.Constant(mesh, np.array([1.0, 0.0, 0.0]))
    s0 = dolfinx.fem.Constant(mesh, np.array([0.0, 1.0, 0.0]))
    given = rodero.material(SimpleNamespace(stiffness_scale=scale, f0=f0, s0=s0))

    for name, value in material["HolzapfelOgden"].items():
        variable = getattr(given, name)
        if name in rodero.MODULI:
            # A DG0 field, in kPa: the modulus times the scale, cell by cell.
            np.testing.assert_allclose(
                variable.value.x.array * variable.factor,
                value * scale.x.array,
                rtol=1e-15,
            )
        else:
            assert float(variable.to_base_units()) == value
    assert given.use_heaviside is material["use_heaviside"]
    assert given.use_subplus is material["use_subplus"]


@pytest.fixture
def no_build(monkeypatch):
    """Replace the builders main reaches first with ones that fail if called."""

    def fail(*args, **kwargs):
        raise AssertionError("main built something before refusing the run")

    for name in ("load_ode_modules", "load_case"):
        monkeypatch.setattr(rodero, name, fail)


def test_refuses_a_save_interval_off_the_step_before_building(tmp_path, no_build):
    """``--save-every 3`` with the 2 ms step is refused, naming the option, before
    anything is built or written."""
    outdir = tmp_path / "out"
    outdir.mkdir()
    with pytest.raises(SystemExit) as refusal:
        rodero.main(["--t-end", "4", "--save-every", "3", "--output-dir", str(outdir)])
    assert "--save-every" in str(refusal.value)
    assert list(outdir.iterdir()) == []


def test_refuses_a_restart_without_a_checkpoint_before_building(tmp_path, fake_case, no_build):
    """``--restart`` into a folder with no checkpoint is refused before anything is
    built or written."""
    outdir = tmp_path / "out"
    outdir.mkdir()
    with pytest.raises(SystemExit) as refusal:
        rodero.main(
            [
                "--t-end",
                "4",
                "--restart",
                "--case-dir",
                str(fake_case),
                "--output-dir",
                str(outdir),
            ],
        )
    assert "No checkpoint to restart from" in str(refusal.value)
    assert list(outdir.iterdir()) == []


def test_refuses_a_case_with_files_missing_before_building(tmp_path, no_build):
    """A case directory missing the files the run reads is refused, naming them, before
    anything is built, and no output folder is made."""
    outdir = tmp_path / "out"
    with pytest.raises(SystemExit) as refusal:
        rodero.main(["--t-end", "4", "--case-dir", str(tmp_path), "--output-dir", str(outdir)])
    assert "geometry.bp" in str(refusal.value)
    assert not outdir.exists()


# ---------------------------------------------------------------------------
# The summary, from log.csv alone
# ---------------------------------------------------------------------------

#: A run of five 2 ms steps after the unloaded solve (row 0). Each step's phase is the
#: phase it was solved under, and ``next_phase`` the cycle's phase after it, so a switch
#: is a row whose two differ: the LV goes PRELOAD -> IVC at 2 ms, IVC -> EJECTION at 6
#: ms and EJECTION -> IVR at 10 ms; the RV PRELOAD -> IVC at 8 ms. The step to 6 ms took
#: two solves: the first diverged (reason -3, 50 iterations), the retry converged.
SYNTHETIC: dict[str, list[float]] = {
    "t_ms": [0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
    "phase_LV": [0, 0, 1, 1, 2, 2],
    "next_phase_LV": [0, 1, 1, 2, 2, 3],
    "V_LV_mL": [120.0, 121.0, 121.0, 121.0, 110.0, 100.0],
    "P_LV_kPa": [0.0, 1.0, 5.0, 11.0, 14.0, 12.0],
    "phase_RV": [0, 0, 0, 0, 0, 1],
    "next_phase_RV": [0, 0, 0, 0, 1, 1],
    "V_RV_mL": [150.0, 151.0, 152.0, 152.5, 153.0, 153.0],
    "P_RV_kPa": [0.0, 0.2, 0.3, 0.4, 0.5, 2.0],
    "newton_iterations": [3, 4, 5, 4, 6, 5],
    "snes_reason": [3, 2, 2, 2, 3, 2],
    "linear_iterations": [10, 20, 25, 30, 22, 21],
    "solve_attempts": [1, 1, 1, 2, 1, 1],
    "first_iterations": [3, 4, 5, 50, 6, 5],
    "first_reason": [3, 2, 2, -3, 3, 2],
    "detF_min": [0.99, 0.98, 0.97, 0.95, 0.96, 0.97],
    "n_detF_nonpositive": [0, 0, 0, 0, 0, 0],
    "Ta_min_kPa": [0.0, -0.1, 0.0, 0.0, 0.0, 0.0],
    "Ta_max_kPa": [0.0, 1.0, 30.0, 60.0, 80.0, 70.0],
    "lmbda_min": [0.99, 0.98, 0.95, 0.93, 0.9, 0.88],
    "lmbda_max": [1.01, 1.02, 1.03, 1.04, 1.05, 1.06],
}


@pytest.fixture
def synthetic_log(tmp_path) -> Path:
    """:data:`SYNTHETIC` written as the run writes ``log.csv`` (every other column 0)."""
    path = tmp_path / "log.csv"
    log = CsvLog(path, rodero.LOG_FIELDS)
    log.start()
    for i in range(len(SYNTHETIC["t_ms"])):
        log.append({field: SYNTHETIC.get(field, [0.0] * 6)[i] for field in rodero.LOG_FIELDS})
    return path


def test_log_has_the_columns_the_summary_needs():
    for name in ("first_reason", "first_iterations", "next_phase_LV", "next_phase_RV"):
        assert name in rodero.LOG_FIELDS


def test_summarise_from_log_csv_alone(synthetic_log):
    columns = post.read_columns(synthetic_log)
    placement = {"ep": ["Tref"], "mechanics": ["Tref", "kws"]}
    summary = post.summarise(columns, None, placement, t_end_ms=10.0)

    assert summary["t_end_ms"] == 10.0
    assert summary["failure"] is None
    assert summary["phase_switches"] == [
        {"chamber": "LV", "from": "PRELOAD", "to": "ISOVOLUMIC_CONTRACTION", "t_ms": 2.0},
        {"chamber": "LV", "from": "ISOVOLUMIC_CONTRACTION", "to": "EJECTION", "t_ms": 6.0},
        {"chamber": "RV", "from": "PRELOAD", "to": "ISOVOLUMIC_CONTRACTION", "t_ms": 8.0},
        {"chamber": "LV", "from": "EJECTION", "to": "ISOVOLUMIC_RELAXATION", "t_ms": 10.0},
    ]
    lv = summary["LV"]
    assert lv["phases"] == ["PRELOAD", "ISOVOLUMIC_CONTRACTION", "EJECTION"]
    assert lv["five_phases_in_order"] is False
    # EDV at the row before the first IVC row (the switch), ESV the smallest since.
    assert lv["ejection"] == {
        "EDV_mL": 121.0,
        "ESV_mL": 100.0,
        "EF_percent": pytest.approx(100.0 * 21.0 / 121.0),
    }
    assert (lv["peak_P_kPa"], lv["t_peak_P_ms"]) == (14.0, 8.0)
    assert summary["RV"]["ejection"] == {"EDV_mL": 153.0, "ESV_mL": 153.0, "EF_percent": 0.0}

    # Row 0 is the unloaded solve; the rest are the steps, each by its last solve.
    assert summary["newton"] == {
        "steps": 5,
        "unloaded_solve": {
            "iterations": 3,
            "reason": 3,
            "linear_iterations": 10,
            "converged": True,
        },
        "iterations_min": 4,
        "iterations_mean": 4.8,
        "iterations_max": 6,
        "final_reasons": {"2": 4, "3": 1},
        "retries": 1,
        "steps_with_retry_ms": [6.0],
        "all_attempts_reasons": ["-3", "2", "3"],
    }
    assert (summary["detF_min"], summary["t_detF_min_ms"]) == (0.95, 6.0)
    assert (summary["Ta_min_kPa"], summary["Ta_max_kPa"]) == (-0.1, 80.0)
    assert (summary["lmbda_min"], summary["lmbda_max"]) == (0.88, 1.06)
    assert summary["criteria"] == {
        "reached_t_end": True,
        "newton_converged_every_step": True,
        "detF_positive_every_step": True,
        "LV_five_phases_in_order": False,
        "RV_five_phases_in_order": False,
    }
    assert summary["land_placement"] == placement


def test_summarise_reports_a_failure_and_an_early_end(synthetic_log):
    columns = post.read_columns(synthetic_log)
    summary = post.summarise(columns, "RuntimeError('x')", {}, t_end_ms=20.0)
    assert summary["failure"] == "RuntimeError('x')"
    assert summary["criteria"]["reached_t_end"] is False
    assert summary["criteria"]["newton_converged_every_step"] is False


def test_a_run_that_never_finished_is_not_reported_converged(synthetic_log):
    """A run killed outright leaves ``run.json`` at ``status: running``; post.py's summary
    then reports it as a failure, not as converged at every step."""
    assert post.run_failure(None) is None
    assert post.run_failure({"status": "finished", "failure": None}) is None
    assert post.run_failure({"status": "failed", "failure": "RuntimeError('x')"}) == (
        "RuntimeError('x')"
    )
    failure = post.run_failure({"status": "running", "failure": None})
    assert "did not finish" in failure
    summary = post.summarise(post.read_columns(synthetic_log), failure, {}, t_end_ms=10.0)
    assert summary["criteria"]["newton_converged_every_step"] is False
