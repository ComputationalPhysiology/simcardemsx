"""The BiV example (``numerical_experiments/circulation_biv``) without its mesh: what its
physics hash holds, its refusals before anything is built, and its summary.

Importing ``circulation_biv.main`` builds nothing, and none of these tests reaches the
mesh cache, the code generation or the problem: the refusals are checked with the
builders replaced by ones that fail if called.
"""

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

from simcardemsx.checkpoint import physics_hash

EXAMPLES = Path(__file__).parent.parent / "numerical_experiments"
sys.path.insert(0, str(EXAMPLES))
try:
    biv = importlib.import_module("circulation_biv.main")
    post = importlib.import_module("circulation_biv.post")
finally:
    sys.path.remove(str(EXAMPLES))


def _physics(*argv: str) -> dict:
    return biv.physics(biv.parse_args(list(argv)))


#: The options that define the physics; every other option must be in ``NOT_PHYSICS``.
PHYSICS_OPTIONS = {"tref", "scheme", "dt_mech"}


def test_every_option_is_physics_or_not():
    """Each option ``parse_args`` defines is either one of :data:`PHYSICS_OPTIONS`, and
    in the physics, or named in ``NOT_PHYSICS``, and not in them. ``physics`` keeps every
    option ``NOT_PHYSICS`` does not name, so an option added without a decision lands in
    the hash, where it would refuse restarts: this test fails until it is placed here or
    in ``NOT_PHYSICS``."""
    options = set(vars(biv.parse_args([])))
    physics = _physics()
    not_physics = set(biv.NOT_PHYSICS)

    assert not_physics <= options, f"NOT_PHYSICS names no option: {sorted(not_physics - options)}"
    assert not (not_physics & set(physics)), sorted(not_physics & set(physics))
    assert options & set(physics) == PHYSICS_OPTIONS, (
        f"options in the physics: {sorted(options & set(physics))}; an output option belongs "
        "in NOT_PHYSICS, a physics option in PHYSICS_OPTIONS"
    )
    unplaced = options - not_physics - PHYSICS_OPTIONS
    assert not unplaced, f"options neither physics nor in NOT_PHYSICS: {sorted(unplaced)}"


@pytest.mark.parametrize(
    "argv",
    [("--tref", "5"), ("--scheme", "stabilized"), ("--dt-mech", "1")],
    ids=lambda argv: argv[0],
)
def test_physics_options_change_the_hash(argv):
    # The value given is not the default, or the comparison would be vacuous.
    assert biv.parse_args(list(argv)) != biv.parse_args([])
    assert physics_hash(_physics(*argv)) != physics_hash(_physics())


@pytest.mark.parametrize(
    "argv",
    [
        ("--t-end", "4"),
        ("--save-every", "4"),
        ("--save-every-ep", "2"),
        ("--checkpoint-every", "10"),
        ("--outdir", "/elsewhere"),
        ("--snapshot-every", "2"),
    ],
    ids=lambda argv: argv[0],
)
def test_output_options_leave_the_hash(argv):
    assert physics_hash(_physics(*argv)) == physics_hash(_physics())


def test_physics_holds_the_material_pulse_is_given():
    """The material in the physics is :func:`material_parameters` (what
    ``cardiac_model`` passes to pulse) in SI base units."""
    material = _physics()["mechanics"]["material"]
    expected = {
        model: {name: float(value.to_base_units()) for name, value in values.items()}
        for model, values in biv.material_parameters().items()
    }
    assert {key: value for key, value in material.items() if key != "units"} == expected
    assert material["HolzapfelOgden"]["a"] == 2280.0  # Pa
    assert material["Viscous"]["eta"] == 100.0  # Pa s


def test_circuit_period_goes_in_before_flattening():
    """The circuit's ``RR`` is the beat's period: the heart rate is set before
    ``flat_ode_parameters`` derives ``RR`` from it (Regazzoni's default ``RR`` is
    0.8 s, which overriding the flat ``HR`` afterwards would leave)."""
    parameters = _physics()["circulation"]["parameters"]
    assert parameters["RR"] == biv.PERIOD / 1000.0
    assert biv.circuit_parameters(500.0)["RR"] == 0.5


@pytest.fixture
def no_build(monkeypatch):
    """Replace the builders main reaches first with ones that fail if called."""

    def fail(*args, **kwargs):
        raise AssertionError("main built something before refusing the run")

    for name in ("load_ode_modules", "generate_mesh", "load_geometry"):
        monkeypatch.setattr(biv, name, fail)


def test_refuses_a_save_interval_off_the_step_before_building(tmp_path, no_build):
    """``--save-every 3`` with the default 2 ms step is refused, naming the option,
    before anything is built or written."""
    outdir = tmp_path / "out"
    outdir.mkdir()
    with pytest.raises(SystemExit) as refusal:
        biv.main(["--t-end", "4", "--save-every", "3", "--outdir", str(outdir)])
    assert "--save-every" in str(refusal.value)
    assert list(outdir.iterdir()) == []


def test_refuses_a_restart_without_a_checkpoint_before_building(tmp_path, no_build):
    """``--restart`` into a folder with no checkpoint is refused before anything is
    built or written."""
    with pytest.raises(SystemExit) as refusal:
        biv.main(["--t-end", "4", "--restart", "--outdir", str(tmp_path)])
    assert "No checkpoint to restart from" in str(refusal.value)
    assert list(tmp_path.iterdir()) == []


def test_valve_open_intervals():
    """Runs of rows with ``p > p_out``, the volume ejected measured from the row before
    each run to its last row (from the first row if the run starts there)."""
    t = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    V = np.array([100.0, 99.0, 90.0, 80.0, 80.0, 75.0])
    p = np.array([90.0, 50.0, 90.0, 85.0, 40.0, 95.0])
    p_out = np.full_like(t, 80.0)

    assert post.valve_open_intervals(t, V, p, p_out) == [
        {"first_open_ms": 0.0, "last_open_ms": 0.0, "open_at_end": False, "ejected_mL": 0.0},
        {"first_open_ms": 4.0, "last_open_ms": 6.0, "open_at_end": False, "ejected_mL": 19.0},
        {"first_open_ms": 10.0, "last_open_ms": 10.0, "open_at_end": True, "ejected_mL": 5.0},
    ]
    assert post.valve_open_intervals(t, V, np.zeros_like(t), p_out) == []


def test_summarise_synthetic_columns():
    t = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
    columns = {
        "t_ms": t,
        "V_LV_mL": np.array([100.0, 100.0, 90.0, 80.0, 80.0]),
        "p_LV_mmHg": np.array([10.0, 50.0, 90.0, 85.0, 40.0]),
        "circuit_p_AR_SYS": np.full_like(t, 80.0),
        "V_RV_mL": np.array([70.0, 71.0, 72.0, 71.0, 70.0]),
        "p_RV_mmHg": np.array([4.0, 6.0, 8.0, 6.0, 4.0]),
        "circuit_p_AR_PUL": np.full_like(t, 15.0),
        "conservation_drift": np.array([0.0, 1e-16, 3e-16, 2e-16, 1e-16]),
        "Ta_mean_kPa": np.array([0.0, 1.0, 5.0, 3.0, 2.0]),
        "newton_iterations": np.array([0.0, 3.0, 2.0, 4.0, 3.0]),
    }
    summary = post.summarise(columns, failed_at=8.5)

    lv, rv = summary["LV"], summary["RV"]
    assert (lv["EDV_mL"], lv["ESV_mL"], lv["V_range_mL"]) == (100.0, 80.0, 20.0)
    assert lv["V_range_fraction"] == 0.2
    assert (lv["peak_p_mmHg"], lv["t_peak_p_ms"]) == (90.0, 4.0)
    assert lv["outflow_valve_open"] == [
        {"first_open_ms": 4.0, "last_open_ms": 6.0, "open_at_end": False, "ejected_mL": 20.0},
    ]
    assert lv["ejects"] is True
    assert rv["outflow_valve_open"] == []
    assert rv["ejects"] is False
    assert (rv["EDV_mL"], rv["ESV_mL"]) == (72.0, 70.0)
    assert summary["max_conservation_drift"] == 3e-16
    # The row at t = 0 has no solve behind it and is left out of the counts.
    assert summary["newton_iterations"] == {
        "steps": 4,
        "min": 2,
        "mean": 3.0,
        "max": 4,
        "failed_at_ms": 8.5,
    }
    assert summary["peak_Ta_mean_kPa"] == 5.0
