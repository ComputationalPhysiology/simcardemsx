"""The composed ToR-ORd + Land helper (examples/odefiles/composed.py)."""

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from simcardemsx.ode_model import load_ode_modules
from simcardemsx.transfer_plan import resolve

EXAMPLES = Path(__file__).parent.parent / "examples"
sys.path.insert(0, str(EXAMPLES))
try:
    composed = importlib.import_module("odefiles.composed")
finally:
    sys.path.remove(str(EXAMPLES))

# (forward, backward, stretch_to_ep) as transfer_plan.resolve returns them. The CaTRPN split
# was determined by running it: CaTRPN in, nothing back to EP (Ta is read by the mechanics
# solver), and λ to EP through Land's passive state Cd.
EXPECTED = {
    "zeta": (("S", "W"), ("Zs", "Zw"), True),
    "cai": (("cai",), ("J_TRPN",), True),
    "catrpn": (("CaTRPN",), (), True),
}


def test_place_puts_each_name_on_every_side_that_declares_it():
    modules = SimpleNamespace(
        ep=SimpleNamespace(parameter={"Tref": 0, "ku": 1}),
        mechanics=SimpleNamespace(parameter={"Tref": 0}),
    )
    assert composed.place({"Tref": 1.0, "ku": 2.0}, modules) == {
        "ep": {"Tref": 1.0, "ku": 2.0},
        "mechanics": {"Tref": 1.0},
    }
    with pytest.raises(KeyError, match="nTm"):
        composed.place({"nTm": 2.4}, modules)


def test_initial_states_are_restricted_to_the_modules_states():
    assert composed.initial_states(SimpleNamespace(state={"B": 0, "S": 1})) == {"B": 1.0}
    assert composed.initial_states(SimpleNamespace(state={"S": 1})) == {}


def test_unknown_split_is_refused(tmp_path):
    with pytest.raises(ValueError, match="zeta"):
        composed.torord_land("bogus", tmp_path)


def test_offline_first_run_names_odezoo(tmp_path, monkeypatch):
    pytest.importorskip("odezoo")

    def fail(*a, **k):
        raise OSError("no network")

    monkeypatch.setattr(composed.odezoo, "get", fail)
    with pytest.raises(RuntimeError, match="odezoo.*torord_dyn_cl.*github-11e463d"):
        composed.torord_land("zeta", tmp_path)


@pytest.mark.slow
@pytest.mark.parametrize("split", ["zeta", "cai", "catrpn"])
def test_each_split_composes_generates_and_resolves(split, tmp_path):
    pytest.importorskip("odezoo")
    path = composed.torord_land(split, tmp_path)
    first = path.read_text()
    assert composed.torord_land(split, tmp_path).read_text() == first  # deterministic
    modules = load_ode_modules(path, tmp_path / "generated")
    crossings = resolve(modules.ep, modules.mechanics)
    assert (crossings.forward, crossings.backward, crossings.stretch_to_ep) == EXPECTED[split]
    # Land's calibration lands on a side that declares it
    assert composed.place(composed.LAND_CALIBRATION, modules)


@pytest.mark.slow
def test_a_stale_composed_file_is_rewritten(tmp_path):
    pytest.importorskip("odezoo")
    path = composed.torord_land("zeta", tmp_path)
    good = path.read_text()
    path.write_text("# stale\n")
    assert composed.torord_land("zeta", tmp_path).read_text() == good
    mtime = path.stat().st_mtime_ns
    composed.torord_land("zeta", tmp_path)
    assert path.stat().st_mtime_ns == mtime  # a correct file is left alone
