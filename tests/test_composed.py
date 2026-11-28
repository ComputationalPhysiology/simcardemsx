"""The composed ToR-ORd + Land helper (examples/odefiles/composed.py)."""

import importlib
import os
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
# solver), and λ to EP. There it acts through Land's length-dependent Ca50
# (Ca50_raw = ca50_ref + beta1·(Lambda_c − 1)) on CaTRPN's binding rate, and so on J_TRPN
# and Ca_i; the passive state Cd is on the EP side too, but feeds nothing EP integrates.
EXPECTED = {
    "zeta": (("S", "W"), ("Zs", "Zw"), True),
    "cai": (("cai",), ("J_TRPN",), True),
    "catrpn": (("CaTRPN",), (), True),
}


@pytest.fixture(autouse=True)
def fresh_composition():
    """Every test composes afresh: ``torord_land`` keeps each split's text for the rest of
    the process, so a test that fetches (or fails to) must not see an earlier test's."""
    composed.compose.cache_clear()
    yield
    composed.compose.cache_clear()


class _FakeODE:
    """What ``torord_land`` uses of gotranx's ODE: ``split(...).save(path)``."""

    def __init__(self, text: str):
        self.text = text

    def split(self, name, inputs, outputs):
        return _FakeODE(f"{self.text} split {name} {inputs} {outputs}\n")

    def save(self, path):
        Path(path).write_text(self.text)


@pytest.fixture
def fake_odezoo(monkeypatch) -> list[dict]:
    """``torord_land`` offline: odezoo and gotranx.compose replaced by fakes; returns the
    recipes ``compose`` was called with."""
    recipes: list[dict] = []

    def compose(recipe, base_dir):
        recipes.append(recipe)
        return _FakeODE("composed")

    monkeypatch.setattr(composed, "_fetch", lambda name: Path(f"/zoo/{name}.ode"))
    monkeypatch.setattr(composed.gotranx, "compose", compose)
    return recipes


def test_composing_happens_once_per_split_per_process(fake_odezoo, tmp_path):
    """The slab asks for its ``.ode`` file several times per run (the physics, the
    settings, the build); only the first call per split composes."""
    a = composed.torord_land("zeta", tmp_path / "a")
    b = composed.torord_land("zeta", tmp_path / "b")
    assert len(fake_odezoo) == 1
    assert a.read_text() == b.read_text() != ""
    composed.torord_land("cai", tmp_path / "a")
    assert len(fake_odezoo) == 2


def test_the_file_is_written_only_when_its_text_differs(fake_odezoo, tmp_path):
    path = composed.torord_land("zeta", tmp_path)
    good = path.read_text()
    mtime = path.stat().st_mtime_ns
    assert composed.torord_land("zeta", tmp_path).stat().st_mtime_ns == mtime
    path.write_text("# stale\n")
    assert composed.torord_land("zeta", tmp_path).read_text() == good
    assert len(fake_odezoo) == 1


def test_the_temporary_file_is_this_process_own(fake_odezoo, tmp_path, monkeypatch):
    """Concurrent runs (another process's temporary file beside the target) do not
    overwrite or remove each other's temporary files, and none is left behind."""
    monkeypatch.setattr(os, "getpid", lambda: 4242)
    other = tmp_path / ".torord_land_zeta.ode.4243.tmp"
    other.write_text("another process's\n")
    path = composed.torord_land("zeta", tmp_path)
    assert other.read_text() == "another process's\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([path.name, other.name])


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
    composed.compose.cache_clear()  # compose again, rather than reuse this process's text
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
    composed.compose.cache_clear()
    assert composed.torord_land("zeta", tmp_path).read_text() == good
    mtime = path.stat().st_mtime_ns
    composed.torord_land("zeta", tmp_path)
    assert path.stat().st_mtime_ns == mtime  # a correct file is left alone
