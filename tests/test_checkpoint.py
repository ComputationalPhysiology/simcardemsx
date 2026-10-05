"""Gates R1 and R3: a coupled run written to disk and read back by :class:`Checkpointer`.

**R1, restart.** Run ``N`` steps straight. Separately, run ``K`` steps, write a
checkpoint, build every object afresh, restore, and run ``N - K`` steps. Everything
:func:`conftest._coupled_state` copies (every restart Function and metadata value, and
EP's whole ``parameters`` and ``missing_variables`` arrays) must then be bit-identical
to the uninterrupted run's. The restored run must also equal the checkpointed one right
after the restore: a difference there is the round trip through ``restart.bp``, and one
that appears only after stepping is state that no component saves.

The cases are on gate 5's harness (``tests/test_round_trip.py``): beat EP on the 3x3x3
cube, ToR-ORd firing on its own stimulus at t = 0, and one element of mechanics,
coupled through :class:`SimulationController` on pulse's default solver, a fresh LU at
every Newton iteration. They are :class:`GeneratedActivation` under every scheme on
the zeta split, and monolithic on the Ca_i split; ``CrossbridgeSegregated`` Land2017 on
the Ca_i split (X2's harness, slow); and D1's ``DynamicProblem`` element (slow). A
checkpoint before any step is its own test: EP's crossing rows then hold their
constructed values, which no backend output reproduces (its λ output is 0 then).

**R3, refusals.** Another physics hash, other function names, or a time that
``restart.bp`` does not hold raise ``ValueError`` naming what differs, before anything is
read; no ``restart.json`` raises ``FileNotFoundError``; writing with a step pending
raises ``RuntimeError``.

A checkpoint killed after ``restart.bp`` but before ``restart.json`` (Review Focus) is
not used: the next restore starts from the previous one, and the later checkpoint at
the killed time is skipped in ``restart.bp`` and read back correctly.
"""

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mpi4py import MPI

import dolfinx
import io4dolfinx
import pytest
from conftest import _assert_same_state, _coupled_state, _crossbridge_factory

from simcardemsx import checkpoint
from simcardemsx.checkpoint import (
    RESTART,
    RESTART_META,
    Checkpointer,
    check_restart,
    physics_hash,
    write_json,
)
from simcardemsx.controller import SimulationController

DT_EP = 0.05
DT_MECH = 1.0
#: The uninterrupted run's steps, and the step the checkpoint is written after.
N, K = 6, 3

#: The physics of the default build, for the refusal tests.
P = {"split": "zetasplit", "scheme": "monolithic"}

_Build = Callable[[], SimulationController]


def _unit_cube(n: int) -> dolfinx.mesh.Mesh:
    return dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n)


@pytest.fixture
def build(split_modules, make_ep_solver, make_mechanics, make_dynamic_mechanics):
    """``build(split, scheme, *, crossbridge=False, dynamic=False, dt_mech=DT_MECH)``:
    gate 5's controller, with every object new: the EP solver, the mechanics problem,
    the backend and the controller.

    ``crossbridge`` puts ``CrossbridgeSegregated`` Land2017 in place of the
    ``GeneratedActivation`` (X2's harness; ``scheme`` is then not read), and
    ``dynamic`` D1's ``DynamicProblem`` element, a 1 cm cube, in place of the static
    unit cube. EP's cube is then scaled to the same 1 cm, so that every EP node lies in
    the mechanics mesh that its crossing rows are averaged from.
    """

    def build(
        split: str = "zetasplit",
        scheme: str = "monolithic",
        *,
        crossbridge: bool = False,
        dynamic: bool = False,
        dt_mech: float = DT_MECH,
    ) -> SimulationController:
        modules = split_modules[split]
        ep_mesh = _unit_cube(3)
        if dynamic:
            ep_mesh.geometry.x[:] *= 0.01
            problem, backend = make_dynamic_mechanics(
                modules.mechanics,
                dt_ms=DT_MECH,
                scheme=scheme,
            )
        else:
            problem, backend = make_mechanics(
                modules.mechanics,
                _unit_cube(1),
                quadrature_degree=2,
                scheme=scheme,
                backend_factory=(
                    _crossbridge_factory("Land2017", modules.mechanics) if crossbridge else None
                ),
            )
        ep_solver = make_ep_solver(modules.ep, ep_mesh)
        return SimulationController(problem, ep_solver, backend, modules, dt_mech, DT_EP)

    return build


def _steps(controller: SimulationController, n: int) -> SimulationController:
    for _ in range(n):
        controller.step()
    return controller


@dataclass
class _Recording:
    """An extra component with sidecar files, recording what the checkpointer calls."""

    namespace: str = "recording"
    metadata: dict[str, Any] = field(default_factory=dict)
    functions: list[tuple[str, dolfinx.fem.Function]] = field(default_factory=list)
    calls: list[tuple] = field(default_factory=list)

    def restart_functions(self) -> list[tuple[str, dolfinx.fem.Function]]:
        return list(self.functions)

    def restart_metadata(self) -> dict[str, Any]:
        return dict(self.metadata)

    def load_restart(self, functions, metadata) -> None:
        self.calls.append(("load_restart", dict(metadata)))

    def write_sidecar(self, folder: Path, t_ms: float) -> None:
        self.calls.append(("write_sidecar", folder, t_ms, (folder / RESTART_META).exists()))

    def read_sidecar(self, folder: Path, t_ms: float) -> None:
        self.calls.append(("read_sidecar", folder, t_ms))


def _restart_matches_the_uninterrupted_run(
    folder: Path,
    build: _Build,
    physics: dict[str, Any],
    k: int = K,
) -> None:
    """R1: ``N`` steps straight against ``k``, a checkpoint, a restore into fresh
    objects, and ``N - k``."""
    a = _steps(build(), N)

    b = _steps(build(), k)
    Checkpointer(b, folder, physics=physics).write()
    at_checkpoint = _coupled_state(b)

    c = build()
    checkpointer = Checkpointer(c, folder, physics=physics)
    assert checkpointer.restore() == k * DT_MECH
    _assert_same_state(_coupled_state(c), at_checkpoint)
    assert len(checkpointer.history) == 2

    _steps(c, N - k)
    _assert_same_state(_coupled_state(a), _coupled_state(c))


@pytest.mark.parametrize(
    "split, scheme",
    [
        ("zetasplit", "monolithic"),
        ("zetasplit", "segregated"),
        ("zetasplit", "stabilized"),
        ("caisplit", "monolithic"),
    ],
)
def test_restart_matches_the_uninterrupted_run(tmp_path, build, split, scheme):
    _restart_matches_the_uninterrupted_run(
        tmp_path,
        lambda: build(split, scheme),
        {"split": split, "scheme": scheme},
    )


@pytest.mark.slow
def test_restart_matches_the_uninterrupted_run_crossbridge(tmp_path, build):
    _restart_matches_the_uninterrupted_run(
        tmp_path,
        lambda: build("caisplit", crossbridge=True),
        {"split": "caisplit", "backend": "crossbridge:Land2017", "stabilized": True},
    )


@pytest.mark.slow
def test_restart_matches_the_uninterrupted_run_dynamic(tmp_path, build):
    _restart_matches_the_uninterrupted_run(
        tmp_path,
        lambda: build("zetasplit", dynamic=True),
        {"split": "zetasplit", "scheme": "monolithic", "problem": "dynamic"},
    )


def test_restart_at_time_zero(tmp_path, build):
    _restart_matches_the_uninterrupted_run(tmp_path, build, P, k=0)


def test_restore_accepts_the_same_names_in_another_order(tmp_path, build):
    """A checkpoint that lists every namespace's function names, and EP's state names,
    in another order than this run's (as one written by a process whose gotranx ordered
    the EP states differently does) is restored by name, and continues bit for bit."""
    a = _steps(build(), N)

    b = _steps(build(), K)
    Checkpointer(b, tmp_path, physics=P).write()
    at_checkpoint = _coupled_state(b)
    meta = json.loads((tmp_path / RESTART_META).read_text())
    functions = meta["simcardemsx"]["functions"]
    meta["simcardemsx"]["functions"] = {ns: names[::-1] for ns, names in functions.items()}
    meta["ep"]["state_names"] = meta["ep"]["state_names"][::-1]
    (tmp_path / RESTART_META).write_text(json.dumps(meta))

    c = build()
    assert Checkpointer(c, tmp_path, physics=P).restore() == K * DT_MECH
    _assert_same_state(_coupled_state(c), at_checkpoint)
    _steps(c, N - K)
    _assert_same_state(_coupled_state(a), _coupled_state(c))


def test_restore_refuses_another_set_of_ep_state_names(tmp_path, build):
    """The same function names, but EP state names that are not this solver's: refused
    by ``EPState``, and undone."""
    b = _steps(build(), 1)
    Checkpointer(b, tmp_path, physics=P).write()
    meta = json.loads((tmp_path / RESTART_META).read_text())
    meta["ep"]["state_names"] = [*meta["ep"]["state_names"][:-1], "not_a_state"]
    (tmp_path / RESTART_META).write_text(json.dumps(meta))

    c = build()
    raised = _refused(
        c,
        Checkpointer(c, tmp_path, physics=P).restore,
        ValueError,
        "EP state names differ",
    )
    assert "not_a_state" in str(raised.value)


def _refused(controller: SimulationController, restore: Callable[[], Any], error, match: str):
    """``restore()`` raises ``error`` matching ``match``, and leaves ``controller`` as it was."""
    before = _coupled_state(controller)
    with pytest.raises(error, match=match) as raised:
        restore()
    _assert_same_state(_coupled_state(controller), before)
    return raised


def test_restore_refuses_another_physics_hash(tmp_path, build):
    b = _steps(build(), 1)
    Checkpointer(b, tmp_path, physics=P).write()

    c = build()
    other = Checkpointer(c, tmp_path, physics={**P, "scheme": "segregated"})
    raised = _refused(c, other.restore, ValueError, "physics")
    assert str(tmp_path / "config.resolved.toml") in str(raised.value)

    with pytest.raises(ValueError, match="physics"):
        check_restart(tmp_path, {**P, "scheme": "segregated"})
    check_restart(tmp_path, dict(reversed(P.items())))


def test_restore_refuses_other_function_names(tmp_path, build):
    """A zeta-split checkpoint into a Ca_i-split run, under the same physics dict."""
    b = _steps(build("zetasplit"), 1)
    Checkpointer(b, tmp_path, physics=P).write()

    c = build("caisplit")
    checkpointer = Checkpointer(c, tmp_path, physics=P)
    raised = _refused(c, checkpointer.restore, ValueError, "activation_output_J_TRPN")
    assert "activation_output_Zetas" in str(raised.value)


@pytest.mark.parametrize(
    "other, match",
    [({"scheme": "segregated"}, "scheme must match"), ({"dt_mech": 0.5}, "dt_mech")],
)
def test_a_component_refusing_the_checkpoint_undoes_the_restore(tmp_path, build, other, match):
    """Another scheme, or another ``dt_mech``, under the same physics dict: the
    checkpointer's own checks pass, every Function is read, and then a component's
    ``load_restart`` refuses (the backend's, or the controller's, the last one). The
    restore must then put back everything the reads overwrote."""
    b = _steps(build(), 2)
    Checkpointer(b, tmp_path, physics=P).write()

    c = build(**other)
    checkpointer = Checkpointer(c, tmp_path, physics=P)
    _refused(c, checkpointer.restore, ValueError, match)
    assert checkpointer.history == [checkpointer.provenance]


@dataclass
class _Refusing(_Recording):
    """Refuses its first ``load_restart`` (the restore), and fails its second (the undo)."""

    def load_restart(self, functions, metadata) -> None:
        super().load_restart(functions, metadata)
        if len(self.calls) == 1:
            raise ValueError("refused")
        raise RuntimeError("cannot undo")


def test_a_failing_undo_is_raised_with_the_refusal_as_its_cause(tmp_path, build):
    b = _steps(build(), 1)
    Checkpointer(b, tmp_path, physics=P, extra=[_Recording()]).write()

    c = build()
    refusing = _Refusing()
    with pytest.raises(RuntimeError, match="cannot undo") as raised:
        Checkpointer(c, tmp_path, physics=P, extra=[refusing]).restore()
    assert isinstance(raised.value.__cause__, ValueError)
    assert str(raised.value.__cause__) == "refused"
    assert len(refusing.calls) == 2


def test_restore_refuses_a_component_this_run_lacks(tmp_path, build):
    b = _steps(build(), 1)
    Checkpointer(b, tmp_path, physics=P, extra=[_Recording()]).write()

    c = build()
    checkpointer = Checkpointer(c, tmp_path, physics=P)
    _refused(c, checkpointer.restore, ValueError, "'recording' only in the checkpoint")


def test_check_restart_refuses_a_restart_json_it_did_not_write(tmp_path):
    """pulse's CLI writes ``restart.json`` too, namespaced under ``"mechanics"`` only."""
    (tmp_path / RESTART_META).write_text(json.dumps({"mechanics": {"physics_hash": "x"}}))
    with pytest.raises(ValueError, match="not written by"):
        check_restart(tmp_path, P)


def test_restore_refuses_a_time_not_in_restart_bp(tmp_path, build):
    b = _steps(build(), 1)
    Checkpointer(b, tmp_path, physics=P).write()
    meta_path = tmp_path / RESTART_META
    meta = json.loads(meta_path.read_text())
    meta["simcardemsx"]["t_ms"] = 7.0
    meta_path.write_text(json.dumps(meta))

    c = build()
    checkpointer = Checkpointer(c, tmp_path, physics=P)
    raised = _refused(c, checkpointer.restore, ValueError, r"t = 7\.0 ms")
    assert "[1.0]" in str(raised.value)


def test_restore_without_a_checkpoint_raises(tmp_path, build):
    c = build()
    checkpointer = Checkpointer(c, tmp_path, physics=P)
    _refused(c, checkpointer.restore, FileNotFoundError, re.escape(str(tmp_path / RESTART_META)))


def test_write_refuses_a_pending_step(tmp_path, build):
    c = build("caisplit", crossbridge=True)
    c.backend.begin_step(c.t, DT_MECH)
    assert c.backend.step_pending

    with pytest.raises(RuntimeError, match="pending"):
        Checkpointer(c, tmp_path, physics=P).write()
    assert not (tmp_path / RESTART).exists()
    assert not (tmp_path / RESTART_META).exists()


def _stored_times(folder: Path, name: str) -> list[float]:
    return [
        float(t)
        for t in io4dolfinx.read_timestamps(folder / RESTART, MPI.COMM_WORLD, function_name=name)
    ]


def _names(folder: Path) -> list[str]:
    functions = json.loads((folder / RESTART_META).read_text())["simcardemsx"]["functions"]
    return [name for names in functions.values() for name in names]


def test_writing_twice_at_one_time_adds_no_timestamp(tmp_path, build):
    b = _steps(build(), K)
    checkpointer = Checkpointer(b, tmp_path, physics=P)
    checkpointer.write()
    checkpointer.write()

    assert "mechanics_u" in _names(tmp_path)
    for name in _names(tmp_path):
        assert _stored_times(tmp_path, name) == [K * DT_MECH], name


class _Killed(Exception):
    """The run is killed while it writes ``restart.json``."""


def test_a_checkpoint_killed_before_restart_json_is_not_used(tmp_path, monkeypatch, build):
    a = _steps(build(), N)

    b = _steps(build(), 2)
    killed = Checkpointer(b, tmp_path, physics=P)
    killed.write()
    _steps(b, 1)

    def kill(*args, **kwargs):
        raise _Killed

    monkeypatch.setattr(checkpoint, "write_json", kill)
    with pytest.raises(_Killed):
        killed.write()
    monkeypatch.undo()
    # restart.bp holds t = 3 for every function, but restart.json still names t = 2.
    assert json.loads((tmp_path / RESTART_META).read_text())["simcardemsx"]["t_ms"] == 2.0
    for name in _names(tmp_path):
        assert _stored_times(tmp_path, name) == [2.0, 3.0], name

    c = build()
    checkpointer = Checkpointer(c, tmp_path, physics=P)
    assert checkpointer.restore() == 2.0
    _steps(c, 1)
    at_3 = _coupled_state(c)
    checkpointer.write()
    _steps(c, N - 3)
    _assert_same_state(_coupled_state(a), _coupled_state(c))
    # The checkpoint at t = 3 was already complete in restart.bp: only restart.json moved.
    assert json.loads((tmp_path / RESTART_META).read_text())["simcardemsx"]["t_ms"] == 3.0
    for name in _names(tmp_path):
        assert _stored_times(tmp_path, name) == [2.0, 3.0], name

    d = build()
    assert Checkpointer(d, tmp_path, physics=P).restore() == 3.0
    _assert_same_state(_coupled_state(d), at_3)
    _steps(d, N - 3)
    _assert_same_state(_coupled_state(a), _coupled_state(d))


def test_history_records_every_process(tmp_path, build):
    b = _steps(build(), 1)
    first = Checkpointer(b, tmp_path, physics=P)
    assert first.history == [first.provenance]
    assert {"git_commit", "versions", "n_ranks", "utc"} <= set(first.provenance)
    first.write()

    c = build()
    second = Checkpointer(c, tmp_path, physics=P)
    second.restore()
    assert len(second.history) == 2
    assert second.history[1] is second.provenance
    _steps(c, 1)
    second.write()

    meta = json.loads((tmp_path / RESTART_META).read_text())["simcardemsx"]
    assert meta["provenance"] == json.loads(json.dumps(second.provenance))
    assert meta["history"] == json.loads(json.dumps([first.provenance, second.provenance]))
    assert meta["physics_hash"] == physics_hash(P)
    assert meta["t_ms"] == 2.0


def test_extra_components_and_their_sidecars(tmp_path, build):
    """An extra component's metadata comes back exactly as written, without the
    checkpointer's keys; its sidecar is written before ``restart.json`` and read after
    ``load_restart``."""
    b = _steps(build(), 1)
    written = _Recording(metadata={"rows": 4})
    Checkpointer(b, tmp_path, physics=P, extra=[written]).write()
    assert written.calls == [("write_sidecar", tmp_path, 1.0, False)]

    c = build()
    read = _Recording()
    Checkpointer(c, tmp_path, physics=P, extra=[read]).restore()
    assert read.calls == [("load_restart", {"rows": 4}), ("read_sidecar", tmp_path, 1.0)]


def test_checkpointer_refuses_shared_names_and_reserved_keys(tmp_path, build):
    c = build()
    v = c.ep_solver.pde.v

    with pytest.raises(ValueError, match="'v'"):
        Checkpointer(c, tmp_path, physics=P, extra=[_Recording(functions=[("v", v)])])
    with pytest.raises(ValueError, match="namespace 'ep'"):
        Checkpointer(c, tmp_path, physics=P, extra=[_Recording(namespace="ep")])
    for key in ("physics_hash", "functions", "provenance", "history"):
        with pytest.raises(ValueError, match=key):
            Checkpointer(c, tmp_path, physics=P, extra=[_Recording(metadata={key: 1})])


def test_physics_hash_is_of_the_sorted_dict():
    assert physics_hash({"a": 1, "b": [1, 2]}) == physics_hash({"b": [1, 2], "a": 1})
    assert physics_hash({"a": 1}) != physics_hash({"a": 2})
    # Values JSON cannot hold are hashed as their str.
    assert physics_hash({"path": Path("x")}) == physics_hash({"path": "x"})


def test_write_json_replaces_the_file_whole(tmp_path):
    path = tmp_path / "run.json"
    write_json(path, {"a": 1})
    write_json(path, {"b": 2.0})
    assert json.loads(path.read_text()) == {"b": 2.0}
    # A document that fails to serialize leaves the old file, and no temporary one.
    with pytest.raises(TypeError):
        write_json(path, {"c": object()})
    assert json.loads(path.read_text()) == {"b": 2.0}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run.json"]
