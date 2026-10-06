"""Gate R2: a step that fails rolls the whole coupled system back.

Gate 5's set-up (``tests/test_round_trip.py``): beat EP on the 3x3x3 cube, ToR-ORd
firing on its own stimulus at t = 0, and one element of :class:`GeneratedActivation`
mechanics, coupled through :class:`SimulationController`. The split is the zeta split,
whose λ and ``Zetas``/``Zetaw`` cross back to EP, so EP's ``parameters`` and
``missing_variables`` are rewritten by every step.

The controller snapshots every component at the start of a step. If anything before
``mech_callback`` raises (the EP micro-steps and ``ep_callback``, the forward transfer,
``begin_step``, the solve, ``post_solve``, ``plan.backward()`` or the step counter), or
the driver reports failure, it restores the snapshot; only ``mech_callback`` is outside
the rollback. Every array and metadata value is then bit-identical to before the step
(EP's ``parameters`` and ``missing_variables`` included, whose crossing rows the
transfer plan holds), ``controller.t`` is the step's start and ``controller.t_failed``
its end. A retry then reproduces the uninterrupted run bit for bit. X3
(``test_segregated_backend.py``) and Review Focus 1 (``test_cycle_coupling.py``) check
the backend and the cycle's mechanics; these check everything, EP included.
"""

import re
from dataclasses import dataclass, field

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
from conftest import _assert_same_state, _coupled_state, _FailOnce

from simcardemsx.checkpoint import (
    CycleState,
    EPState,
    Snapshot,
    restore_snapshot,
    take_snapshot,
)
from simcardemsx.controller import SimulationController
from simcardemsx.mechanics import Cycle, Solve

DT_EP = 0.05
DT_MECH = 1.0
SPLIT = "zetasplit"


def _unit_cube(n: int) -> dolfinx.mesh.Mesh:
    return dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n)


@dataclass
class _RecordingSolve(Solve):
    """:class:`Solve`, keeping a copy of ``u`` after each ``advance``."""

    u_after: list[np.ndarray] = field(default_factory=list)

    def advance(self, t_n: float, dt: float) -> bool:
        converged = super().advance(t_n, dt)
        self.u_after.append(self.problem.u.x.array.copy())
        return converged


def _controller(
    split_modules,
    make_ep_solver,
    make_mechanics,
    *,
    scheme: str = "monolithic",
    driver=None,
    dt_mech: float = DT_MECH,
) -> SimulationController:
    """Gate 5's controller. ``driver(problem)`` builds the mechanics driver; without it
    the controller wraps the problem in :class:`Solve`."""
    modules = split_modules[SPLIT]
    ep_solver = make_ep_solver(modules.ep, _unit_cube(3))
    problem, backend = make_mechanics(
        modules.mechanics,
        _unit_cube(1),
        quadrature_degree=2,
        scheme=scheme,
    )
    mechanics = problem if driver is None else driver(problem)
    return SimulationController(mechanics, ep_solver, backend, modules, dt_mech, DT_EP)


def _changed(a, b) -> set[str]:
    """The array keys whose values differ between two coupled states."""
    return {key for key, value in a[0].items() if not np.array_equal(value, b[0][key])}


#: Arrays that every step changes, so a restore that left them alone could not pass.
_MOVING = {"ep/v", "mechanics/mechanics_u", "activation/activation_states_prev"}


def test_snapshot_restore_is_bit_identical(split_modules, make_ep_solver, make_mechanics):
    controller = _controller(split_modules, make_ep_solver, make_mechanics)
    controller.step()
    snapshot = controller.snapshot()
    reference = _coupled_state(controller)

    controller.step()
    controller.step()
    assert _MOVING <= _changed(_coupled_state(controller), reference)

    controller.restore(snapshot)
    _assert_same_state(_coupled_state(controller), reference)
    assert controller.t == 1.0

    # The snapshot is not consumed, or changed by what follows a restore.
    controller.step()
    controller.restore(snapshot)
    _assert_same_state(_coupled_state(controller), reference)


def test_snapshot_before_any_step_restores_ep_as_constructed(
    split_modules,
    make_ep_solver,
    make_mechanics,
):
    """A snapshot at t = 0, before any step, holds EP's crossing rows as constructed,
    which no backend output reproduces (``outputs["lmbda"]`` is 0 then). Restoring it
    after two steps puts them back bit for bit, into beat's own arrays."""
    controller = _controller(split_modules, make_ep_solver, make_mechanics)
    ode = controller.ep_solver.ode
    missing_variables, parameters = ode.missing_variables, ode.parameters
    constructed = missing_variables.copy(), parameters.copy()
    snapshot = controller.snapshot()

    controller.step()
    controller.step()
    assert not np.array_equal(missing_variables, constructed[0])
    assert not np.array_equal(parameters, constructed[1])

    controller.restore(snapshot)
    # In place: beat's inner solver holds these arrays by reference.
    assert ode.missing_variables is missing_variables
    assert ode.parameters is parameters
    assert np.array_equal(missing_variables, constructed[0])
    assert np.array_equal(parameters, constructed[1])
    assert controller.t == 0.0


@pytest.mark.parametrize("scheme", ["monolithic", "segregated", "stabilized"])
def test_failed_step_rolls_everything_back_and_a_retry_matches(
    scheme,
    split_modules,
    make_ep_solver,
    make_mechanics,
):
    inner: list[_RecordingSolve] = []

    def failing(problem):
        inner.append(_RecordingSolve(problem))
        return _FailOnce(inner[0], at_step=2)

    controller = _controller(
        split_modules,
        make_ep_solver,
        make_mechanics,
        scheme=scheme,
        driver=failing,
    )
    controller.step()
    before = _coupled_state(controller)

    with pytest.raises(RuntimeError, match="did not converge"):
        controller.step()

    # The failed step's solve did move u; the rollback undid it, and everything else.
    assert not np.array_equal(inner[0].u_after[-1], before[0]["mechanics/mechanics_u"])
    _assert_same_state(_coupled_state(controller), before)
    assert controller.t == 1.0
    assert controller.t_failed == 2.0

    controller.step()  # the retry
    assert controller.t_failed is None
    controller.step()

    uninterrupted = _controller(split_modules, make_ep_solver, make_mechanics, scheme=scheme)
    for _ in range(3):
        uninterrupted.step()
    _assert_same_state(_coupled_state(controller), _coupled_state(uninterrupted))


def test_interrupt_in_ep_micro_steps_rolls_back(split_modules, make_ep_solver, make_mechanics):
    controller = _controller(split_modules, make_ep_solver, make_mechanics)
    controller.step()
    before = _coupled_state(controller)

    third = controller.ep_step_idx + 3
    v_at_interrupt: list[np.ndarray] = []

    def interrupt(t: float, ep_step_idx: int) -> None:
        if ep_step_idx == third:
            v_at_interrupt.append(controller.ep_solver.pde.v.x.array.copy())
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        controller.step(ep_callback=interrupt)

    assert not np.array_equal(v_at_interrupt[0], before[0]["ep/v"])
    _assert_same_state(_coupled_state(controller), before)
    assert controller.t == 1.0
    assert controller.t_failed == 2.0


def test_failed_first_step_leaves_ep_as_it_was(split_modules, make_ep_solver, make_mechanics):
    """Before the first accepted step, EP's arrays are not the backend's outputs: EP's
    ``lmbda`` is the module's default, 1, and the backend's ``outputs["lmbda"]`` is 0,
    since nothing has moved back yet. A rollback of the first step must leave EP's
    arrays as they were, not rewrite them from those outputs, and the retry must
    match the uninterrupted run."""
    controller = _controller(
        split_modules,
        make_ep_solver,
        make_mechanics,
        driver=lambda problem: _FailOnce(Solve(problem), at_step=1),
    )
    before = _coupled_state(controller)
    lmbda = controller.ode_modules.ep.parameter["lmbda"]
    assert np.all(before[0]["ep/parameters"][lmbda] == 1.0)

    with pytest.raises(RuntimeError, match="did not converge"):
        controller.step()
    _assert_same_state(_coupled_state(controller), before)
    assert controller.t == 0.0
    assert controller.t_failed == 1.0

    controller.step()
    controller.step()
    uninterrupted = _controller(split_modules, make_ep_solver, make_mechanics)
    uninterrupted.step()
    uninterrupted.step()
    _assert_same_state(_coupled_state(controller), _coupled_state(uninterrupted))


def test_t_failed_is_where_the_step_would_have_ended(
    split_modules,
    make_ep_solver,
    make_mechanics,
):
    """``t_failed`` (and the ``RuntimeError``'s end time) is the ``t`` that an
    uninterrupted run reaches after that step, bit for bit: the EP loop's
    ``t_n + n * dt_ep``, not ``t_n + dt_mech``. The examples record it, and at
    dt_mech 0.15 and dt_ep 0.05 the two differ in the last bit on the first step."""
    dt_mech = 0.15
    assert 0.0 + 3 * DT_EP != 0.0 + dt_mech  # 0.15000000000000002 and 0.15

    uninterrupted = _controller(split_modules, make_ep_solver, make_mechanics, dt_mech=dt_mech)
    uninterrupted.step()

    controller = _controller(
        split_modules,
        make_ep_solver,
        make_mechanics,
        dt_mech=dt_mech,
        driver=lambda problem: _FailOnce(Solve(problem), at_step=1),
    )
    with pytest.raises(RuntimeError, match=re.escape(f"to t = {uninterrupted.t!r}")):
        controller.step()
    assert controller.t_failed == uninterrupted.t
    assert controller.t == 0.0


def test_a_failure_moving_values_back_to_ep_rolls_back(
    split_modules,
    make_ep_solver,
    make_mechanics,
    monkeypatch,
):
    """An interrupt after ``post_solve``, while the outputs move back to EP, rolls the
    step back too: only ``mech_callback`` comes after the step is complete."""
    controller = _controller(split_modules, make_ep_solver, make_mechanics)
    controller.step()
    before = _coupled_state(controller)
    backward = controller.plan.backward

    def interrupted() -> None:
        backward()
        raise KeyboardInterrupt

    monkeypatch.setattr(controller.plan, "backward", interrupted)
    with pytest.raises(KeyboardInterrupt):
        controller.step()
    monkeypatch.undo()

    _assert_same_state(_coupled_state(controller), before)
    assert controller.t == 1.0
    assert controller.t_failed == 2.0


def test_a_failing_rollback_keeps_both_errors(
    split_modules,
    make_ep_solver,
    make_mechanics,
    monkeypatch,
):
    """If the restore itself raises, its error propagates with the step's as its cause."""
    controller = _controller(
        split_modules,
        make_ep_solver,
        make_mechanics,
        driver=lambda problem: _FailOnce(Solve(problem), at_step=1),
    )

    def broken(components, snapshot: Snapshot) -> None:
        raise OSError("the restore failed")

    monkeypatch.setattr("simcardemsx.controller.restore_snapshot", broken)
    with pytest.raises(OSError, match="the restore failed") as info:
        controller.step()
    assert isinstance(info.value.__cause__, RuntimeError)
    assert "did not converge" in str(info.value.__cause__)


def test_restore_refuses_a_snapshot_of_other_components(
    split_modules,
    make_ep_solver,
    make_mechanics,
):
    """A snapshot whose namespaces or function names differ from the components' is
    refused with ``ValueError`` naming the difference, before anything is written."""
    controller = _controller(split_modules, make_ep_solver, make_mechanics)
    controller.step()
    snapshot = controller.snapshot()
    controller.step()
    state = _coupled_state(controller)

    without_backend = Snapshot(
        {k: v for k, v in snapshot.arrays.items() if k != "activation"},
        {k: v for k, v in snapshot.metadata.items() if k != "activation"},
    )
    with pytest.raises(ValueError, match="activation"):
        controller.restore(without_backend)

    (_, u), *rest = snapshot.arrays["mechanics"]
    renamed = Snapshot(
        {**snapshot.arrays, "mechanics": [("mechanics_x", u), *rest]},
        snapshot.metadata,
    )
    with pytest.raises(ValueError, match="mechanics_x"):
        controller.restore(renamed)

    _assert_same_state(_coupled_state(controller), state)


def test_load_restart_refuses_another_configuration(
    split_modules,
    make_ep_solver,
    make_mechanics,
):
    controller = _controller(split_modules, make_ep_solver, make_mechanics)
    ep = controller.components()[0]
    assert isinstance(ep, EPState)

    # The same state names in another order are accepted: a checkpoint is read by name,
    # and another process may generate the states in another order. Another set is not.
    metadata = ep.restart_metadata()
    reordered = {**metadata, "state_names": metadata["state_names"][::-1]}
    ep.load_restart(ep.restart_functions(), reordered)
    other = {**metadata, "state_names": [*metadata["state_names"][:-1], "not_a_state"]}
    with pytest.raises(ValueError, match="state names"):
        ep.load_restart(ep.restart_functions(), other)

    rows = controller.plan.restart_functions()
    with pytest.raises(ValueError, match="transfer_parameter_lmbda"):
        controller.plan.load_restart(rows[:-1], {})

    own = controller.restart_metadata()
    for key in ("dt_mech_ms", "dt_ep_ms"):
        with pytest.raises(ValueError, match=key.removesuffix("_ms")):
            controller.load_restart([], {**own, key: 2 * own[key]})


@dataclass
class _CycleStub:
    """Stands in for ``pulse.cycle.CycleController``: a problem and a state dict."""

    problem: object
    state: dict = field(default_factory=lambda: {"initialized": True})

    def state_dict(self) -> dict:
        return dict(self.state)

    def load_state_dict(self, state) -> None:
        self.state = dict(state)


def test_components_and_controller_metadata(split_modules, make_ep_solver, make_mechanics):
    controller = _controller(split_modules, make_ep_solver, make_mechanics)
    names = sorted(controller.ode_modules.ep.state, key=controller.ode_modules.ep.state.__getitem__)

    components = controller.components()
    assert [c.namespace for c in components] == [
        "ep",
        "transfer",
        "mechanics",
        "activation",
        "simcardemsx",
    ]
    assert components[1] is controller.plan
    assert components[3] is controller.backend
    assert components[4] is controller
    assert [name for name, _ in components[0].restart_functions()] == [
        "v",
        *(f"state_{name}" for name in names),
    ]
    assert components[0].restart_metadata()["state_names"] == names
    # The rows of beat's arrays that backward() writes, for the zeta split.
    assert [name for name, _ in controller.plan.restart_functions()] == [
        "transfer_missing_Zetas",
        "transfer_missing_Zetaw",
        "transfer_parameter_lmbda",
    ]
    assert controller.plan.restart_metadata() == {}
    assert controller.restart_functions() == []
    assert controller.t_failed is None

    controller.step()
    assert controller.restart_metadata() == {
        "t_ms": 1.0,
        "ep_step_idx": 20,
        "mech_step_idx": 1,
        "dt_mech_ms": DT_MECH,
        "dt_ep_ms": DT_EP,
    }

    stubs: list[_CycleStub] = []

    def cycle(problem):
        stubs.append(_CycleStub(problem))
        return Cycle(stubs[0])  # type: ignore[arg-type]

    with_cycle = _controller(split_modules, make_ep_solver, make_mechanics, driver=cycle)
    components = with_cycle.components()
    assert [c.namespace for c in components] == [
        "ep",
        "transfer",
        "mechanics",
        "cycle",
        "activation",
        "simcardemsx",
    ]
    state = components[3]
    assert isinstance(state, CycleState)
    assert state.restart_functions() == []
    assert state.restart_metadata() == {"initialized": True}
    state.load_restart([], {"initialized": False})
    assert stubs[0].state == {"initialized": False}


@dataclass
class _Component:
    """A minimal ``Checkpointable``: given Functions, and a record of each load."""

    namespace: str
    functions: list
    loaded: list = field(default_factory=list)

    def restart_functions(self) -> list:
        return list(self.functions)

    def restart_metadata(self) -> dict:
        return {"names": [name for name, _ in self.functions]}

    def load_restart(self, functions, metadata) -> None:
        self.loaded.append(metadata)


def test_snapshot_guards():
    """Two components may not share a namespace, and a snapshot is refused, before
    anything is written, by components whose names are in another order or whose
    arrays have another size."""
    mesh = _unit_cube(1)
    V = dolfinx.fem.functionspace(mesh, ("P", 1))
    a, b = dolfinx.fem.Function(V), dolfinx.fem.Function(V)
    a.x.array[:] = 1.0
    component = _Component("x", [("a", a), ("b", b)])
    with pytest.raises(ValueError, match="namespace 'x'"):
        take_snapshot([component, _Component("x", [])])

    snapshot = take_snapshot([component])
    a.x.array[:] = 2.0
    with pytest.raises(ValueError, match="another order"):
        restore_snapshot([_Component("x", [("b", b), ("a", a)])], snapshot)
    larger = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("P", 2)))
    with pytest.raises(ValueError, match="'b'"):
        restore_snapshot([_Component("x", [("a", a), ("b", larger)])], snapshot)
    assert np.all(a.x.array == 2.0)

    restore_snapshot([component], snapshot)
    assert np.all(a.x.array == 1.0)
    assert component.loaded == [{"names": ["a", "b"]}]
