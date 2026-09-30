"""Gates A1 and A2: a closed-loop circulation driven through the controller.

pulse solves Regazzoni's circuit in the same Newton system as the displacement
(``pulse.circulation.GotranxCirculation``): the LV's volume is a circuit state tied
to the deformed cavity, and its pressure is the cavity's Lagrange multiplier. The
coupling's rows and Jacobian blocks are covered by pulse's own
``tests/test_circulation_coupling.py`` and are not re-tested here. What is
simcardemsx's is the circuit's clock (:class:`~simcardemsx.mechanics.CirculationClock`),
and that the closed loop -- the 3D LV and the rest of the circuit -- is driven,
through :class:`SimulationController`, by the tension of the EP-driven backend.

The setup is :func:`conftest._ellipsoid_ep_mechanics`: the zeta split on a coarse LV
ellipsoid, ToR-ORd firing everywhere at t = 0, the circuit started from the unloaded
cavity volume, 20 mechanics steps of 2 ms.

- **A1**: the total blood volume -- the circuit's compartments plus the *3D* LV's
  cavity volume -- is conserved. Summed over the circuit's own states it is linear, and
  backward Euler preserves a linear invariant exactly, so that sum is conserved
  whatever the mechanics does. With the deformed cavity in place of the circuit's
  ``V_LV`` state, it is conserved only if the cavity follows the circuit, so drift is a
  coupling error, not discretization.
- **A2**: the circuit sees the backend. Wherever the backend's mean tension is above
  0.01 kPa, the LV pressure is strictly higher than in the same run with
  ``tension_scale = 0``.

The ellipsoid's own fibres are checked first: they must have unit length at the points
the backend and the material read them at, or λ is not the fibre stretch.
"""

from dataclasses import dataclass

from mpi4py import MPI

import basix.ufl
import dolfinx
import numpy as np
import pytest
import ufl
from conftest import (
    ELLIPSOID_DT_MS,
    ELLIPSOID_PERIOD_MS,
    ELLIPSOID_QUADRATURE_DEGREE,
    _ellipsoid_ep_mechanics,
)
from pulse.circulation import GotranxCirculation, mL

from simcardemsx.controller import SimulationController
from simcardemsx.mechanics import CirculationClock

DT_EP = 0.05
NUM_STEPS = 20

#: Mean active tension, in kPa, above which A2 requires the pressure to be higher.
TENSION_ON = 0.01


@dataclass
class _Run:
    """What a run records. ``total`` and ``V_LV`` (mL) are at t = 0 and after each
    step; ``tension`` (mean ``active_tension``, kPa) and ``pressure``
    (``problem.cavity_pressures[0]``, Pa) after each step."""

    total: np.ndarray
    V_LV: np.ndarray
    tension: np.ndarray
    pressure: np.ndarray


def _total_blood_volume(problem) -> float:
    """The total blood volume of the closed loop, in mL: the 3D LV plus the circuit.

    The circulation package's definition (``Regazzoni2020.compute_volumes``) -- the
    four chamber volumes, plus each vessel compartment's compliance times its pressure
    (states in mL and mmHg, compliances in mL/mmHg) -- with the LV's volume taken from
    the deformed ENDO cavity, ``V(u)``, instead of the circuit's ``V_LV`` state.
    """
    circuit = problem.circulation
    assert isinstance(circuit, GotranxCirculation)
    assert circuit.parameters is not None
    c = circuit.parameters
    y = {
        name: float(state.x.array[0])
        for name, state in zip(circuit.state_names, problem.circulation_states)
    }
    comm = problem.geometry.mesh.comm
    cavity = comm.allreduce(problem.geometry.volume("ENDO", u=problem.u), op=MPI.SUM) / mL
    return (
        y["V_LA"]
        + cavity
        + y["V_RA"]
        + y["V_RV"]
        + c["C_AR_SYS"] * y["p_AR_SYS"]
        + c["C_VEN_SYS"] * y["p_VEN_SYS"]
        + c["C_AR_PUL"] * y["p_AR_PUL"]
        + c["C_VEN_PUL"] * y["p_VEN_PUL"]
    )


def _run(modules, ep_solver, problem, backend) -> _Run:
    """Run :data:`NUM_STEPS` controller steps, the circuit clocked by
    :class:`CirculationClock` in seconds at the circuit's own beat length."""
    circuit = problem.circulation
    assert circuit is not None
    i_LV = list(circuit.state_names).index("V_LV")
    clock = CirculationClock(
        problem,
        beat_phase=problem.circulation_missing["beat_phase"],
        period=ELLIPSOID_PERIOD_MS,
    )
    controller = SimulationController(clock, ep_solver, backend, modules, ELLIPSOID_DT_MS, DT_EP)

    total = [_total_blood_volume(problem)]
    V_LV = [float(problem.circulation_states[i_LV].x.array[0])]
    tension, pressure = [], []
    for _ in range(NUM_STEPS):
        controller.step()
        total.append(_total_blood_volume(problem))
        V_LV.append(float(problem.circulation_states[i_LV].x.array[0]))
        tension.append(float(np.mean(backend.active_tension.x.array)))
        pressure.append(float(problem.cavity_pressures[0].x.array[0]))
    return _Run(*(np.array(x) for x in (total, V_LV, tension, pressure)))


@pytest.fixture(scope="module")
def coupled(split_modules, lv_ellipsoid):
    """``(ep_solver, problem, backend)``: the zeta split, closed by the circuit.

    Shared by the clock test and :func:`full_tension`. The clock test only sets the
    circuit's clock constants on it, with ``solve`` replaced, and every
    :meth:`CirculationClock.advance` sets all three again before solving, so the run
    is the same whichever test comes first.
    """
    modules = split_modules["zetasplit"]
    return _ellipsoid_ep_mechanics(modules.ep, modules.mechanics, lv_ellipsoid, circulation=True)


@pytest.fixture(scope="module")
def full_tension(split_modules, coupled) -> _Run:
    """The run with the backend's own tension, shared by A1 and (as its first run) A2."""
    return _run(split_modules["zetasplit"], *coupled)


@pytest.mark.slow
def test_lv_ellipsoid_fibres_are_unit_where_they_are_read(split_modules, lv_ellipsoid):
    """The fixture's fibre field has unit length at the backend's quadrature points.

    ``GeneratedActivation``'s λ = sqrt(f0 · C f0) is the fibre stretch only for a unit
    f0, and Holzapfel-Ogden's fibre term is off while f0 · C f0 < 1. A field that is unit
    at mesh nodes need not be between them: P1 fibres on this ellipsoid, whose fibre
    angle turns through 120 degrees across a wall about one element thick, are 0.32 to
    0.95 long at these points.

    Since the fibres live on those points, ``_ellipsoid_ep_mechanics`` refuses any other
    quadrature degree.
    """
    mesh = lv_ellipsoid.mesh
    Q = dolfinx.fem.functionspace(
        mesh,
        basix.ufl.quadrature_element(
            mesh.basix_cell(),
            value_shape=(),
            degree=ELLIPSOID_QUADRATURE_DEGREE,
        ),
    )
    length = dolfinx.fem.Function(Q)
    f0 = lv_ellipsoid.f0
    length.interpolate(
        dolfinx.fem.Expression(ufl.sqrt(ufl.inner(f0, f0)), Q.element.interpolation_points),
    )
    np.testing.assert_allclose(length.x.array, 1.0, rtol=1e-12, atol=0)

    modules = split_modules["zetasplit"]
    with pytest.raises(ValueError, match="quadrature_degree"):
        _ellipsoid_ep_mechanics(
            modules.ep,
            modules.mechanics,
            lv_ellipsoid,
            circulation=False,
            quadrature_degree=ELLIPSOID_QUADRATURE_DEGREE + 1,
        )


@pytest.mark.slow
def test_ellipsoid_builder_refuses_circulation_with_cycle(split_modules, lv_ellipsoid):
    """The ENDO cavity is either a chamber of the circuit or controlled by the
    five-phase cycle; ``_ellipsoid_ep_mechanics`` refuses both at once, before it
    builds anything."""
    modules = split_modules["zetasplit"]
    with pytest.raises(ValueError, match="circulation and cycle are exclusive"):
        _ellipsoid_ep_mechanics(
            modules.ep,
            modules.mechanics,
            lv_ellipsoid,
            circulation=True,
            cycle=True,
        )


@pytest.mark.slow
def test_circulation_clock_units_and_guards(coupled, split_modules, lv_ellipsoid, monkeypatch):
    """The clock sets the circuit's time, step and beat phase in the circuit's own unit
    before it solves, and returns what the solve returns.

    ``solve`` is replaced by a recorder. The clock's contract is what it has set by
    the time it solves; the circuit stepped by 2 *s* (``time_unit="ms"`` on a circuit
    written in seconds) would be meaningless, and would leave the shared problem in
    whatever state a failed solve leaves.
    """
    _, problem, _ = coupled
    beat_phase = problem.circulation_missing["beat_phase"]
    seen: list[tuple[float, float, float]] = []

    def solve() -> bool:
        seen.append(
            (
                float(problem.circulation_time.value),
                float(problem.circulation_dt.value),
                float(beat_phase.value),
            ),
        )
        return False

    monkeypatch.setattr(problem, "solve", solve)

    seconds = CirculationClock(problem, time_unit="s", beat_phase=beat_phase, period=800.0)
    assert seconds.advance(10.0, 2.0) is False
    assert seen[-1] == pytest.approx((0.012, 0.002, 0.012))
    seconds.advance(800.0, 2.0)
    assert seen[-1][2] == pytest.approx(0.002)

    milliseconds = CirculationClock(problem, time_unit="ms", beat_phase=beat_phase, period=800.0)
    milliseconds.advance(10.0, 2.0)
    assert seen[-1] == pytest.approx((12.0, 2.0, 12.0))
    assert len(seen) == 3

    with pytest.raises(ValueError, match="together"):
        CirculationClock(problem, beat_phase=beat_phase)
    with pytest.raises(ValueError, match="together"):
        CirculationClock(problem, period=800.0)
    with pytest.raises(ValueError, match="'min'"):
        CirculationClock(problem, time_unit="min")
    # pulse compiled the circuit's form against the Constants in circulation_missing;
    # any other Constant, even one on the same mesh with the same value, is never read.
    stray = dolfinx.fem.Constant(lv_ellipsoid.mesh, dolfinx.default_scalar_type(0.0))
    with pytest.raises(ValueError, match=r"circulation_missing \(keys \['beat_phase'\]\)"):
        CirculationClock(problem, beat_phase=stray, period=800.0)

    modules = split_modules["zetasplit"]
    _, open_problem, _ = _ellipsoid_ep_mechanics(
        modules.ep,
        modules.mechanics,
        lv_ellipsoid,
        circulation=False,
    )
    with pytest.raises(ValueError, match="circulation"):
        CirculationClock(open_problem)


@pytest.mark.slow
def test_closed_loop_conserves_blood_volume(full_tension):
    """A1: the total blood volume, the 3D LV's included, after every step equals its
    initial value.

    ``V_LV`` has to move for this to say anything: the circuit starts at the unloaded
    cavity volume, below its own ``V_LV``, so the LV fills from the atrium.
    """
    np.testing.assert_allclose(full_tension.total, full_tension.total[0], rtol=1e-8, atol=0)
    assert abs(full_tension.V_LV[-1] - full_tension.V_LV[0]) > 1e-3


@pytest.mark.slow
def test_circuit_sees_the_backends_tension(split_modules, lv_ellipsoid, full_tension):
    """A2: with the backend's tension, the LV pressure is strictly higher than without
    it, at every step where the mean tension is above :data:`TENSION_ON`."""
    modules = split_modules["zetasplit"]
    ep_solver, problem, backend = _ellipsoid_ep_mechanics(
        modules.ep,
        modules.mechanics,
        lv_ellipsoid,
        circulation=True,
        tension_scale=dolfinx.fem.Constant(lv_ellipsoid.mesh, 0.0),
    )
    no_tension = _run(modules, ep_solver, problem, backend)

    active = full_tension.tension > TENSION_ON
    assert active.any(), full_tension.tension
    assert np.all(full_tension.pressure[active] > no_tension.pressure[active]), (
        full_tension.pressure[active],
        no_tension.pressure[active],
    )
