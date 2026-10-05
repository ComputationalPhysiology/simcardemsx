"""Gate B2 and Review Focus 1: pulse's five-phase cardiac cycle, driven through the controller.

``pulse.cycle.CycleController`` steps a controlled cavity through preload, isovolumic
contraction, ejection, isovolumic relaxation and filling, switching the cavity's
constraint between phases. pulse checks that order on its own ellipsoid with a
prescribed tension (its ``tests/test_cycle.py``), and its rollback on a failed step.
What is simcardemsx's is :class:`~simcardemsx.mechanics.Cycle`, the driver that puts
the controller's ms on pulse's SI clock, and that the cycle runs, through
:class:`SimulationController`, on the tension of the EP-driven backend.

The setup is :func:`conftest._ellipsoid_ep_mechanics` with ``cycle=True``: the zeta
split on a coarse LV ellipsoid, the ENDO cavity controlled by em_tref7's LV cycle
(physcardems ``configs/elife/em_tref7.toml``) in SI, 2 ms mechanics steps.

- **B2**: with em_tref7's ``Tref`` and ToR-ORd's stimulus moved to the end of diastole,
  one beat runs through the five phases in order, and the cavity volume is constant
  over the isovolumic phases.
- **Review Focus 1**: a mechanics step the cycle cannot solve raises from the
  controller, and the backend has not accepted it.
"""

from dataclasses import dataclass

import beat
import numpy as np
import pytest
from conftest import ELLIPSOID_DT_MS, _ellipsoid_ep_mechanics
from pulse.circulation import mL, mmHg
from pulse.cycle import CycleController, CycleParams, Phase, PrescribedInflow, Windkessel

from simcardemsx.backends import GeneratedActivation
from simcardemsx.controller import SimulationController
from simcardemsx.mechanics import Cycle

DT_EP = 0.05

#: em_tref7's beat length (``[time] pcl_ms``), in s. B2 stops after one, at most.
LV_PERIOD = 0.8

#: em_tref7's end of diastole, in ms: PRELOAD ends here, and B2 starts ToR-ORd's
#: stimulus here, so that contraction follows preload.
T_END_DIASTOLE_MS = 120.0

#: em_tref7's Tref, in kPa: ``[land] scales = { Tref = 7.0 }`` on physcardems'
#: ``LAND_BASE["Tref"] = 120``. Only Tref: em_tref7's other Land changes are not B2's.
EM_TREF7_TREF = 7 * 120.0

#: ToR-ORd has not fired while every point's membrane potential is below this, in mV.
AT_REST_MV = -80.0


def lv_cycle_params() -> CycleParams:
    """em_tref7's LV cycle in SI: ``[circulation.lv]``, its ``windkessel``, and
    :data:`LV_PERIOD`. The same values as pulse's ``tests/test_cycle.py``."""
    return CycleParams(
        t_zero=0.05,
        preload_pressure=500.0,
        t_end_diastole=T_END_DIASTOLE_MS * 1e-3,
        p_end_diastole=1000.0,
        p_fill=500.0,
        period=LV_PERIOD,
        windkessel=Windkessel(
            p_init=9000.0,
            compliance=1.5 * mL / mmHg,
            resistance=1.1 * mmHg / mL,
            characteristic_impedance=0.03 * mmHg / mL,
        ),
        filling=PrescribedInflow(rate=0.046 * mL / 1e-3),
    )


def _coupled(
    modules,
    lv_ellipsoid,
    **kwargs,
) -> tuple[SimulationController, CycleController]:
    """The controller, driving the ellipsoid's ENDO cavity through em_tref7's cycle.

    The cycle's clock starts at t = 0, as the controller's does. No preconditioner
    lag: pulse's cycle gate on this ellipsoid needed 17 retries in isovolumic
    relaxation with a lag of 20.
    """
    ep_solver, problem, backend = _ellipsoid_ep_mechanics(
        modules.ep,
        modules.mechanics,
        lv_ellipsoid,
        circulation=False,
        cycle=True,
        **kwargs,
    )
    cycle = CycleController(problem, {"ENDO": lv_cycle_params()}, preconditioner_lag=None)
    cycle.initialize(0.0)
    controller = SimulationController(
        Cycle(cycle),
        ep_solver,
        backend,
        modules,
        ELLIPSOID_DT_MS,
        DT_EP,
    )
    return controller, cycle


@dataclass
class _Step:
    """One converged step: ``t`` (ms) at its end, the ``phase`` it was solved under,
    the cavity's ``V`` (m³) and ``P`` (Pa), its Newton iterations, and the highest
    membrane potential over EP's points (mV)."""

    t: float
    phase: Phase
    V: float
    P: float
    newton: int
    v_max: float


def _run_one_beat(controller: SimulationController, cycle: CycleController) -> list[_Step]:
    """Step until the cycle enters FILLING, or for one :data:`LV_PERIOD`.

    Each step is labelled by the phase read *before* it: ``CycleController.step``
    transitions after solving, so the phase in force afterwards is the next step's.
    """
    ode = controller.ep_solver.ode
    assert isinstance(ode, beat.odesolver.DolfinODESolver)
    v_index = controller.ode_modules.ep.state_index("v")
    problem = cycle.problem

    newton: list[int] = []
    steps: list[_Step] = []
    for _ in range(round(LV_PERIOD * 1e3 / ELLIPSOID_DT_MS)):
        solved_under = cycle.cycles["ENDO"].phase
        try:
            controller.step(mech_callback=lambda t, i, iterations: newton.append(iterations))
        except RuntimeError as error:
            pytest.fail(
                f"{error}, solved under {solved_under.name}; the last Newton solve took "
                f"{problem.problem.solver.getIterationNumber()} iterations",
            )
        record = cycle.records["ENDO"]
        v_max = float(np.max(ode.values[v_index]))
        steps.append(_Step(controller.t, solved_under, record.V, record.P, newton[-1], v_max))
        if cycle.cycles["ENDO"].phase == Phase.FILLING:
            break
    return steps


@pytest.mark.slow
def test_cycle_phases_through_the_controller(split_modules, lv_ellipsoid):
    """B2: one beat of the EP-driven zeta split runs through the five phases in order,
    and the cavity volume is constant over the steps solved under IVC and under IVR.

    The phases are those the steps were solved under, then the one the run ended in:
    FILLING is entered, not solved under, before the run stops.
    """
    modules = split_modules["zetasplit"]
    controller, cycle = _coupled(modules, lv_ellipsoid, mech_parameters={"Tref": EM_TREF7_TREF})
    ode = controller.ep_solver.ode
    assert isinstance(ode, beat.odesolver.DolfinODESolver)
    # In place: beat's DolfinODESolver integrates with the array it was built with.
    ode.parameters[modules.ep.parameter["i_Stim_Start"], :] = T_END_DIASTOLE_MS

    steps = _run_one_beat(controller, cycle)

    preload = [s for s in steps if s.phase == Phase.PRELOAD]
    assert preload
    assert max(s.v_max for s in preload) < AT_REST_MV, "ToR-ORd fired during PRELOAD"

    phases = [s.phase for s in steps] + [cycle.cycles["ENDO"].phase]
    distinct = [phases[0]] + [b for a, b in zip(phases, phases[1:]) if b != a]
    peak_ivc = max((s.P for s in steps if s.phase == Phase.ISOVOLUMIC_CONTRACTION), default=0.0)
    assert distinct == [
        Phase.PRELOAD,
        Phase.ISOVOLUMIC_CONTRACTION,
        Phase.EJECTION,
        Phase.ISOVOLUMIC_RELAXATION,
        Phase.FILLING,
    ], (
        f"phases {[p.name for p in distinct]} by t = {steps[-1].t} ms; "
        f"peak IVC pressure {peak_ivc:.1f} Pa"
    )

    for phase in (Phase.ISOVOLUMIC_CONTRACTION, Phase.ISOVOLUMIC_RELAXATION):
        volumes = [s.V for s in steps if s.phase == phase]
        assert volumes, f"no step solved under {phase.name}"
        np.testing.assert_allclose(volumes, volumes[0], rtol=1e-6, atol=0, err_msg=phase.name)


def _accepted(controller: SimulationController) -> dict[str, np.ndarray]:
    """Copies of what a converged step changes, from the mechanics solution to EP.

    The backend's accepted step (what ``post_solve`` writes): the states, ``λ``, every
    output and the averaged tension. What reaches EP only after it
    (``TransferPlan.backward``): the ODE solver's missing variables and parameters.
    And the mechanics unknowns, the displacement and the cavity pressure.
    """
    backend = controller.backend
    assert isinstance(backend, GeneratedActivation)
    problem = controller.mechanics.problem
    ode = controller.ep_solver.ode
    assert isinstance(ode, beat.odesolver.DolfinODESolver)
    assert ode.missing_variables is not None
    return {
        "states_prev": backend.states_prev.x.array.copy(),
        "lmbda_prev": backend.lmbda_prev.x.array.copy(),
        **{f"outputs[{name!r}]": f.x.array.copy() for name, f in backend.outputs.items()},
        "active_tension": backend.active_tension.x.array.copy(),
        "ode.missing_variables": ode.missing_variables.copy(),
        "ode.parameters": ode.parameters.copy(),
        "u": problem.u.x.array.copy(),
        "cavity_pressure": problem.cavity_pressures[0].x.array.copy(),
    }


@pytest.mark.slow
def test_failed_cycle_step_leaves_backend_unaccepted(split_modules, lv_ellipsoid):
    """Review Focus 1: a step the cycle cannot solve raises ``RuntimeError`` from the
    controller, and nothing of it has been accepted.

    One converged step comes first, so the backend's states are no longer their
    all-zero initial values: from those, a rollback that zeroed them would pass too.
    The next step is then made infeasible as pulse's own rollback test makes it: one
    Newton iteration to reach an IVC volume of half the current one.
    ``CycleController.step`` returns ``False`` after ``problem.reset_states()``, so
    ``Cycle.advance`` does, and the controller must raise without calling
    ``backend.post_solve()`` or moving anything back to EP.

    The controller then rolls the whole step back, its EP micro-steps included: that is
    its behaviour for any driver, not this one's (gate R2, ``tests/test_rollback.py``).
    """
    controller, cycle = _coupled(split_modules["zetasplit"], lv_ellipsoid)
    controller.step()

    before = _accepted(controller)
    assert np.any(before["states_prev"] != 0.0)
    assert np.any(before["lmbda_prev"] != 1.0)

    cycle.problem.problem.solver.setTolerances(max_it=1)
    endo = cycle.cycles["ENDO"]
    endo.phase = Phase.ISOVOLUMIC_CONTRACTION
    endo.end_dia_vol = 0.5 * endo.volume_n

    with pytest.raises(RuntimeError, match="did not converge"):
        controller.step()

    after = _accepted(controller)
    for name, value in before.items():
        assert np.array_equal(after[name], value), name
