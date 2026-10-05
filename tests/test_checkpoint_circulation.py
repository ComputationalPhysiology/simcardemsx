"""Gates R1 and R2 on the closed-loop circulation (BDF2) and on the five-phase cycle.

The harness is :func:`conftest._ellipsoid_ep_mechanics`: the zeta split on the coarse LV
ellipsoid, ToR-ORd firing everywhere at t = 0, 2 ms mechanics steps. The state that
:func:`conftest._coupled_state` compares then includes what the fast cases lack: the
circuit's states (and, under BDF2, the history of its previous step and the step count),
and the cycle controller's phases and records.

- **R1, circulation.** BDF2 needs two earlier steps, so a restart that forgot the
  history would step with backward Euler and differ. ``problem._circulation_steps`` must
  be at least 1 at the checkpoint, or the case does not exercise the history.
- **R1, cycle.** The checkpoint is written in PRELOAD and the run continues into
  ISOVOLUMIC_CONTRACTION, so the restored cycle must carry its phase clock across a
  switch. The controller is passed to :class:`SimulationController` as a bare
  :class:`~simcardemsx.mechanics.Cycle`, which is what makes ``components()`` include the
  cycle. The restored run does *not* call ``cycle.initialize``: restoring does.
- **R2.** A step that fails after the solve moved ``u`` is rolled back bit for bit, and
  its retry reproduces the uninterrupted run.

``refresh_pending``: pulse's ``CycleController.load_state_dict`` never clears a pending
refresh (``pending or saved``), and a fresh controller starts with one pending. So right
after the restore the cycle metadata says ``refresh_pending: True`` where the
checkpointed run's says ``False``. That is pulse's documented design (the flag only asks
for a fresh preconditioner), so at that point the cycle metadata is compared without it
and the restored run is asserted to carry ``True``. The first step refreshes, and after
it the whole state, the flag included, equals the uninterrupted run's.
"""

import dataclasses
from pathlib import Path

import pytest
from conftest import (
    ELLIPSOID_DT_MS,
    ELLIPSOID_PERIOD_MS,
    _assert_same_state,
    _coupled_state,
    _ellipsoid_ep_mechanics,
    _FailOnce,
)
from pulse.cycle import CycleController, CycleParams, Phase
from test_cycle_coupling import lv_cycle_params

from simcardemsx.checkpoint import Checkpointer
from simcardemsx.controller import SimulationController
from simcardemsx.mechanics import CirculationClock, Cycle

DT_EP = 0.05

pytestmark = pytest.mark.slow


def _circulation(modules, lv_ellipsoid, *, scheme: str, fail_at: int | None = None):
    """``(controller, problem)``: the closed loop under :class:`CirculationClock`,
    wrapped in :class:`_FailOnce` when ``fail_at`` is given."""
    ep_solver, problem, backend = _ellipsoid_ep_mechanics(
        modules.ep,
        modules.mechanics,
        lv_ellipsoid,
        circulation=True,
        circulation_scheme=scheme,
    )
    driver = CirculationClock(
        problem,
        beat_phase=problem.circulation_missing["beat_phase"],
        period=ELLIPSOID_PERIOD_MS,
    )
    mechanics = driver if fail_at is None else _FailOnce(driver, at_step=fail_at)
    controller = SimulationController(
        mechanics,
        ep_solver,
        backend,
        modules,
        ELLIPSOID_DT_MS,
        DT_EP,
    )
    return controller, problem


def _cycle_params() -> CycleParams:
    """B2's LV cycle, with PRELOAD shortened to end at 8 ms (``t_zero`` 4 ms)."""
    return dataclasses.replace(lv_cycle_params(), t_zero=0.004, t_end_diastole=0.008)


def _cycle(modules, lv_ellipsoid, *, initialize: bool):
    """``(controller, cycle)``. ``Cycle`` is passed unwrapped, so that ``components()``
    finds the cycle state. A restored run leaves ``initialize`` to the restore."""
    ep_solver, problem, backend = _ellipsoid_ep_mechanics(
        modules.ep,
        modules.mechanics,
        lv_ellipsoid,
        circulation=False,
        cycle=True,
    )
    cycle = CycleController(problem, {"ENDO": _cycle_params()}, preconditioner_lag=None)
    if initialize:
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


def _steps(controller: SimulationController, n: int) -> None:
    for _ in range(n):
        controller.step()


def _without_refresh_pending(state):
    arrays, metadata = state
    cycle = {k: v for k, v in metadata["cycle"].items() if k != "refresh_pending"}
    return arrays, {**metadata, "cycle": cycle}


def test_restart_matches_the_uninterrupted_run_circulation_bdf2(
    tmp_path: Path,
    split_modules,
    lv_ellipsoid,
):
    modules = split_modules["zetasplit"]
    N, K = 6, 3
    physics = {"split": "zetasplit", "circulation": "regazzoni2020", "scheme": "bdf2"}

    def build():
        return _circulation(modules, lv_ellipsoid, scheme="bdf2")

    a, _ = build()
    _steps(a, N)

    b, problem_b = build()
    _steps(b, K)
    # The BDF2 history is in use at the checkpoint, or this would be backward Euler.
    assert problem_b._circulation_steps >= 1
    Checkpointer(b, tmp_path, physics=physics).write()
    at_checkpoint = _coupled_state(b)

    c, problem_c = build()
    assert Checkpointer(c, tmp_path, physics=physics).restore() == K * ELLIPSOID_DT_MS
    assert problem_c._circulation_steps == problem_b._circulation_steps
    _assert_same_state(_coupled_state(c), at_checkpoint)

    _steps(c, N - K)
    _assert_same_state(_coupled_state(c), _coupled_state(a))


def test_restart_matches_the_uninterrupted_run_cycle(tmp_path: Path, split_modules, lv_ellipsoid):
    modules = split_modules["zetasplit"]
    N, K = 8, 3
    physics = {"split": "zetasplit", "cycle": "lv", "scheme": "monolithic"}

    a, cycle_a = _cycle(modules, lv_ellipsoid, initialize=True)
    _steps(a, K)
    assert cycle_a.records["ENDO"].phase == Phase.PRELOAD
    _steps(a, N - K)
    assert cycle_a.records["ENDO"].phase == Phase.ISOVOLUMIC_CONTRACTION

    b, _ = _cycle(modules, lv_ellipsoid, initialize=True)
    _steps(b, K)
    Checkpointer(b, tmp_path, physics=physics).write()
    at_checkpoint = _coupled_state(b)

    # No initialize: the restore sets `initialized`, the phases and the records.
    c, cycle_c = _cycle(modules, lv_ellipsoid, initialize=False)
    assert Checkpointer(c, tmp_path, physics=physics).restore() == K * ELLIPSOID_DT_MS
    assert cycle_c.records["ENDO"].phase == Phase.PRELOAD
    # See the module docstring: refresh_pending is `pending or saved` in pulse, so the
    # restored controller still asks for a fresh preconditioner. Everything else equal.
    restored = _coupled_state(c)
    assert restored[1]["cycle"]["refresh_pending"] is True
    _assert_same_state(_without_refresh_pending(restored), _without_refresh_pending(at_checkpoint))

    # The first step refreshes the preconditioner, so by now the flag agrees as well.
    _steps(c, N - K)
    _assert_same_state(_coupled_state(c), _coupled_state(a))


def test_failed_step_rolls_back_under_the_circulation(split_modules, lv_ellipsoid):
    modules = split_modules["zetasplit"]
    controller, _ = _circulation(modules, lv_ellipsoid, scheme="backward_euler", fail_at=2)
    controller.step()
    before = _coupled_state(controller)

    with pytest.raises(RuntimeError, match="did not converge"):
        controller.step()
    _assert_same_state(_coupled_state(controller), before)
    assert controller.t == ELLIPSOID_DT_MS
    assert controller.t_failed == 2 * ELLIPSOID_DT_MS

    controller.step()  # the retry
    assert controller.t_failed is None
    controller.step()

    uninterrupted, _ = _circulation(modules, lv_ellipsoid, scheme="backward_euler")
    _steps(uninterrupted, 3)
    _assert_same_state(_coupled_state(controller), _coupled_state(uninterrupted))
