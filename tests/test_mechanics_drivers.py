"""Tests for :mod:`simcardemsx.mechanics`: the ``MechanicsDriver`` protocol, the
plain ``Solve`` driver, ``as_driver`` (a problem is wrapped by type, a driver passed
through, anything else refused), that :class:`SimulationController` advances
mechanics through whichever driver it is given rather than calling
``problem.solve()`` itself, and ``Cycle``'s conversion to pulse's SI clock.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

from mpi4py import MPI

import dolfinx
import pytest
from pulse.cycle import CycleController

from simcardemsx.controller import SimulationController
from simcardemsx.mechanics import Cycle, MechanicsDriver, Solve, as_driver


def _unit_cube(n: int) -> dolfinx.mesh.Mesh:
    return dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n)


def test_bare_problem_is_wrapped_in_solve(split_modules, make_ep_solver, make_mechanics):
    modules = split_modules["caisplit"]
    ep_solver = make_ep_solver(modules.ep, _unit_cube(1))
    problem, backend = make_mechanics(modules.mechanics, _unit_cube(1))

    controller = SimulationController(problem, ep_solver, backend, modules, 1.0, 0.1)

    assert isinstance(controller.mechanics, Solve)
    assert controller.mechanics.problem is problem


def test_problem_is_wrapped_even_if_it_looks_like_a_driver(split_modules, make_mechanics):
    """``as_driver`` wraps a problem by its type, not by the protocol check: a
    ``pulse.StaticProblem`` already has a ``problem`` (its ``NonlinearProblem``), so one
    that also had an ``advance`` would pass that check, and would otherwise be taken for
    a driver whose ``problem`` is the ``NonlinearProblem``."""
    problem, _ = make_mechanics(split_modules["caisplit"].mechanics, _unit_cube(1))
    setattr(problem, "advance", lambda t_n, dt: True)
    assert isinstance(problem, MechanicsDriver)

    driver = as_driver(problem)

    assert isinstance(driver, Solve)
    assert driver.problem is problem


def test_driver_is_passed_through_and_anything_else_refused(split_modules, make_mechanics):
    problem, _ = make_mechanics(split_modules["caisplit"].mechanics, _unit_cube(1))
    driver = Solve(problem)

    assert as_driver(driver) is driver
    with pytest.raises(TypeError, match="StaticProblem"):
        as_driver(object())


def test_controller_advances_through_the_driver(split_modules, make_ep_solver, make_mechanics):
    modules = split_modules["caisplit"]
    ep_solver = make_ep_solver(modules.ep, _unit_cube(1))
    problem, backend = make_mechanics(modules.mechanics, _unit_cube(1))

    @dataclass
    class _CountingSolve(Solve):
        calls: list[tuple[float, float]] = field(default_factory=list)

        def advance(self, t_n: float, dt: float) -> bool:
            self.calls.append((t_n, dt))
            return super().advance(t_n, dt)

    driver = _CountingSolve(problem)
    controller = SimulationController(driver, ep_solver, backend, modules, 1.0, 0.5)

    controller.step()
    controller.step()

    assert driver.calls == [(0.0, 1.0), (1.0, 1.0)]


@dataclass
class _RecordingCycleController:
    """Stands in for ``pulse.cycle.CycleController``: records each ``step(t, dt)`` and
    returns ``converged``."""

    problem: object
    converged: bool
    calls: list[tuple[float, float]] = field(default_factory=list)

    def step(self, t: float, dt: float) -> bool:
        self.calls.append((t, dt))
        return self.converged


@pytest.mark.parametrize("converged", [True, False])
def test_cycle_steps_pulse_in_seconds(converged):
    """``Cycle.advance(t_n, dt)``, in the controller's ms, solves pulse's step ending at
    ``t_n + dt`` with both in seconds, and returns what that step returns.

    Everything in ``pulse.cycle`` is SI, and nothing else checks the conversion: a ``dt``
    left in ms steps the Windkessel by 2 s, which still converges.
    """
    stub = _RecordingCycleController(problem=object(), converged=converged)
    driver = Cycle(cast(CycleController, stub))

    assert driver.advance(10.0, 2.0) is converged
    assert len(stub.calls) == 1
    assert stub.calls[0] == pytest.approx((0.012, 0.002), rel=1e-15, abs=0.0)
    assert driver.problem is stub.problem
