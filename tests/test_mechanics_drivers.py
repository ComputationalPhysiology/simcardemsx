"""Tests for :mod:`simcardemsx.mechanics`: the ``MechanicsDriver`` protocol, the
plain ``Solve`` driver, ``as_driver``, and that :class:`SimulationController`
advances mechanics through whichever driver it is given rather than calling
``problem.solve()`` itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mpi4py import MPI

import dolfinx

from simcardemsx.controller import SimulationController
from simcardemsx.mechanics import Solve


def _unit_cube(n: int) -> dolfinx.mesh.Mesh:
    return dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n)


def test_bare_problem_is_wrapped_in_solve(split_modules, make_ep_solver, make_mechanics):
    modules = split_modules["caisplit"]
    ep_solver = make_ep_solver(modules.ep, _unit_cube(1))
    problem, backend = make_mechanics(modules.mechanics, _unit_cube(1))

    controller = SimulationController(problem, ep_solver, backend, modules, 1.0, 0.1)

    assert isinstance(controller.mechanics, Solve)
    assert controller.mechanics.problem is problem


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
