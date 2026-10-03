"""The contract the controller drives an activation backend through."""

from mpi4py import MPI

import dolfinx
import pytest

from simcardemsx.backends import GeneratedActivation
from simcardemsx.backends.base import CoupledBackend
from simcardemsx.controller import SimulationController
from simcardemsx.transfer_plan import resolve

SPLITS = ["zetasplit", "caisplit"]


def _generated(modules, mesh):
    from test_generated_activation import _f0

    return GeneratedActivation(modules.mechanics, mesh, _f0(mesh), quadrature_degree=2)


@pytest.mark.parametrize("split", SPLITS)
def test_generated_activation_is_a_coupled_backend(split_modules, split):
    modules = split_modules[split]
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    backend = _generated(modules, mesh)

    assert backend.missing == modules.mechanics.missing
    assert backend.provides == modules.mechanics.provides
    assert isinstance(backend, CoupledBackend)

    backend.begin_step(3.0, 0.5)
    assert float(backend.t.value) == 3.0
    assert float(backend.dt.value) == 0.5


@pytest.mark.parametrize("split", SPLITS)
def test_resolve_against_the_backend_equals_against_its_module(split_modules, split):
    modules = split_modules[split]
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    backend = _generated(modules, mesh)
    assert resolve(modules.ep, backend) == resolve(modules.ep, backend.module)


def test_controller_calls_begin_step_then_post_solve(
    split_modules,
    make_ep_solver,
    make_mechanics,
):
    modules = split_modules["zetasplit"]
    ep_solver = make_ep_solver(
        modules.ep,
        dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 3, 3, 3),
    )
    problem, backend = make_mechanics(
        modules.mechanics,
        dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1),
        quadrature_degree=2,
    )
    controller = SimulationController(problem, ep_solver, backend, modules, 1.0, 0.05)

    calls: list = []
    begin, post = backend.begin_step, backend.post_solve

    def record_begin(t_n, dt):
        calls.append(("begin_step", float(t_n), float(dt)))
        begin(t_n, dt)

    def record_post():
        calls.append("post_solve")
        post()

    backend.begin_step = record_begin
    backend.post_solve = record_post
    controller.step()
    controller.step()

    assert calls == [
        ("begin_step", 0.0, 1.0),
        "post_solve",
        ("begin_step", 1.0, 1.0),
        "post_solve",
    ]
