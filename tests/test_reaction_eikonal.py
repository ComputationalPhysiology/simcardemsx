"""The controller accepts beat's reaction-eikonal EP (``FootCurrentODESolver``)."""

from mpi4py import MPI

import dolfinx
import pytest

from simcardemsx.controller import SimulationController, dolfin_ode_solver

DT_EP = 0.05
DT_MECH = 1.0


def _build(split_modules, make_mechanics, ep_solver_factory):
    modules = split_modules["zetasplit"]
    ep, mech = modules
    ep_solver = ep_solver_factory(ep, dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 3, 3, 3))
    problem, backend = make_mechanics(
        mech,
        dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1),
        quadrature_degree=2,
    )
    return modules, ep_solver, problem, backend


@pytest.mark.parametrize("mode", ["re+", "re-"])
def test_controller_accepts_reaction_eikonal(
    mode,
    split_modules,
    make_re_ep_solver,
    make_mechanics,
):
    modules, ep_solver, problem, backend = _build(
        split_modules,
        make_mechanics,
        lambda ep, mesh: make_re_ep_solver(ep, mesh, mode),
    )
    controller = SimulationController(problem, ep_solver, backend, modules, DT_MECH, DT_EP)
    assert controller.plan.ode is ep_solver.ode.ode


def test_dolfin_ode_solver_unwraps(split_modules, make_ep_solver, make_re_ep_solver):
    ep = split_modules["zetasplit"].ep
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 3, 3, 3)
    re = make_re_ep_solver(ep, mesh, "re+")
    assert dolfin_ode_solver(re) is re.ode.ode
    mono = make_ep_solver(ep, mesh)
    assert dolfin_ode_solver(mono) is mono.ode


def test_refuses_a_foot_around_another_solver(split_modules, make_re_ep_solver, make_mechanics):
    modules, ep_solver, problem, backend = _build(
        split_modules,
        make_mechanics,
        lambda ep, mesh: make_re_ep_solver(ep, mesh, "re+"),
    )
    foot = ep_solver.ode
    foot.ode = object()
    with pytest.raises(TypeError, match="DolfinODESolver"):
        SimulationController(problem, ep_solver, backend, modules, DT_MECH, DT_EP)
