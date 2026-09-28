"""The guards :class:`SimulationController` checks at construction.

That values actually cross through the controller is gate 5, in
``tests/test_round_trip.py``.
"""

from mpi4py import MPI

import dolfinx
import numpy as np
import pulse
import pytest

from simcardemsx.backends import GeneratedActivation
from simcardemsx.controller import SimulationController


def _unit_cube(n: int) -> dolfinx.mesh.Mesh:
    return dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n)


def test_quadrature_degree_mismatch_raises(split_modules, make_ep_solver, make_mechanics):
    modules = split_modules["caisplit"]
    ep_solver = make_ep_solver(modules.ep, _unit_cube(1))
    problem, backend = make_mechanics(
        modules.mechanics,
        _unit_cube(1),
        quadrature_degree=2,
        backend_quadrature_degree=4,
    )
    with pytest.raises(ValueError, match="4.*2|2.*4"):
        SimulationController(problem, ep_solver, backend, modules, 1.0, 0.1)


def test_backend_must_be_the_one_in_the_problem(split_modules, make_ep_solver, make_mechanics):
    modules = split_modules["caisplit"]
    ep_solver = make_ep_solver(modules.ep, _unit_cube(1))
    mesh = _unit_cube(1)
    problem, backend = make_mechanics(modules.mechanics, mesh)
    other = GeneratedActivation(
        modules.mechanics,
        mesh,
        backend.f0,
        quadrature_degree=backend.quadrature_degree,
    )
    with pytest.raises(ValueError, match="model.active"):
        SimulationController(problem, ep_solver, other, modules, 1.0, 0.1)


@pytest.mark.parametrize("dt_mech, dt_ep", [(1.0, 0.3), (0.05, 0.1)])
def test_dt_mech_must_be_a_multiple_of_dt_ep(
    split_modules,
    make_ep_solver,
    make_mechanics,
    dt_mech,
    dt_ep,
):
    modules = split_modules["caisplit"]
    ep_solver = make_ep_solver(modules.ep, _unit_cube(1))
    problem, backend = make_mechanics(modules.mechanics, _unit_cube(1))
    with pytest.raises(ValueError, match="multiple"):
        SimulationController(problem, ep_solver, backend, modules, dt_mech, dt_ep)


def test_dt_mech_a_multiple_of_dt_ep_up_to_round_off_is_accepted(
    split_modules,
    make_ep_solver,
    make_mechanics,
):
    """0.7 / 0.1 is 6.999... and 0.7 % 0.1 is 0.0999... in floating point; 0.7 is
    still seven steps of 0.1, and the check must say so."""
    modules = split_modules["caisplit"]
    ep_solver = make_ep_solver(modules.ep, _unit_cube(1))
    problem, backend = make_mechanics(modules.mechanics, _unit_cube(1))
    controller = SimulationController(problem, ep_solver, backend, modules, 0.7, 0.1)
    assert controller.ep_steps_per_mech == 7


def test_ep_ode_solver_must_be_a_dolfin_ode_solver(split_modules, make_ep_solver, make_mechanics):
    modules = split_modules["caisplit"]
    ep_solver = make_ep_solver(modules.ep, _unit_cube(1))
    ep_solver.ode = object()
    problem, backend = make_mechanics(modules.mechanics, _unit_cube(1))
    with pytest.raises(TypeError, match="DolfinODESolver"):
        SimulationController(problem, ep_solver, backend, modules, 1.0, 0.1)


def _dynamic_mechanics(mech_module, mesh, dt: pulse.Variable, *, quadrature_degree: int = 2):
    """A one-element ``pulse.DynamicProblem`` with ``parameters["dt"]`` set to ``dt``.

    Like ``_mechanics`` in ``tests/conftest.py``, minus boundary conditions: this is
    only ever constructed, never solved, so the rigid-body nullspace they would pin
    down does not matter here.
    """
    geometry = pulse.Geometry(mesh=mesh, metadata={"quadrature_degree": quadrature_degree})
    f0 = dolfinx.fem.Constant(mesh, np.array([1.0, 0.0, 0.0]))
    s0 = dolfinx.fem.Constant(mesh, np.array([0.0, 1.0, 0.0]))
    backend = GeneratedActivation(mech_module, mesh, f0, quadrature_degree=quadrature_degree)
    material = pulse.HolzapfelOgden(
        f0=f0,
        s0=s0,
        **pulse.HolzapfelOgden.transversely_isotropic_parameters(),
    )
    model = pulse.CardiacModel(
        material=material,
        active=backend,
        compressibility=pulse.Incompressible(),
    )
    problem = pulse.DynamicProblem(
        model=model,
        geometry=geometry,
        parameters={"dt": dt},
    )
    return problem, backend


def test_dynamic_problem_dt_must_equal_dt_mech(split_modules, make_ep_solver):
    modules = split_modules["caisplit"]
    ep_solver = make_ep_solver(modules.ep, _unit_cube(1))
    problem, backend = _dynamic_mechanics(
        modules.mechanics,
        _unit_cube(1),
        pulse.Variable(2e-3, "s"),
    )
    with pytest.raises(ValueError, match="2.*1|1.*2"):
        SimulationController(problem, ep_solver, backend, modules, 1.0, 0.1)


def test_dynamic_problem_dt_matching_dt_mech_is_accepted(split_modules, make_ep_solver):
    modules = split_modules["caisplit"]
    ep_solver = make_ep_solver(modules.ep, _unit_cube(1))
    problem, backend = _dynamic_mechanics(
        modules.mechanics,
        _unit_cube(1),
        pulse.Variable(1e-3, "s"),
    )
    controller = SimulationController(problem, ep_solver, backend, modules, 1.0, 0.1)
    assert controller.mechanics.problem is problem
