"""The controller accepts beat's reaction-eikonal EP (``FootCurrentODESolver``)."""

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
from beat.eikonal import Source, solve_eikonal

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


def _eikonal_times(mesh):
    axes = tuple(
        dolfinx.fem.Constant(mesh, np.array(e, dtype=dolfinx.default_scalar_type))
        for e in ([1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0])
    )
    return solve_eikonal(
        mesh,
        (0.6, 0.3, 0.3),
        axes,
        [Source(lambda x: np.isclose(x[0], 0.0), 0.0)],
    )


@pytest.mark.slow
@pytest.mark.parametrize("split", ["zetasplit", "caisplit"])
@pytest.mark.parametrize("mode", ["re+", "re-"])
def test_round_trip_under_reaction_eikonal(
    split,
    mode,
    split_modules,
    make_re_ep_solver,
    make_mechanics,
):
    """Gate 5 under R-E: values cross both ways through the controller, and EP's nodes
    activate when the eikonal times say."""
    modules = split_modules[split]
    ep_mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 3, 3, 3)
    t_a = _eikonal_times(ep_mesh)
    ep_solver = make_re_ep_solver(modules.ep, ep_mesh, mode, t_a=t_a)
    problem, backend = make_mechanics(
        modules.mechanics,
        dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1),
        quadrature_degree=2,
    )
    controller = SimulationController(problem, ep_solver, backend, modules, DT_MECH, DT_EP)
    ode = dolfin_ode_solver(ep_solver)
    v = ode.v_ode.x.array
    n = v.size
    first = np.full(n, np.inf)
    upstrokes = np.zeros(n, dtype=int)
    previous = v.copy()

    def record(t, i):
        nonlocal previous
        above = v > 0
        upstrokes[above & ~(previous > 0)] += 1
        newly = above & np.isinf(first)
        first[newly] = t
        previous = v.copy()

    for _ in range(10):
        controller.step(ep_callback=record)

    ta = t_a.x.array
    print(mode, split, "activation - t_a: min", (first - ta).min(), "max", (first - ta).max())
    assert np.all(np.isfinite(first))
    assert np.all(first <= ta + 2.0)
    assert np.all(upstrokes <= 1)
    if mode == "re-":
        assert np.all(first >= ta)
    assert any(np.any(f.x.array != 0.0) for f in backend.inputs.values())
    if split == "zetasplit":
        mv = ode.missing_variables
        assert np.ptp(mv, axis=1).max() > 0.0
