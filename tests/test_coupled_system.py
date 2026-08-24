"""The controller's construction-time guards.

What this file used to hold -- a smoke test asserting that EP state reached
mechanics and that the mesh deformed -- is now covered by `test_coupler.py`,
which does it against a real EP ODE integration rather than a mock that
simulated the transfer it was meant to be testing.

What is left here is the part no other test covers: the controller refusing
configurations that would silently decouple the two subsystems.
"""

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest

from simcardemsx.backends import ZetaSplitUFL
from simcardemsx.controller import SimulationController
from simcardemsx.ode_model import RuntimeODEModel, load_ode_modules

from .test_coupler import (
    ZETA_SPLIT_ODE,
    ODELevelEPSolver,
    _fibres,
    _problem,
    _roller_bcs,
    _zeta_backend,
)


def _parts(tmp_path):
    comm = MPI.COMM_WORLD
    mesh_mech = dolfinx.mesh.create_unit_cube(comm, 1, 1, 1)
    mesh_ep = dolfinx.mesh.create_unit_cube(comm, 2, 2, 2)

    tmp_path.mkdir(parents=True, exist_ok=True)
    ode_file = tmp_path / "split.ode"
    ode_file.write_text(ZETA_SPLIT_ODE)
    modules = load_ode_modules(ode_file, tmp_path / "generated")

    V_mech = dolfinx.fem.functionspace(mesh_mech, ("DG", 1))
    V_ep = dolfinx.fem.functionspace(mesh_ep, ("DG", 1))
    ode_model = RuntimeODEModel(
        ep_module_dict=modules.ep.__dict__,
        mech_module_dict=modules.mechanics.__dict__,
        mech_ode_space=V_mech,
        ep_ode_space=V_ep,
    )
    backend = _zeta_backend(mesh_mech)
    problem = _problem(mesh_mech, backend, _roller_bcs(mesh_mech))
    ep_solver = ODELevelEPSolver(modules.ep, V_ep, ode_model.missing_ep.values_ep)
    return mesh_mech, ode_model, backend, problem, ep_solver


def test_refuses_a_backend_that_is_not_in_the_mechanics_form(tmp_path):
    """Advancing one backend while solving with another decouples them silently.

    The backend is reachable as `mechanics_problem.model.active`, so passing it
    separately creates the possibility of the two disagreeing. That is worth a
    parameter -- it puts the controller's most important collaborator in its
    signature -- but only if the disagreement is caught.
    """
    mesh_mech, ode_model, backend, problem, ep_solver = _parts(tmp_path)

    f0, s0, n0 = _fibres(mesh_mech)
    impostor = ZetaSplitUFL(f0=f0, s0=s0, n0=n0, mesh=mesh_mech)
    assert impostor is not backend

    with pytest.raises(ValueError, match="not the activation backend"):
        SimulationController(
            mechanics_problem=problem,
            ep_solver=ep_solver,
            ode_model=ode_model,
            backend=impostor,
            dt_mech=1.0,
            dt_ep=0.1,
        )


def test_requires_the_mechanics_step_to_be_a_whole_number_of_ep_steps(tmp_path):
    """A fractional ratio would silently drift the two clocks apart."""
    _, ode_model, backend, problem, ep_solver = _parts(tmp_path)

    with pytest.raises(ValueError, match="exact multiple"):
        SimulationController(
            mechanics_problem=problem,
            ep_solver=ep_solver,
            ode_model=ode_model,
            backend=backend,
            dt_mech=1.0,
            dt_ep=0.3,
        )


def test_rejects_arguments_the_backend_no_longer_takes(tmp_path):
    """`ZetaSplitUFL` used to accept its transfer Functions from the caller.

    It owns them now. A caller still passing them must be told, not silently
    ignored -- which is what a catch-all **kwargs would have done.
    """
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    f0, s0, n0 = _fibres(mesh)
    V = dolfinx.fem.functionspace(mesh, ("DG", 1))

    with pytest.raises(TypeError):
        ZetaSplitUFL(
            f0=f0,
            s0=s0,
            n0=n0,
            mesh=mesh,
            XS=dolfinx.fem.Function(V),
            XW=dolfinx.fem.Function(V),
        )


def test_a_coupled_step_advances_both_clocks(tmp_path):
    """The controller runs dt_mech/dt_ep micro-steps per mechanics step."""
    _, ode_model, backend, problem, ep_solver = _parts(tmp_path)

    controller = SimulationController(
        mechanics_problem=problem,
        ep_solver=ep_solver,
        ode_model=ode_model,
        backend=backend,
        dt_mech=1.0,
        dt_ep=0.1,
    )
    controller.step()

    assert controller.ep_step_idx == 10
    assert controller.mech_step_idx == 1
    assert np.isclose(controller.t, 1.0)
