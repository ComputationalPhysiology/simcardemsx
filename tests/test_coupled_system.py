# tests/test_coupled_system.py
from mpi4py import MPI

import dolfinx
import numpy as np
import pulse

from simcardemsx.controller import SimulationController
from simcardemsx.land import LandModel
from simcardemsx.mechanicsproblem import MechanicsProblem

# Assuming you have a way to initialize a basic EP solver from fenicsx-beat
# from beat import MonodomainModel, ...


def test_coupled_smoke_test():
    """
    Integration test to ensure the EP solver, Mechanics solver, and
    Transfer Operators can step through time together without crashing.
    """
    comm = MPI.COMM_WORLD
    mesh_mech = dolfinx.mesh.create_unit_cube(comm, 1, 1, 1)
    mesh_ep = dolfinx.mesh.create_unit_cube(comm, 2, 2, 2)

    geo = pulse.Geometry(mesh=mesh_mech, metadata={"quadrature_degree": 2})
    f0 = dolfinx.fem.Constant(mesh_mech, np.array([1.0, 0.0, 0.0]))
    s0 = dolfinx.fem.Constant(mesh_mech, np.array([0.0, 1.0, 0.0]))
    n0 = dolfinx.fem.Constant(mesh_mech, np.array([0.0, 0.0, 1.0]))

    V_mech_dg = dolfinx.fem.functionspace(mesh_mech, ("DG", 1))
    XS_mech = dolfinx.fem.Function(V_mech_dg)
    XW_mech = dolfinx.fem.Function(V_mech_dg)

    active_model = LandModel(f0=f0, s0=s0, n0=n0, XS=XS_mech, XW=XW_mech, mesh=mesh_mech)
    material_params = pulse.HolzapfelOgden.transversely_isotropic_parameters()
    material = pulse.HolzapfelOgden(f0=f0, s0=s0, **material_params)

    cardiac_model = pulse.CardiacModel(
        material=material,
        active=active_model,
        compressibility=pulse.Incompressible(),
    )

    # Roller boundaries
    def dirichlet_bc(V):
        V0, _ = V.sub(0).collapse()
        zero = dolfinx.fem.Function(V0)
        zero.x.array[:] = 0.0
        fdim = mesh_mech.topology.dim - 1
        x0_facets = dolfinx.mesh.locate_entities_boundary(
            mesh_mech,
            fdim,
            lambda x: np.isclose(x[0], 0.0),
        )
        y0_facets = dolfinx.mesh.locate_entities_boundary(
            mesh_mech,
            fdim,
            lambda x: np.isclose(x[1], 0.0),
        )
        z0_facets = dolfinx.mesh.locate_entities_boundary(
            mesh_mech,
            fdim,
            lambda x: np.isclose(x[2], 0.0),
        )
        return [
            dolfinx.fem.dirichletbc(
                zero,
                dolfinx.fem.locate_dofs_topological((V.sub(0), V0), fdim, x0_facets),
                V.sub(0),
            ),
            dolfinx.fem.dirichletbc(
                zero,
                dolfinx.fem.locate_dofs_topological((V.sub(1), V0), fdim, y0_facets),
                V.sub(1),
            ),
            dolfinx.fem.dirichletbc(
                zero,
                dolfinx.fem.locate_dofs_topological((V.sub(2), V0), fdim, z0_facets),
                V.sub(2),
            ),
        ]

    bcs = pulse.BoundaryConditions(dirichlet=[dirichlet_bc])
    mech_problem = MechanicsProblem(
        model=cardiac_model,
        geometry=geo,
        bcs=bcs,
        parameters={"base_bc": pulse.problem.BaseBC.free},
    )

    class DummyEPSolver:
        def __init__(self):
            self.V = dolfinx.fem.functionspace(mesh_ep, ("DG", 1))
            self.state_function = dolfinx.fem.Function(self.V)

            class MockODE:
                def __init__(self, state):
                    self._values = state.x.array
                    self.parameters = None

            self.ode = MockODE(self.state_function)

        def step(self, t_span):
            self.state_function.x.array[:] += 0.05

    ep_solver = DummyEPSolver()

    # Create a lightweight Mock ODE Model to satisfy the SimulationController API
    class MockODEModel:
        def __init__(self):
            class MissingMech:
                def interpolate_ep_to_mechanics(self):
                    # Simulate the TransferOperator moving EP state to Mechanics XS
                    XS_mech.x.array[:] = ep_solver.state_function.x.array[: len(XS_mech.x.array)]

                def mechanics_function_to_values(self):
                    pass

            self.missing_mech = MissingMech()
            self.missing_ep = None

        def update_ep_missing_values(self, t, vals, params):
            pass

        def update_prev_missing_mech(self):
            pass

    ode_model = MockODEModel()

    controller = SimulationController(
        mechanics_problem=mech_problem,
        ep_solver=ep_solver,
        ode_model=ode_model,
        dt_mech=1.0,
        dt_ep=0.1,
    )

    for step in range(3):
        controller.step()

    assert np.max(XS_mech.x.array) > 0.0, "EP state failed to transfer to mechanics"
    disp = np.linalg.norm(mech_problem.u.x.array)
    assert disp > 0.0, "Mechanics mesh did not deform"
