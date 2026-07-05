# tests/test_coupled_system.py
from mpi4py import MPI

import dolfinx
import numpy as np
import pulse

from simcardemsx.controller import SimulationController
from simcardemsx.interpolation import TransferOperator
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

    # 1. Coarse meshes for fast testing
    mesh_mech = dolfinx.mesh.create_unit_cube(comm, 1, 1, 1)
    mesh_ep = dolfinx.mesh.create_unit_cube(comm, 2, 2, 2)

    # 2. Setup Mechanics (using the successful setup from previous tests)
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

    def left_face(x):
        return np.isclose(x[0], 0.0)

    fdim = mesh_mech.topology.dim - 1
    left_facets = dolfinx.mesh.locate_entities_boundary(mesh_mech, fdim, left_face)

    def dirichlet_bc(V):
        dofs = dolfinx.fem.locate_dofs_topological(V, fdim, left_facets)
        u_bc = dolfinx.fem.Function(V)
        u_bc.x.array[:] = 0.0
        return [dolfinx.fem.dirichletbc(u_bc, dofs)]

    bcs = pulse.BoundaryConditions(dirichlet=[dirichlet_bc])

    # 2. Initialize problem WITH boundary conditions
    mech_problem = MechanicsProblem(
        model=cardiac_model,
        geometry=geo,
        bcs=bcs,
        parameters={
            "base_bc": pulse.problem.BaseBC.free,
            "petsc_options": {
                "snes_type": "newtonls",
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            },
        },
    )

    # 3. Setup EP Solver (Mocking the interface slightly if beat isn't fully ready)
    # In a real test, instantiate your beat.MonodomainModel here.
    class DummyEPSolver:
        def __init__(self):
            self.V = dolfinx.fem.functionspace(mesh_ep, ("DG", 1))
            self.state_function = dolfinx.fem.Function(self.V)
            self.stretch_function = dolfinx.fem.Function(self.V)
            self.t = 0.0

        def step(self):
            # Simulate an action potential rising
            self.state_function.x.array[:] += 0.05
            self.t += 0.1

    ep_solver = DummyEPSolver()

    # 4. Setup Transfer Operators
    transfer_ep2mech = TransferOperator(V_source=ep_solver.V, V_target=V_mech_dg)
    transfer_mech2ep = TransferOperator(V_source=V_mech_dg, V_target=ep_solver.V)

    # 5. Initialize Controller
    controller = SimulationController(
        mechanics_problem=mech_problem,
        ep_solver=ep_solver,
        transfer_ep_to_mech=transfer_ep2mech,
        transfer_mech_to_ep=transfer_mech2ep,
        dt_mech=1.0,
        dt_ep=0.1,
    )

    # --- Execute Coupling Loop ---
    for step in range(3):
        controller.step()

    # --- Assertions ---
    # 1. Check EP successfully transferred to Mechanics
    assert np.max(XS_mech.x.array) > 0.0, "EP state failed to transfer to mechanics"

    # 2. Check Mechanics solver deformed the mesh
    disp = np.linalg.norm(mech_problem.u.x.array)
    assert disp > 0.0, "Mechanics mesh did not deform"

    # 3. Check Mechanics stretch transferred back to EP
    assert np.max(ep_solver.stretch_function.x.array) > 0.0, "Stretch failed to transfer to EP"
