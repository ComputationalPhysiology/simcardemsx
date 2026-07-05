from mpi4py import MPI

import dolfinx
import numpy as np
import pulse

from simcardemsx.land import LandModel
from simcardemsx.mechanicsproblem import MechanicsProblem


def test_mechanics_activation_ramp():
    comm = MPI.COMM_WORLD

    # 1. Create mesh and geometry wrapper
    mesh = dolfinx.mesh.create_unit_cube(comm, 3, 3, 3)
    # Wrap in pulse.Geometry

    geo = pulse.Geometry(mesh=mesh, metadata={"quadrature_degree": 2})

    # Microstructure
    f0 = dolfinx.fem.Constant(mesh, np.array([1.0, 0.0, 0.0]))
    s0 = dolfinx.fem.Constant(mesh, np.array([0.0, 1.0, 0.0]))
    n0 = dolfinx.fem.Constant(mesh, np.array([0.0, 0.0, 1.0]))

    # 2. Setup mock EP state variables (XS, XW)
    V_dg = dolfinx.fem.functionspace(mesh, ("DG", 1))
    XS = dolfinx.fem.Function(V_dg)
    XW = dolfinx.fem.Function(V_dg)
    XS.x.array[:] = 0.0
    XW.x.array[:] = 0.0

    # 3. Initialize the Active Model
    active_model = LandModel(f0=f0, s0=s0, n0=n0, XS=XS, XW=XW, mesh=mesh)
    active_model.t.value = 0.0

    # 4. Setup Cardiac Model Components
    material_parameters = pulse.HolzapfelOgden.transversely_isotropic_parameters()
    material = pulse.HolzapfelOgden(f0=f0, s0=s0, **material_parameters)
    compressibility = pulse.Incompressible()

    cardiac_model = pulse.CardiacModel(
        material=material,
        active=active_model,
        compressibility=compressibility,
    )

    # 5. Define Boundary Conditions (fenicsx-pulse requires callables for Dirichlet)
    def left_face(x):
        return np.isclose(x[0], 0.0)

    fdim = mesh.topology.dim - 1
    left_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, left_face)

    def dirichlet_bc(V):
        dofs = dolfinx.fem.locate_dofs_topological(V, fdim, left_facets)
        u_bc = dolfinx.fem.Function(V)
        u_bc.x.array[:] = 0.0
        return [dolfinx.fem.dirichletbc(u_bc, dofs)]

    bcs = pulse.BoundaryConditions(dirichlet=[dirichlet_bc])

    # 6. Initialize MechanicsProblem
    problem = MechanicsProblem(
        model=cardiac_model,
        geometry=geo,
        bcs=bcs,
        parameters={
            "base_bc": pulse.problem.BaseBC.free,
            "petsc_options": {
                # Add these monitors to see the Newton solver iterations in your terminal
                "snes_monitor": None,
                "snes_linesearch_monitor": None,
                "snes_type": "newtonls",
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            },
        },
    )

    # --- SIMULATION LOOP ---

    # Step 1: Solve at resting state (XS = 0.0)
    problem.solve()
    problem.post_solve()

    disp_rest = np.linalg.norm(problem.u.x.array)
    ta_rest = np.max(active_model.Ta_current.x.array)
    assert np.isclose(disp_rest, 0.0, atol=1e-8), "Resting displacement should be zero."
    assert np.isclose(ta_rest, 0.0, atol=1e-8), "Resting active tension should be zero."

    # Step 2: Trigger activation
    dt = 1.0
    active_model.t.value += dt
    XS.x.array[:] = 0.05
    XW.x.array[:] = 0.02

    problem.solve()
    problem.post_solve()

    disp_active = np.linalg.norm(problem.u.x.array)
    ta_active = np.max(active_model.Ta_current.x.array)

    assert ta_active > 0.0, "Active tension should increase when XS > 0."
    assert disp_active > 1e-4, "Tissue should deform under active tension."
