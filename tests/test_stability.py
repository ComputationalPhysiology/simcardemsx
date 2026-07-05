from mpi4py import MPI

import dolfinx
import numpy as np
import pulse

from simcardemsx.land import LandModel
from simcardemsx.mechanicsproblem import MechanicsProblem


def test_land_relaxation_stability():
    """
    Tests that rapid muscle relaxation (negative stretch rate)
    does not produce physically impossible negative active tension.
    """
    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_unit_cube(comm, 1, 1, 1)

    f0 = dolfinx.fem.Constant(mesh, np.array([1.0, 0.0, 0.0]))
    s0 = dolfinx.fem.Constant(mesh, np.array([0.0, 1.0, 0.0]))
    n0 = dolfinx.fem.Constant(mesh, np.array([0.0, 0.0, 1.0]))

    V_dg = dolfinx.fem.functionspace(mesh, ("DG", 1))
    XS = dolfinx.fem.Function(V_dg)
    XW = dolfinx.fem.Function(V_dg)

    # 1. Simulate fully contracted state
    XS.x.array[:] = 0.5
    active_model = LandModel(f0=f0, s0=s0, n0=n0, XS=XS, XW=XW, mesh=mesh)

    # 2. Force a massive negative stretch rate (rapid relaxation)
    active_model.lmbda_prev.x.array[:] = 1.2  # Was stretched
    lmbda_current = 0.9  # Suddenly compressed
    active_model.t.value = 1.0
    active_model._t_prev.value = 0.0  # dt = 1.0

    # 3. Evaluate active tension
    # We use a numerical evaluation of the UFL expression
    Ta_expr = dolfinx.fem.Expression(
        active_model.Ta(lmbda_current),
        V_dg.element.interpolation_points,
    )
    Ta_eval = dolfinx.fem.Function(V_dg)
    Ta_eval.interpolate(Ta_expr)

    # Muscle fibers can ONLY pull (Ta >= 0). They cannot push!
    min_Ta = np.min(Ta_eval.x.array)
    assert min_Ta >= 0.0, f"Instability detected! Active tension became negative: {min_Ta}"


def test_coupled_long_term_stability():
    """
    Runs a 1-element cube for 500ms to ensure the solver survives
    the relaxation phase without oscillating into divergence.
    """
    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_unit_cube(comm, 1, 1, 1)
    geo = pulse.Geometry(mesh=mesh, metadata={"quadrature_degree": 2})

    f0 = dolfinx.fem.Constant(mesh, np.array([1.0, 0.0, 0.0]))
    s0 = dolfinx.fem.Constant(mesh, np.array([0.0, 1.0, 0.0]))
    n0 = dolfinx.fem.Constant(mesh, np.array([0.0, 0.0, 1.0]))

    V_dg = dolfinx.fem.functionspace(mesh, ("DG", 1))
    XS = dolfinx.fem.Function(V_dg)
    XW = dolfinx.fem.Function(V_dg)

    active_model = LandModel(f0=f0, s0=s0, n0=n0, XS=XS, XW=XW, mesh=mesh)
    mat_params = pulse.HolzapfelOgden.transversely_isotropic_parameters()
    cardiac_model = pulse.CardiacModel(
        material=pulse.HolzapfelOgden(f0=f0, s0=s0, **mat_params),
        active=active_model,
        compressibility=pulse.Incompressible(),
    )

    # Allow the incompressible element to expand laterally to preserve volume
    def dirichlet_bc(V):
        V0, _ = V.sub(0).collapse()
        zero = dolfinx.fem.Function(V0)
        zero.x.array[:] = 0.0

        fdim = mesh.topology.dim - 1

        def x0_face(x):
            return np.isclose(x[0], 0.0)

        def y0_face(x):
            return np.isclose(x[1], 0.0)

        def z0_face(x):
            return np.isclose(x[2], 0.0)

        x0_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, x0_face)
        y0_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, y0_face)
        z0_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, z0_face)

        x0_dofs = dolfinx.fem.locate_dofs_topological((V.sub(0), V0), fdim, x0_facets)
        y0_dofs = dolfinx.fem.locate_dofs_topological((V.sub(1), V0), fdim, y0_facets)
        z0_dofs = dolfinx.fem.locate_dofs_topological((V.sub(2), V0), fdim, z0_facets)

        return [
            dolfinx.fem.dirichletbc(zero, x0_dofs, V.sub(0)),
            dolfinx.fem.dirichletbc(zero, y0_dofs, V.sub(1)),
            dolfinx.fem.dirichletbc(zero, z0_dofs, V.sub(2)),
        ]

    problem = MechanicsProblem(
        model=cardiac_model,
        geometry=geo,
        bcs=pulse.BoundaryConditions(dirichlet=[dirichlet_bc]),
        parameters={"base_bc": pulse.problem.BaseBC.free},
    )

    # Simulate a realistic EP transient directly (0 to 500ms)
    time_steps = np.arange(0, 500, 2.0)

    stretch_history = []

    for t in time_steps:
        # Create a synthetic AP transient (peaks at 150ms, returns to 0 by 400ms)
        xs_val = 0.2 * np.exp(-0.0001 * (t - 150) ** 2)
        XS.x.array[:] = xs_val
        XW.x.array[:] = xs_val * 0.5

        active_model.t.value = t
        problem.solve()
        problem.post_solve()
        active_model.update_prev()

        # Track the stretch to check for oscillations
        stretch_history.append(np.mean(active_model.lmbda.x.array))

    # Check for oscillatory behavior
    stretch_diff = np.diff(stretch_history)
    sign_changes = np.sum(np.diff(np.sign(stretch_diff)) != 0)

    # A single beat should have exactly 1 sign change (contraction -> relaxation)
    assert sign_changes <= 2, (
        f"Highly oscillatory behavior detected! Sign changed {sign_changes} times."
    )
