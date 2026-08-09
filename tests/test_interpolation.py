from mpi4py import MPI

import basix
import dolfinx
import numpy as np

from simcardemsx.interpolation import MissingValue, TransferOperator


def create_meshes():
    """Helper to create two non-matching 2D unit square meshes."""
    comm = MPI.COMM_WORLD
    # Coarse mesh (e.g., representing mechanics)
    mesh_coarse = dolfinx.mesh.create_unit_square(comm, 4, 4)
    # Fine mesh (e.g., representing electrophysiology)
    mesh_fine = dolfinx.mesh.create_unit_square(comm, 16, 16)
    return mesh_coarse, mesh_fine


def exact_solution(x):
    """A linear function f(x, y) = 2x + 3y.
    Linear functions are represented exactly by CG1 elements,
    so interpolation error should be near machine zero."""
    return 2.0 * x[0] + 3.0 * x[1]


def test_transfer_operator_accuracy():
    mesh_coarse, mesh_fine = create_meshes()

    # Create function spaces
    V_coarse = dolfinx.fem.functionspace(mesh_coarse, ("Lagrange", 1))
    V_fine = dolfinx.fem.functionspace(mesh_fine, ("Lagrange", 1))

    # Setup source and target functions
    u_source = dolfinx.fem.Function(V_coarse)
    u_target = dolfinx.fem.Function(V_fine)

    # Initialize source with the exact analytical solution
    u_source.interpolate(exact_solution)

    # Create the operator and perform the transfer
    transfer_op = TransferOperator(V_source=V_coarse, V_target=V_fine)
    transfer_op.interpolate(u_source, u_target)

    # Verify the target function matches the exact solution
    u_exact_target = dolfinx.fem.Function(V_fine)
    u_exact_target.interpolate(exact_solution)

    # L-infinity norm error should be near machine epsilon
    error = np.max(np.abs(u_target.x.array - u_exact_target.x.array))
    assert error < 1e-12, f"Interpolation failed. Max error: {error}"


def test_missing_value_data_flow():
    mesh_mechanics, mesh_ep = create_meshes()
    element = basix.ufl.element(basix.ElementFamily.P, mesh_mechanics.basix_cell(), 1)
    # Usually EP and interpolation elements are the same in this setup
    interp_element = basix.ufl.element(basix.ElementFamily.P, mesh_ep.basix_cell(), 1)

    num_values = 2
    mv = MissingValue(
        element=element,
        interpolation_element=interp_element,
        mechanics_mesh=mesh_mechanics,
        ep_mesh=mesh_ep,
        num_values=num_values,
    )

    # --- Test 1: EP Array to Mechanics Array Pipeline ---

    # 1. Provide raw array data from EP (e.g., from ODE solver)
    # Fill value index 0 with 5.0, value index 1 with 10.0
    mv.values_ep[0, :] = 5.0
    mv.values_ep[1, :] = 10.0

    # 2. Push to FEniCSx function (normally u_ep, but the code currently expects
    # to interpolate from u_ep_int in interpolate_ep_to_mechanics). Let's simulate
    # the exact flow the user takes. For this test, we load u_ep_int directly.
    mv.u_ep_int[0].x.array[:] = mv.values_ep[0, :]
    mv.u_ep_int[1].x.array[:] = mv.values_ep[1, :]

    # 3. Perform interpolation from EP mesh to Mechanics mesh
    mv.interpolate_ep_to_mechanics()

    # 4. Pull to Mechanics array
    mv.mechanics_function_to_values()

    # Assert values transferred accurately
    assert np.allclose(mv.values_mechanics[0, :], 5.0)
    assert np.allclose(mv.values_mechanics[1, :], 10.0)
