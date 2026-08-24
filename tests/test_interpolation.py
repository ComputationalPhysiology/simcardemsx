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


def test_missing_value_holds_the_ep_side_of_a_transfer():
    """The buffers a transfer passes through on the EP mesh.

    The mechanics side of a transfer is the activation backend's own Function,
    which the coupler interpolates into and out of directly, so nothing here
    holds one. Two of the mechanics-side buffers this class used to allocate
    were not merely unused: one was read as an interpolation source and never
    written, which is why the EP subsystem received zeros for every distortion
    state.
    """
    _, mesh_ep = create_meshes()
    element = basix.ufl.element(basix.ElementFamily.P, mesh_ep.basix_cell(), 1)

    mv = MissingValue(
        element=element,
        interpolation_element=element,
        ep_mesh=mesh_ep,
        num_values=2,
    )

    # The coupler interpolates into u_ep; the EP solver reads values_ep.
    mv.u_ep[0].x.array[:] = 7.0
    mv.u_ep[1].x.array[:] = 11.0
    mv.ep_function_to_values()

    assert np.allclose(mv.values_ep[0, :], 7.0)
    assert np.allclose(mv.values_ep[1, :], 11.0)


def test_a_direction_that_carries_nothing_is_representable():
    """gotranx omits a side's `missing` entry entirely when it needs nothing
    from the other, as in the CaTrpn split. Zero-width must not crash."""
    _, mesh_ep = create_meshes()
    element = basix.ufl.element(basix.ElementFamily.P, mesh_ep.basix_cell(), 1)

    mv = MissingValue(
        element=element,
        interpolation_element=element,
        ep_mesh=mesh_ep,
        num_values=0,
    )

    assert mv.values_ep.shape[0] == 0
    mv.ep_function_to_values()
