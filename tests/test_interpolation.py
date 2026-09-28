from mpi4py import MPI

import basix
import dolfinx
import numpy as np
import pytest

from simcardemsx.interpolation import TransferOperator


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


def test_refuses_a_quadrature_source():
    """A quadrature source must be rejected before any interpolation is attempted.

    dolfinx does not raise a Python exception for this -- it aborts the whole
    process -- so this only constructs the TransferOperator (never calling
    interpolate_nonmatching / create_interpolation_data from a quadrature source)
    and checks that construction itself raises.
    """
    mesh, _ = create_meshes()
    Q = dolfinx.fem.functionspace(
        mesh,
        basix.ufl.quadrature_element(mesh.basix_cell(), value_shape=(), degree=2),
    )
    with pytest.raises(ValueError, match="quadrature"):
        TransferOperator(V_source=Q, V_target=dolfinx.fem.functionspace(mesh, ("P", 1)))
