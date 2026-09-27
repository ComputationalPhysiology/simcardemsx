from mpi4py import MPI

import basix
import dolfinx
import numpy as np
import pytest

from simcardemsx.averaging import make_averager


def _tet_volume(pts: np.ndarray) -> float:
    a, b, c, d = pts
    return abs(np.dot(np.cross(b - a, c - a), d - a)) / 6.0


def _hand_lumped(mesh: dolfinx.mesh.Mesh, V_p1, V_dg0, dg0_array: np.ndarray) -> np.ndarray:
    """Reference lumped nodal average, computed by hand from cell volumes.

    On a tet, sum, over the cells touching vertex i, the DG0 cell value weighted by
    the cell volume, then divide by the summed cell volume -- exactly the ratio
    ``Σ_K∋i f_K|K| / Σ_K∋i |K|`` from the brief (the |K|/4 factor common to every
    term of both the numerator and the denominator cancels).

    Dof indices are read through ``V.dofmap`` rather than assumed to equal cell/vertex
    indices -- dolfinx does not guarantee that correspondence.
    """
    tdim = mesh.topology.dim
    num_cells = mesh.topology.index_map(tdim).size_local
    x = mesh.geometry.x
    geom_dofmap = mesh.geometry.dofmaps[0]

    num_p1_dofs = V_p1.dofmap.index_map.size_local + V_p1.dofmap.index_map.num_ghosts
    acc_num = np.zeros(num_p1_dofs)
    acc_den = np.zeros(num_p1_dofs)
    for c in range(num_cells):
        vol = _tet_volume(x[geom_dofmap[c]])
        dg0_dof = V_dg0.dofmap.cell_dofs(c)[0]
        f_val = dg0_array[dg0_dof]
        for p1_dof in V_p1.dofmap.cell_dofs(c):
            acc_num[p1_dof] += f_val * vol
            acc_den[p1_dof] += vol
    return acc_num / acc_den


def test_lumped_average_is_bounded_and_exact():
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
    V_dg0 = dolfinx.fem.functionspace(mesh, ("DG", 0))
    V_p1 = dolfinx.fem.functionspace(mesh, ("P", 1))
    dg0 = dolfinx.fem.Function(V_dg0)
    p1 = dolfinx.fem.Function(V_p1)

    tdim = mesh.topology.dim
    num_cells = mesh.topology.index_map(tdim).size_local
    mid = mesh.geometry.x[mesh.geometry.dofmaps[0]].mean(axis=1)  # cell midpoints

    # Place the jump through the DG0 dofmap, not by assuming dof index == cell index.
    for c in range(num_cells):
        dof = V_dg0.dofmap.cell_dofs(c)[0]
        dg0.x.array[dof] = 1.0 if mid[c, 0] < 0.5 else 0.9

    make_averager(dg0, p1)()

    assert np.all((p1.x.array >= 0.9 - 1e-14) & (p1.x.array <= 1.0 + 1e-14))
    np.testing.assert_allclose(p1.x.array, _hand_lumped(mesh, V_p1, V_dg0, dg0.x.array), rtol=1e-12)


def test_cell_average_of_quadrature_data_is_exact_for_linear_fields():
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
    degree = 4
    Q_element = basix.ufl.quadrature_element(mesh.basix_cell(), value_shape=(), degree=degree)
    V_quad = dolfinx.fem.functionspace(mesh, Q_element)
    V_dg0 = dolfinx.fem.functionspace(mesh, ("DG", 0))

    quad = dolfinx.fem.Function(V_quad)
    quad.interpolate(lambda x: x[0] + 2.0 * x[1])
    dg0 = dolfinx.fem.Function(V_dg0)

    make_averager(quad, dg0)()

    tdim = mesh.topology.dim
    num_cells = mesh.topology.index_map(tdim).size_local
    mid = mesh.geometry.x[mesh.geometry.dofmaps[0]].mean(axis=1)

    dg0_by_cell = np.array([dg0.x.array[V_dg0.dofmap.cell_dofs(c)[0]] for c in range(num_cells)])
    np.testing.assert_allclose(dg0_by_cell, mid[:, 0] + 2.0 * mid[:, 1], rtol=1e-12)


def test_unsupported_target_raises():
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
    src = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("DG", 0)))
    with pytest.raises(NotImplementedError, match="DG"):
        make_averager(src, dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("DG", 1))))
