import logging
from dataclasses import dataclass

import dolfinx
import numpy as np

logger = logging.getLogger(__name__)


def _num_dofs(V: dolfinx.fem.FunctionSpace) -> int:
    """Local dof count of ``V``, including ghosts.

    Equal to ``dolfinx.fem.Function(V).x.array.size``, but available without
    allocating a Function.
    """
    index_map = V.dofmap.index_map
    return (index_map.size_local + index_map.num_ghosts) * V.dofmap.index_map_bs


class TransferOperator:
    """
    A generic operator to interpolate functions between two non-matching FEniCSx function spaces.
    """

    def __init__(self, V_source: dolfinx.fem.FunctionSpace, V_target: dolfinx.fem.FunctionSpace):
        self.V_source = V_source
        self.V_target = V_target

        # In FEniCSx, you compute interpolation data on the *target* mesh cells
        target_mesh = self.V_target.mesh
        cell_map = target_mesh.topology.index_map(target_mesh.topology.dim)
        num_cells = cell_map.size_local + cell_map.num_ghosts
        self.cells_target = np.arange(num_cells, dtype=np.int32)

        # Pre-compute non-matching interpolation data
        self.interpolation_data = dolfinx.fem.create_interpolation_data(
            self.V_target,
            self.V_source,
            self.cells_target,
        )

    def interpolate(self, u_source: dolfinx.fem.Function, u_target: dolfinx.fem.Function) -> None:
        """Interpolates the source function into the target function in-place."""
        u_target.interpolate_nonmatching(u_source, self.cells_target, self.interpolation_data)
        u_target.x.scatter_forward()


@dataclass
class MissingValue:
    """The EP-mesh Functions one direction of transfer passes through.

    One instance per direction, each holding the Functions for that
    direction's variables plus the array the EP solver reads. The mechanics
    side of a transfer is the activation backend's own Function, which the
    coupler interpolates into and out of directly, so nothing here holds one.

    It used to hold four lists -- Functions on both meshes, in two element
    families. Only one was ever on a path per direction. Two of the others
    were not merely unused: ``u_mechanics_int`` was read as an interpolation
    source and written by nothing, which is why the EP subsystem received
    zeros for every distortion state, and a separate previous-values structure
    was written every step and read by nothing.
    """

    ep_space: dolfinx.fem.FunctionSpace
    num_values: int

    def __post_init__(self):
        #: Forward transfers read from these; backward transfers write to them.
        self.u_ep = [dolfinx.fem.Function(self.ep_space) for _ in range(self.num_values)]

        # Sized from the function space rather than from u_ep[0], so that
        # num_values == 0 is representable. That case is real: gotranx omits a
        # side's `missing` entry entirely when it needs nothing from the other,
        # as in the CaTrpn split where EP needs nothing back from mechanics.
        self.values_ep = np.zeros((self.num_values, _num_dofs(self.ep_space)))

    def ep_function_to_values(self) -> None:
        """Read the transferred Functions into the array the EP solver consumes."""
        for i in range(self.num_values):
            self.values_ep[i, :] = self.u_ep[i].x.array[:]
