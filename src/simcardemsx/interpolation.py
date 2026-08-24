import logging
from dataclasses import dataclass

import dolfinx
import numpy as np
import ufl

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
    """EP-mesh buffers for the variables crossing between the two subsystems.

    One instance per direction. It holds the EP-side Functions a transfer
    passes through and the array the EP solver reads its missing variables
    from; the mechanics side of a transfer is the activation backend's own
    Function, which the coupler interpolates into and out of directly.

    It used to hold mechanics-side Functions too. Two of those were the cause
    of a bug rather than a cost: one was read as an interpolation source and
    written by nothing, so every distortion state delivered to the EP
    subsystem was zero, and another was written every step and read by
    nothing.
    """

    element: ufl.finiteelement.AbstractFiniteElement
    interpolation_element: ufl.finiteelement.AbstractFiniteElement
    ep_mesh: dolfinx.mesh.Mesh
    num_values: int

    def __post_init__(self):
        self.V_ep = dolfinx.fem.functionspace(self.ep_mesh, self.element)
        self.V_ep_int = dolfinx.fem.functionspace(self.ep_mesh, self.interpolation_element)

        #: Targets for what the EP side is missing, on the EP mesh.
        self.u_ep = [dolfinx.fem.Function(self.V_ep) for _ in range(self.num_values)]
        #: Sources for what the mechanics side is missing, on the EP mesh.
        self.u_ep_int = [dolfinx.fem.Function(self.V_ep_int) for _ in range(self.num_values)]

        # Sized from the function space rather than from u_ep[0], so that
        # num_values == 0 is representable. That case is real: gotranx omits a
        # side's `missing` entry entirely when it needs nothing from the other,
        # as in the CaTrpn split where EP needs nothing back from mechanics.
        self.values_ep = np.zeros((self.num_values, _num_dofs(self.V_ep)))

    @property
    def domain_ep(self):
        return self.ep_mesh

    def ep_function_to_values(self) -> None:
        """Read the transferred Functions into the array the EP solver consumes."""
        for i in range(self.num_values):
            self.values_ep[i, :] = self.u_ep[i].x.array[:]
