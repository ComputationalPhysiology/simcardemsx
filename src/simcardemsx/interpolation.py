import logging
from dataclasses import dataclass

import dolfinx
import numpy as np
import ufl

logger = logging.getLogger(__name__)


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
    element: ufl.finiteelement.AbstractFiniteElement
    interpolation_element: ufl.finiteelement.AbstractFiniteElement
    mechanics_mesh: dolfinx.mesh.Mesh
    ep_mesh: dolfinx.mesh.Mesh
    num_values: int

    def __post_init__(self):
        self.V_ep = dolfinx.fem.functionspace(self.ep_mesh, self.element)
        self.V_mechanics = dolfinx.fem.functionspace(self.mechanics_mesh, self.element)

        self.V_ep_int = dolfinx.fem.functionspace(self.ep_mesh, self.interpolation_element)
        self.V_mechanics_int = dolfinx.fem.functionspace(
            self.mechanics_mesh,
            self.interpolation_element,
        )

        self.u_ep = [dolfinx.fem.Function(self.V_ep) for _ in range(self.num_values)]
        self.u_mechanics = [dolfinx.fem.Function(self.V_mechanics) for _ in range(self.num_values)]

        self.u_ep_int = [dolfinx.fem.Function(self.V_ep_int) for _ in range(self.num_values)]
        self.u_mechanics_int = [
            dolfinx.fem.Function(self.V_mechanics_int) for _ in range(self.num_values)
        ]

        self.values_ep = np.zeros((self.num_values, self.u_ep[0].x.array.size))
        self.values_mechanics = np.zeros((self.num_values, self.u_mechanics[0].x.array.size))

        # Setup Transfer Operators instead of manual interpolation data
        self.transfer_ep2mech = TransferOperator(V_source=self.V_ep_int, V_target=self.V_mechanics)
        self.transfer_mech2ep = TransferOperator(V_source=self.V_mechanics_int, V_target=self.V_ep)

    @property
    def domain_mechanics(self):
        return self.mechanics_mesh

    @property
    def domain_ep(self):
        return self.ep_mesh

    def ep_values_to_function(self) -> None:
        for i in range(self.num_values):
            self.u_ep[i].x.array[:] = self.values_ep[i]

    def ep_function_to_values(self) -> None:
        for i in range(self.num_values):
            self.values_ep[i, :] = self.u_ep[i].x.array[:]

    def mechanics_values_to_function(self) -> None:
        for i in range(self.num_values):
            self.u_mechanics[i].x.array[:] = self.values_mechanics[i]

    def mechanics_function_to_values(self) -> None:
        for i in range(self.num_values):
            self.values_mechanics[i, :] = self.u_mechanics[i].x.array[:]

    def interpolate_ep_to_mechanics(self) -> None:
        logger.debug("Interpolate ep to mechanics")
        for i in range(self.num_values):
            self.transfer_ep2mech.interpolate(self.u_ep_int[i], self.u_mechanics[i])

    def interpolate_mechanics_to_ep(self) -> None:
        logger.debug("Interpolate mechanics to ep")
        for i in range(self.num_values):
            self.transfer_mech2ep.interpolate(self.u_mechanics_int[i], self.u_ep[i])
