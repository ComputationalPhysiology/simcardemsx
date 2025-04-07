from dataclasses import dataclass
import logging
import dolfinx
import numpy as np
import ufl

logger = logging.getLogger(__name__)


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
            self.mechanics_mesh, self.interpolation_element
        )

        self.u_ep = [dolfinx.fem.Function(self.V_ep) for _ in range(self.num_values)]
        self.u_mechanics = [dolfinx.fem.Function(self.V_mechanics) for _ in range(self.num_values)]

        self.u_ep_int = [dolfinx.fem.Function(self.V_ep_int) for _ in range(self.num_values)]
        self.u_mechanics_int = [
            dolfinx.fem.Function(self.V_mechanics_int) for _ in range(self.num_values)
        ]

        self.values_ep = np.zeros((self.num_values, self.u_ep[0].x.array.size))
        self.values_mechanics = np.zeros((self.num_values, self.u_mechanics[0].x.array.size))

        bbox_mechanics = dolfinx.geometry.bb_tree(
            self.domain_mechanics, self.domain_mechanics.topology.dim, padding=1e-6
        )
        bbox_ep = dolfinx.geometry.bb_tree(
            self.domain_ep, self.domain_ep.topology.dim, padding=1e-6
        )

        # Mechanics to EP
        global_mech_tree = bbox_mechanics.create_global_tree(self.domain_mechanics.comm)
        collisions = dolfinx.geometry.compute_collisions_trees(bbox_ep, global_mech_tree)
        self.cells_mech2ep = np.unique(collisions[:, 0])

        self.V_mechanics_interpolation_data_mech2ep = dolfinx.fem.create_interpolation_data(
            self.V_ep, self.V_mechanics, self.cells_mech2ep
        )

        # EP to Mechanics
        global_ep_tree = bbox_ep.create_global_tree(self.domain_ep.comm)
        collisions = dolfinx.geometry.compute_collisions_trees(bbox_mechanics, global_ep_tree)
        self.cells_ep2mech = np.unique(collisions[:, 0])
        self.V_ep_interpolation_data_ep2mech = dolfinx.fem.create_interpolation_data(
            self.V_mechanics, self.V_ep, self.cells_ep2mech
        )

    @property
    def domain_mechanics(self):
        return self.V_mechanics.mesh

    @property
    def domain_ep(self):
        return self.V_ep.mesh

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
            logger.debug(f"Interpolate {i}")
            self.u_mechanics[i].interpolate_nonmatching(
                self.u_ep_int[i], self.cells_ep2mech, self.V_ep_interpolation_data_ep2mech
            )

    def interpolate_mechanics_to_ep(self) -> None:
        logger.debug("Interpolate mechanics to ep")
        for i in range(self.num_values):
            logger.debug(f"Interpolate {i}")
            self.u_ep[i].interpolate_nonmatching(
                self.u_mechanics_int[i],
                self.cells_mech2ep,
                self.V_mechanics_interpolation_data_mech2ep,
            )
