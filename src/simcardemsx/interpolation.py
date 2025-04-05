from dataclasses import dataclass
import dolfinx
import numpy as np
import ufl


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
        for i in range(self.num_values):
            self.u_mechanics[i].interpolate(self.u_ep_int[i])

    def interpolate_mechanics_to_ep(self) -> None:
        for i in range(self.num_values):
            self.u_ep[i].interpolate(self.u_mechanics_int[i])
