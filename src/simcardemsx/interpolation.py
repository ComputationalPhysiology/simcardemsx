import dolfinx
import numpy as np


class TransferOperator:
    """
    A generic operator to interpolate functions between two non-matching FEniCSx function spaces.
    """

    def __init__(self, V_source: dolfinx.fem.FunctionSpace, V_target: dolfinx.fem.FunctionSpace):
        # dolfinx does not support interpolating *from* a quadrature space -- it
        # segfaults/aborts the whole process instead of raising, so this must be
        # caught before anything else here (create_interpolation_data included) runs.
        if V_source.ufl_element().family_name == "quadrature":
            raise ValueError(
                "TransferOperator cannot use a quadrature space as its source "
                f"(V_source has family {V_source.ufl_element().family_name!r}); "
                "quadrature values only exist at that element's own points and "
                "cannot be interpolated from.",
            )

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
