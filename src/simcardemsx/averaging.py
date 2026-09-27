"""Averaging a discontinuous field into a continuous or cellwise-constant target.

Values sent from the mechanics mesh back to the EP mesh (fibre stretch λ, active
tension, ...) are discontinuous across cells -- there is no single "the" value at a
shared vertex. Point interpolation of such a field into a continuous (P1) space picks
up whichever neighbouring cell's value the interpolation happens to visit last, so the
result silently depends on cell numbering. :func:`make_averager` instead builds a
proper average: a mass-lumped nodal average for a P1 target, an exact cell average for
a DG0 target.

Both source and target are assumed to live on the same mesh -- this module averages
*within* a mesh; :mod:`simcardemsx.interpolation` moves values *between* meshes.
"""

from __future__ import annotations

from typing import Callable

import dolfinx
import ufl


def make_averager(
    source: dolfinx.fem.Function,
    target: dolfinx.fem.Function,
) -> Callable[[], None]:
    """Build a callable that refreshes ``target`` in place with an average of ``source``.

    Compiles the assembly forms once, here, rather than on every call. The returned
    closure re-assembles the numerator against the current ``source`` values each time
    it is called (the denominator -- cell volumes / patch measures -- does not depend on
    ``source`` and is assembled once, up front).

    Supported ``target`` spaces:

    - continuous Lagrange degree 1 (P1): the lumped nodal average
      ``target_i = (∫ source φ_i dx) / (∫ φ_i dx)``. Because a nodal basis function
      φ_i vanishes outside the cells touching node i, this is exactly the volume-weighted
      average of ``source`` over the cells sharing that node.
    - discontinuous Lagrange degree 0 (DG0): the exact cell average
      ``target_K = (∫_K source dx) / |K|``. DG0 basis functions do not overlap between
      cells, so the same numerator/denominator machinery reduces to a per-cell average
      with no approximation.

    Any other ``target`` space raises :class:`NotImplementedError`.

    When ``source`` lives on a quadrature space, the measure is given
    ``metadata={"quadrature_degree": source's degree}`` so FFCx reuses the same
    quadrature rule the values were stored at, instead of picking its own (which would
    not line up with the stored point values at all).

    Parameters
    ----------
    source : dolfinx.fem.Function
        The (generally discontinuous, possibly quadrature) function to average.
    target : dolfinx.fem.Function
        The function refreshed, in place, by the returned callable. Its function space
        determines which averaging scheme is used.

    Returns
    -------
    Callable[[], None]
        Call it with no arguments to refresh ``target`` from the current ``source``.
    """
    V_target = target.function_space
    target_element = V_target.ufl_element()
    family = "DG" if target_element.discontinuous else target_element.family_name
    degree = target_element.degree

    is_p1 = family == "P" and degree == 1
    is_dg0 = family == "DG" and degree == 0
    if not (is_p1 or is_dg0):
        raise NotImplementedError(
            "make_averager only supports a continuous P1 or a DG0 target, got "
            f"family={family!r}, degree={degree}",
        )

    mesh = V_target.mesh
    source_element = source.function_space.ufl_element()
    metadata = None
    if source_element.family_name == "quadrature":
        metadata = {"quadrature_degree": source_element.degree}
    dx = ufl.dx(domain=mesh, metadata=metadata)

    v = ufl.TestFunction(V_target)
    numerator_form = dolfinx.fem.form(source * v * dx)
    denominator_form = dolfinx.fem.form(v * dx)

    # The denominator (cell volumes for DG0, nodal patch measures for P1) depends only
    # on the mesh and V_target, both fixed -- assemble it once.
    denominator = dolfinx.fem.assemble_vector(denominator_form)
    denominator.scatter_reverse(dolfinx.la.InsertMode.add)
    denominator.scatter_forward()

    def refresh() -> None:
        numerator = dolfinx.fem.assemble_vector(numerator_form)
        numerator.scatter_reverse(dolfinx.la.InsertMode.add)
        numerator.scatter_forward()
        target.x.array[:] = numerator.array / denominator.array
        target.x.scatter_forward()

    return refresh
