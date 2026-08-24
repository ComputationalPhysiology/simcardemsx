"""Resolving what crosses between EP and activation.

An activation backend declares what it needs from the EP subsystem and what it
sends back, by name. The generated ODE modules independently describe the same
split, as name-to-index mappings, because gotranx derives it from where the
``.ode`` file was cut. This module reconciles the two and refuses when they
disagree.

The reconciliation is not bookkeeping. The transfer buffers are positional --
row ``i`` of the missing-values array -- while backends are named, so something
has to map between them. Doing that by hand is how a caller ends up wiring row
0 to ``XS`` because that happened to be right for one ODE file, and then
silently transferring calcium into a crossbridge population when a different
file is loaded. Nothing raises on that today; the numbers just come out wrong.

Kept separate from :mod:`simcardemsx.interpolation`, which owns *how* a value
physically moves between two non-matching meshes. This module owns *what*
moves, and in what units.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import dolfinx

from .interpolation import TransferOperator


class TransferMismatch(ValueError):
    """A backend and an ODE file describe different splits, or different units."""


@dataclass(frozen=True)
class ResolvedTransfer:
    """One variable, with everywhere it lives resolved.

    Attributes
    ----------
    name:
        The variable's name, as both sides call it.
    index:
        Its row in the positional missing-values array of the consuming side.
    unit:
        The unit the producing side emits it in, as the backend declared.
    function:
        The backend's Function, which is the transfer's target on the way in
        and its source on the way out. The backend owns it; the coupler writes
        into it rather than deciding where it lives.
    """

    name: str
    index: int
    unit: str
    function: dolfinx.fem.Function


@dataclass(frozen=True)
class TransferPlan:
    """Everything one coupled step needs in order to move state both ways."""

    from_ep: tuple[ResolvedTransfer, ...]
    to_ep: tuple[ResolvedTransfer, ...]
    _into_backend: tuple[TransferOperator, ...]
    _out_of_backend: tuple[TransferOperator, ...]

    def push_to_backend(self, sources: list[dolfinx.fem.Function]) -> None:
        """Interpolate EP-mesh values into the backend's own Functions."""
        for transfer, operator in zip(self.from_ep, self._into_backend):
            operator.interpolate(sources[transfer.index], transfer.function)

    def pull_from_backend(self, targets: list[dolfinx.fem.Function]) -> None:
        """Interpolate the backend's outputs back onto the EP mesh.

        The source is the backend's Function itself. It used to be a separate
        buffer that nothing ever wrote, which is why the EP side received zeros
        for every distortion state.
        """
        for transfer, operator in zip(self.to_ep, self._out_of_backend):
            operator.interpolate(transfer.function, targets[transfer.index])


def _check_names(
    declared: tuple,
    derived: Mapping[str, int],
    direction: str,
    backend_name: str,
) -> None:
    declared_names = {t.name for t in declared}
    derived_names = set(derived)
    if declared_names == derived_names:
        return

    only_backend = sorted(declared_names - derived_names)
    only_ode = sorted(derived_names - declared_names)
    lines = [
        f"{backend_name} and the ODE file describe different splits ({direction}).",
    ]
    if only_backend:
        lines.append(f"  the backend expects, but the ODE file does not provide: {only_backend}")
    if only_ode:
        lines.append(f"  the ODE file provides, but the backend does not expect: {only_ode}")
    lines.append(
        "  Load the ODE file whose split matches this backend, or use the "
        "backend that matches this ODE file.",
    )
    raise TransferMismatch("\n".join(lines))


def _check_unit(
    transfer,
    units: Mapping[str, str | None],
    backend_name: str,
) -> None:
    declared = units.get(transfer.name)
    if declared is None:
        # The source did not say. Not the same as dimensionless, and not
        # something to guess at -- currently the case for every variable that
        # crosses in the shipped ODE files.
        return
    if declared != transfer.unit:
        raise TransferMismatch(
            f"{backend_name} expects {transfer.name!r} in {transfer.unit!r}, but "
            f"the ODE file declares it in {declared!r}. Convert on one side, or "
            f"correct whichever declaration is wrong.",
        )


def resolve(
    backend,
    *,
    ep_missing: Mapping[str, int],
    mech_missing: Mapping[str, int],
    units: Mapping[str, str | None],
    ep_sources: Mapping[str, dolfinx.fem.Function],
    ep_targets: Mapping[str, dolfinx.fem.Function],
) -> TransferPlan:
    """Reconcile a backend's declared transfers with the ODE file's split.

    ``ep_missing`` and ``mech_missing`` are the generated modules' own
    name-to-index mappings: each names what that side needs *from the other*,
    so what the backend wants from EP is what the mechanics module is missing.

    ``ep_sources`` and ``ep_targets`` are the EP-mesh Functions the values pass
    through, keyed by name; they supply the function spaces the interpolation
    operators are built between.

    Raises :class:`TransferMismatch` if the two descriptions disagree, in names
    or in units.
    """
    backend_name = type(backend).__name__

    wants = tuple(backend.wants_from_ep())
    gives = tuple(backend.gives_to_ep())

    _check_names(wants, mech_missing, "EP -> mechanics", backend_name)
    _check_names(gives, ep_missing, "mechanics -> EP", backend_name)

    for transfer in wants + gives:
        _check_unit(transfer, units, backend_name)

    inputs = backend.ep_inputs
    outputs = backend.ep_outputs

    from_ep = tuple(
        ResolvedTransfer(
            name=t.name,
            index=mech_missing[t.name],
            unit=t.unit,
            function=inputs[t.name],
        )
        for t in wants
    )
    to_ep = tuple(
        ResolvedTransfer(
            name=t.name,
            index=ep_missing[t.name],
            unit=t.unit,
            function=outputs[t.name],
        )
        for t in gives
    )

    into_backend = tuple(
        TransferOperator(
            V_source=ep_sources[t.name].function_space,
            V_target=t.function.function_space,
        )
        for t in from_ep
    )
    out_of_backend = tuple(
        TransferOperator(
            V_source=t.function.function_space,
            V_target=ep_targets[t.name].function_space,
        )
        for t in to_ep
    )

    return TransferPlan(
        from_ep=from_ep,
        to_ep=to_ep,
        _into_backend=into_backend,
        _out_of_backend=out_of_backend,
    )
