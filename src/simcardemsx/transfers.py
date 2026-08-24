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

import warnings
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import TYPE_CHECKING, Mapping

import dolfinx

if TYPE_CHECKING:
    from .backends.base import Transfer

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
    unit:
        The unit the producing side emits it in, as the backend declared.
    function:
        The backend's Function, which is the transfer's target on the way in
        and its source on the way out. The backend owns it; the coupler writes
        into it rather than deciding where it lives.
    """

    name: str
    unit: str
    function: dolfinx.fem.Function
    ep_function: dolfinx.fem.Function
    operator: TransferOperator


@dataclass(frozen=True)
class TransferPlan:
    """Everything one coupled step needs in order to move state both ways.

    Each transfer holds both of its ends, resolved once, so moving a step's
    state needs no arguments and no positional lookup. The caller does not
    re-supply buffers it has already described.
    """

    from_ep: tuple[ResolvedTransfer, ...]
    to_ep: tuple[ResolvedTransfer, ...]

    def push_to_backend(self) -> None:
        """Interpolate EP-mesh values into the backend's own Functions."""
        for transfer in self.from_ep:
            transfer.operator.interpolate(transfer.ep_function, transfer.function)

    def pull_from_backend(self) -> None:
        """Interpolate the backend's outputs back onto the EP mesh.

        The source is the backend's Function itself. It used to be a separate
        buffer that nothing ever wrote, which is why the EP side received zeros
        for every distortion state.
        """
        for transfer in self.to_ep:
            transfer.operator.interpolate(transfer.function, transfer.ep_function)


def _check_names(
    declared: "tuple[Transfer, ...]",
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


class AssumedUnitWarning(UserWarning):
    """A transfer's unit was not declared, so the backend's was assumed.

    A warning rather than an error because the assumption is almost always
    right, and because some variables *cannot* be declared: a derived
    expression, such as the troponin buffering flux, has nowhere in the ODE
    syntax to carry a unit. A warning rather than silence because this is
    exactly the assumption that produces a plausible wrong answer when it is
    wrong -- millimolar read as micromolar gives a calcium transient that looks
    entirely reasonable and is off by a thousand.

    Escalate with ``warnings.simplefilter("error", AssumedUnitWarning)`` to
    require every crossing variable to be declared.
    """


class UnitMismatchWarning(UserWarning):
    """A declared unit disagreed with what the backend expects, under
    :attr:`UnitPolicy.warn`, where it would otherwise have been an error."""


class UnitPolicy(str, Enum):
    """How hard to insist on units.

    Only units. The *names* that cross are always checked: a backend paired
    with the wrong split is a correctness bug that no preference makes
    acceptable, and it is the error this module mainly exists to catch.
    """

    #: Disagreement raises; an undeclared unit is assumed and warned about.
    #: The default, because a scale error here is silent and large.
    strict = "strict"

    #: Disagreement warns instead of raising. For working through an ODE file
    #: whose annotations are known to be incomplete or wrong.
    warn = "warn"

    #: No unit checking and no warnings at all. For a user who does not
    #: annotate units and does not want to hear about it.
    off = "off"


@lru_cache(maxsize=1)
def _registry():
    # Built lazily: constructing a pint registry is not free, and a user who
    # has opted out should never pay for it.
    import pint

    return pint.UnitRegistry()


def _quantity(unit: str):
    """``unit`` as a pint quantity, or ``None`` if pint cannot read it."""
    try:
        return _registry()(unit)
    except Exception:
        # A unit we cannot read is not the same as a wrong one. Fall back to comparing the
        # strings, rather than rejecting a unit we merely failed to understand.
        return None


def _compare(declared: str, expected: str) -> tuple[bool, float | None]:
    """``(agree, ratio)``.

    ``ratio`` is how many ``expected`` units make one ``declared`` unit, when
    the two share a dimension -- so 1000.0 for millimolar declared against
    micromolar expected. It is ``None`` when the units have different
    dimensions, or when pint could not read one of them.
    """
    if declared.strip() == expected.strip():
        return True, 1.0

    a, b = _quantity(declared), _quantity(expected)
    if a is None or b is None:
        # Neither could be understood well enough to call it a disagreement.
        return declared.strip().lower() == expected.strip().lower(), None
    if a.dimensionality != b.dimensionality:
        return False, None

    ratio = (1 * a).to(b).magnitude
    return bool(abs(ratio - 1.0) < 1e-12), float(ratio)


def _check_unit(
    transfer,
    units: Mapping[str, str | None],
    backend_name: str,
    policy: "UnitPolicy",
) -> None:
    if policy is UnitPolicy.off:
        return

    declared = units.get(transfer.name)
    if declared is None:
        # The source did not say. Assume what the backend expects -- the only
        # assumption under which the coupling is correct -- and say so, rather
        # than proceeding silently. Some variables cannot be declared at all:
        # one derived by an expression, such as the troponin buffering flux,
        # has nowhere in the ODE syntax to carry a unit.
        warnings.warn(
            f"{backend_name}: the ODE file declares no unit for "
            f"{transfer.name!r}, so {transfer.unit!r} is assumed, which is what "
            f"{backend_name} expects. Annotate {transfer.name!r} in the .ode "
            f"source to make this explicit; if it is a derived expression, it "
            f"cannot be annotated and this warning is the record.",
            AssumedUnitWarning,
            stacklevel=4,
        )
        return

    agree, ratio = _compare(declared, transfer.unit)
    if agree:
        return

    # Stated as a conversion rather than a bare factor: "a factor of 0.001"
    # leaves the reader to work out which way round it goes, which is the one
    # thing they most need to know.
    scale = (
        f" One {declared} is {ratio:g} {transfer.unit}."
        if ratio is not None
        else " They do not even have the same dimensions, so one of the two is"
        " describing a different quantity."
    )
    message = (
        f"{backend_name} expects {transfer.name!r} in {transfer.unit!r}, but "
        f"the ODE file declares it in {declared!r}.{scale} Correct whichever "
        f"declaration is wrong, or pass units=UnitPolicy.warn to proceed anyway."
    )
    if policy is UnitPolicy.warn:
        warnings.warn(message, UnitMismatchWarning, stacklevel=4)
        return
    raise TransferMismatch(message)


def check(
    backend,
    *,
    ep_missing: Mapping[str, int],
    mech_missing: Mapping[str, int],
    units: Mapping[str, str | None],
    policy: UnitPolicy = UnitPolicy.strict,
) -> tuple[tuple, tuple]:
    """Verify a backend's declared transfers against the ODE file's split.

    Pure: no meshes, no function spaces, no dolfinx objects beyond whatever the
    backend already holds. That is deliberate -- this is a comparison between
    two mappings, and keeping it cheap is what makes it practical to check
    every backend against every split.

    ``policy`` governs the unit check only; see :class:`UnitPolicy`. The names
    that cross are always checked.

    Returns the declared transfers, in both directions. Raises
    :class:`TransferMismatch` if the two descriptions disagree.
    """
    backend_name = type(backend).__name__

    wants = tuple(backend.wants_from_ep())
    gives = tuple(backend.gives_to_ep())

    _check_names(wants, mech_missing, "EP -> mechanics", backend_name)
    _check_names(gives, ep_missing, "mechanics -> EP", backend_name)

    for transfer in wants + gives:
        _check_unit(transfer, units, backend_name, policy)

    return wants, gives


def resolve(
    backend,
    *,
    ep_missing: Mapping[str, int],
    mech_missing: Mapping[str, int],
    units: Mapping[str, str | None],
    ep_sources: Mapping[str, dolfinx.fem.Function],
    ep_targets: Mapping[str, dolfinx.fem.Function],
    policy: UnitPolicy = UnitPolicy.strict,
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
    wants, gives = check(
        backend,
        ep_missing=ep_missing,
        mech_missing=mech_missing,
        units=units,
        policy=policy,
    )

    inputs = backend.ep_inputs
    outputs = backend.ep_outputs

    from_ep = tuple(
        ResolvedTransfer(
            name=t.name,
            unit=t.unit,
            function=inputs[t.name],
            ep_function=ep_sources[t.name],
            operator=TransferOperator(
                V_source=ep_sources[t.name].function_space,
                V_target=inputs[t.name].function_space,
            ),
        )
        for t in wants
    )
    to_ep = tuple(
        ResolvedTransfer(
            name=t.name,
            unit=t.unit,
            function=outputs[t.name],
            ep_function=ep_targets[t.name],
            operator=TransferOperator(
                V_source=outputs[t.name].function_space,
                V_target=ep_targets[t.name].function_space,
            ),
        )
        for t in gives
    )

    return TransferPlan(from_ep=from_ep, to_ep=to_ep)
