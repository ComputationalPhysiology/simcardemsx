"""What crosses between EP and activation, derived from the generated modules.

`generate_ode_code` (see `ode_model.py`) splits one gotranx `.ode` file into two
modules -- an EP remainder and a `mechanics` (activation) component -- and each
one is annotated with three name -> index dicts: `missing` (what it needs from
the *other* side), `provides` (what its own `missing_values()` returns, i.e.
what it hands to the other side) and `parameter`. Nothing declares the
crossings directly; they are read off those dicts.

:func:`resolve` is the pure half of that: no meshes, no function spaces, just
the two modules' dicts in and a :class:`Crossings` out. It pairs each side's
`missing` names against the other side's `provides` by name, adds `lmbda`
per the rule below, and raises if a name either side needs is produced by
neither -- which cannot happen for two modules generated from the same `.ode`
file, since gotranx builds each side's `missing` to match the other's
`missing_values()` output. It is the guard for modules that were not, e.g.
mismatched splits or stale generated code.

`lmbda` is not read from any `provides` dict: it comes from the fibre stretch
of the mechanics solve, not from a generated `missing_values()`. Whether it
crosses back to EP is therefore not a name-matching question but a capability
one -- does the EP module have a use for it -- answered by whether `lmbda`
appears in the EP module's `parameter` dict. `dLambda` never crosses back:
no shipped EP remainder consumes it.

`TransferPlan` (the next layer, not this module) adds the function spaces and
`interpolation.TransferOperator`s that actually move these names' values
between the EP and mechanics meshes at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import Mapping


@dataclass(frozen=True)
class Crossings:
    """The names that cross between EP and activation for one `.ode` split.

    `forward` and `backward` are name tuples, not index tuples: the two
    sides' index spaces are unrelated (each is private to that module's own
    `missing_values()` signature), so an index from one means nothing on the
    other. Ordering is preserved -- by the *consuming* side's `missing`
    index -- only because it is a natural, deterministic choice; nothing
    downstream relies on it lining up with anything else.
    """

    forward: tuple[str, ...]
    """EP -> activation: names the activation module's `missing` dict lists,
    ordered by its index."""

    backward: tuple[str, ...]
    """Activation -> EP: names the EP module's `missing` dict lists, ordered
    by its index."""

    stretch_to_ep: bool
    """Whether `lmbda` should also cross activation -> EP: true iff the EP
    module's `parameter` dict has an `lmbda` entry."""


def _names_by_index(missing: Mapping[str, int]) -> tuple[str, ...]:
    """`missing`'s keys, ordered by their index value."""
    return tuple(name for name, _ in sorted(missing.items(), key=lambda item: item[1]))


def resolve(ep_module: ModuleType, activation_module: ModuleType) -> Crossings:
    """Derive what crosses between `ep_module` and `activation_module`.

    Both are the modules `ode_model.load_ode_modules` loads from one `.ode`
    file's split. A module may lack a `missing` or `provides` attribute
    entirely (gotranx omits `missing` when a side needs nothing; treat a
    missing `provides` the same way) -- both are read with `getattr(...,
    {})`, not indexing, for that reason.

    Raises `ValueError` naming every name either side's `missing` dict lists
    that the other side's `provides` dict does not produce, checked in both
    directions before raising so a single error reports the whole mismatch
    rather than just the first direction checked.
    """
    ep_missing: Mapping[str, int] = getattr(ep_module, "missing", {})
    ep_provides: Mapping[str, int] = getattr(ep_module, "provides", {})
    ep_parameter: Mapping[str, int] = getattr(ep_module, "parameter", {})
    activation_missing: Mapping[str, int] = getattr(activation_module, "missing", {})
    activation_provides: Mapping[str, int] = getattr(activation_module, "provides", {})

    unmet_forward = sorted(name for name in activation_missing if name not in ep_provides)
    unmet_backward = sorted(name for name in ep_missing if name not in activation_provides)

    if unmet_forward or unmet_backward:
        problems = []
        if unmet_forward:
            problems.append(
                f"activation module needs {unmet_forward} from EP, "
                "but the EP module's `provides` does not produce them",
            )
        if unmet_backward:
            problems.append(
                f"EP module needs {unmet_backward} from activation, "
                "but the activation module's `provides` does not produce them",
            )
        raise ValueError("; ".join(problems))

    return Crossings(
        forward=_names_by_index(activation_missing),
        backward=_names_by_index(ep_missing),
        stretch_to_ep="lmbda" in ep_parameter,
    )
