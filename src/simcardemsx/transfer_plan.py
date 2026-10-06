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

:class:`TransferPlan` is the runtime half: it adds the function spaces,
averagers and `interpolation.TransferOperator`s that move these names' values
between beat's arrays on the EP mesh and the activation backend on the
mechanics mesh. It is also a `~simcardemsx.checkpoint.Checkpointable`
(namespace ``"transfer"``), holding the rows of beat's arrays that its backward
direction writes, so that a restore never needs to move anything back.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable, Mapping, NamedTuple, Protocol

import dolfinx
import numpy as np

from .averaging import family_name, make_averager
from .interpolation import TransferOperator

if TYPE_CHECKING:
    import beat

    from .backends.base import CoupledBackend


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


class CrossingSide(Protocol):
    """What :func:`resolve` reads of the activation side: a generated module, or a
    backend exposing the same two name -> index dicts."""

    @property
    def missing(self) -> Mapping[str, int]: ...

    @property
    def provides(self) -> Mapping[str, int]: ...


def resolve(ep_module: ModuleType, activation_module: CrossingSide) -> Crossings:
    """Derive what crosses between `ep_module` and `activation_module`.

    `ep_module` is the EP module `ode_model.load_ode_modules` loads from one `.ode`
    file's split; `activation_module` is the other side, any `CrossingSide`: the
    mechanics module of the same split, or a backend that carries `missing` and
    `provides` (`GeneratedActivation`, `CrossbridgeSegregated`). Either may lack a
    `missing` or `provides` attribute
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


#: Averaging target on the mechanics mesh, by EP ODE space: the space matching it,
#: so that the operator onto the EP mesh is the identity when the meshes coincide.
_AVERAGING_ELEMENT: dict[str, tuple[str, int]] = {"P1": ("P", 1), "DG0": ("DG", 0)}


def _space_name(V: dolfinx.fem.FunctionSpace) -> str:
    """``"P1"``, ``"DG0"``, ``"DG1"``, ``"quadrature2"``, ...: the family and degree of ``V``."""
    element = V.ufl_element()
    return f"{family_name(element)}{element.degree}"


def _require_shape(what: str, array, expected: tuple[int, int], why: str) -> None:
    if not isinstance(array, np.ndarray) or array.shape != expected:
        shape = array.shape if isinstance(array, np.ndarray) else type(array).__name__
        raise ValueError(f"{what} must be an array of shape {expected}, got {shape}: {why}")


class _Backward(NamedTuple):
    """One name crossing activation -> EP, and where in beat's arrays it lands."""

    name: str
    average: Callable[[], None]
    averaged: dolfinx.fem.Function  # on the mechanics mesh, in the space matching EP's
    received: dolfinx.fem.Function  # on the EP ODE space
    array: str  # attribute of the ODE solver: "missing_variables" or "parameters"
    row: int

    @property
    def restart_name(self) -> str:
        """``transfer_missing_<name>`` or ``transfer_parameter_<name>``."""
        kind = "missing" if self.array == "missing_variables" else "parameter"
        return f"transfer_{kind}_{self.name}"


class TransferPlan:
    """Move the values named by ``crossings`` between beat's EP arrays and ``backend``.

    Forward (EP -> activation): :meth:`forward` calls the EP module's generated
    ``missing_values()`` on beat's state and parameter arrays, and interpolates
    each row it produces from the EP ODE space straight into the backend's input
    Function of that name (a quadrature space, typically, on another mesh).

    Backward (activation -> EP): :meth:`backward` **averages, never
    point-interpolates**, each backend output -- the backward names, plus
    ``lmbda`` when ``crossings.stretch_to_ep`` -- onto a space on the mechanics
    mesh matching the EP ODE space (lumped projection onto P1, or the cell
    average onto DG0; see :func:`~simcardemsx.averaging.make_averager`),
    interpolates that onto the EP ODE space, and writes it into row
    ``ep_module.missing[name]`` of ``ode.missing_variables``, or, for ``lmbda``,
    row ``ep_module.parameter["lmbda"]`` of ``ode.parameters``. λ is
    discontinuous across cells, and point interpolation into P1 would take each
    node's value from whichever neighbouring cell was visited last.

    Both writes are **in place**: beat's ``DolfinODESolver`` hands its
    ``parameters`` and ``missing_variables`` arrays to its inner solver by
    reference at construction, so rebinding the attributes would leave EP
    integrating with the old arrays. For the same reason the arrays must already
    have one column per point: a λ that varies between points cannot be written
    into a parameter array that has one value for all of them.

    As a ``Checkpointable`` (namespace ``"transfer"``), the plan's state is those
    rows themselves, as they are now: one Function on the EP ODE space per row that
    :meth:`backward` writes. They cannot be recomputed from the backend's outputs in
    general: before the backend has accepted a step its outputs are zero, while EP
    holds its own initial values (``lmbda`` = 1, for one).

    Parameters
    ----------
    crossings:
        From :func:`resolve` on ``ep_module`` and ``backend``.
    ep_module:
        The generated EP module; the one ``ode`` integrates.
    ode:
        beat's ODE solver. Its ODE space (``ode.v_ode.function_space``) must be P1
        or DG0.
    backend:
        The activation backend on the mechanics mesh.

    Raises
    ------
    NotImplementedError
        If the EP ODE space is neither P1 nor DG0.
    ValueError
        If values cross back and ``ode.missing_variables`` is not an array of
        shape ``(len(ep_module.missing), num_points)``, or if ``lmbda`` crosses
        back and ``ode.parameters`` is not an array of shape
        ``(num_parameters, num_points)``.
    """

    def __init__(
        self,
        crossings: Crossings,
        ep_module: ModuleType,
        ode: beat.odesolver.DolfinODESolver,
        backend: CoupledBackend,
    ):
        V_ep = ode.v_ode.function_space
        space = _space_name(V_ep)
        if space not in _AVERAGING_ELEMENT:
            raise NotImplementedError(
                f"The EP ODE space is {space}; only {' and '.join(_AVERAGING_ELEMENT)} are "
                "supported, because values going back to EP are averaged onto a matching "
                "space on the mechanics mesh first, and averaging is only defined onto those.",
            )

        num_points = ode.v_ode.x.array.size
        ep_missing: Mapping[str, int] = getattr(ep_module, "missing", {})
        if crossings.backward:
            _require_shape(
                "ode.missing_variables",
                ode.missing_variables,
                (len(ep_missing), num_points),
                f"{list(crossings.backward)} cross back to EP, one row each, one column "
                "per point of the EP ODE space",
            )
        if crossings.stretch_to_ep:
            _require_shape(
                "ode.parameters",
                ode.parameters,
                (len(ep_module.init_parameter_values()), num_points),
                "lmbda crosses back to EP and differs between points, so the parameters "
                "need one column per point of the EP ODE space (e.g. np.tile them)",
            )

        self.crossings = crossings
        self.ep_module = ep_module
        self.ode = ode
        self.backend = backend

        # The generated missing_values() takes a missing_variables argument only
        # when the EP side needs something back, i.e. when there are backward names.
        self._ep_takes_missing = bool(crossings.backward)

        self._forward_sources = {
            name: dolfinx.fem.Function(V_ep, name=name) for name in crossings.forward
        }
        self._forward_operator = (
            TransferOperator(V_source=V_ep, V_target=backend.space) if crossings.forward else None
        )

        targets = [("missing_variables", name, ep_missing[name]) for name in crossings.backward]
        if crossings.stretch_to_ep:
            targets.append(("parameters", "lmbda", ep_module.parameter["lmbda"]))
        V_averaged = dolfinx.fem.functionspace(backend.space.mesh, _AVERAGING_ELEMENT[space])
        self._backward_operator = (
            TransferOperator(V_source=V_averaged, V_target=V_ep) if targets else None
        )
        self._backward: list[_Backward] = []
        for array, name, row in targets:
            averaged = dolfinx.fem.Function(V_averaged, name=name)
            self._backward.append(
                _Backward(
                    name=name,
                    average=make_averager(backend.outputs[name], averaged),
                    averaged=averaged,
                    received=dolfinx.fem.Function(V_ep, name=name),
                    array=array,
                    row=row,
                ),
            )

    def forward(self, t: float) -> None:
        """EP -> ``backend.inputs``, from beat's current states at time ``t``."""
        if self._forward_operator is None:
            return
        ode = self.ode
        args = [t, ode.values, ode.parameters]
        if self._ep_takes_missing:
            args.append(ode.missing_variables)
        values = self.ep_module.missing_values(*args)
        for name, source in self._forward_sources.items():
            source.x.array[:] = values[self.ep_module.provides[name]]
            self._forward_operator.interpolate(source, self.backend.inputs[name])

    def backward(self) -> None:
        """``backend.outputs`` -> ``ode.missing_variables`` / ``ode.parameters``, in place."""
        if self._backward_operator is None:
            return
        for crossing in self._backward:
            crossing.average()
            self._backward_operator.interpolate(crossing.averaged, crossing.received)
            getattr(self.ode, crossing.array)[crossing.row, :] = crossing.received.x.array

    # ------------------------------------------------------------------
    # Checkpoint / restart (simcardemsx.checkpoint.Checkpointable)
    # ------------------------------------------------------------------

    namespace = "transfer"

    def restart_functions(self) -> list[tuple[str, dolfinx.fem.Function]]:
        """Fresh copies of the rows of beat's arrays that :meth:`backward` writes:
        ``transfer_missing_<name>`` for each backward name, then
        ``transfer_parameter_lmbda`` when λ crosses to EP."""
        V_ep = self.ode.v_ode.function_space
        functions = []
        for crossing in self._backward:
            f = dolfinx.fem.Function(V_ep, name=crossing.restart_name)
            f.x.array[:] = getattr(self.ode, crossing.array)[crossing.row, :]
            functions.append((crossing.restart_name, f))
        return functions

    def restart_metadata(self) -> dict[str, Any]:
        return {}

    def load_restart(
        self,
        functions: Sequence[tuple[str, dolfinx.fem.Function]],
        metadata: Mapping[str, Any],
    ) -> None:
        """Copy the rows back into beat's arrays, **in place**.

        Raises ``ValueError`` if ``functions`` are not the rows this plan writes.
        """
        by_name = dict(functions)
        expected = [crossing.restart_name for crossing in self._backward]
        if sorted(by_name) != sorted(expected):
            raise ValueError(
                f"The transfer rows are {sorted(by_name)}, this plan writes {sorted(expected)}",
            )
        for crossing in self._backward:
            values = by_name[crossing.restart_name].x.array
            getattr(self.ode, crossing.array)[crossing.row, :] = values
