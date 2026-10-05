"""Checkpoint and restart: what a component must offer to be saved and restored.

A component is :class:`Checkpointable` when it can name the ``dolfinx`` Functions that
hold its state, describe the rest of it as JSON, and take both back. A checkpointer
writes every component's :meth:`~Checkpointable.restart_functions` to one file and
their :meth:`~Checkpointable.restart_metadata` to another, and on restart hands each
component a fresh call of ``restart_functions()`` whose values it has overwritten.

The controller's components are its EP solver (:class:`EPState`), its transfer plan
(the rows of EP's arrays that come from the backend), its mechanics problem
(:class:`MechanicsState`), a cycle controller if it drives one (:class:`CycleState`),
the backend, and the controller itself. A :class:`Snapshot` holds copies of all their
state in memory (:func:`take_snapshot`), so that a step that fails can be undone
(:func:`restore_snapshot`). A restore is exact on its own: nothing has to be moved
back to EP afterwards.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import dolfinx
import numpy as np

if TYPE_CHECKING:
    import beat
    import pulse
    import pulse.cycle


@runtime_checkable
class Checkpointable(Protocol):
    """A component whose state can be written to disk and read back.

    Attributes
    ----------
    namespace
        Prefix of this component's restart function names (``activation_...``), so
        that names stay unique across components. Names must be unique within it.

    Notes
    -----
    ``restart_metadata`` must be JSON-able: plain floats, ints, bools, strings, lists
    and dicts, with no numpy scalars.

    ``load_restart`` always receives a list from a **fresh** call of
    ``restart_functions()`` on the same object, whose values the caller has
    overwritten with the saved ones. It raises ``ValueError`` if the metadata does not
    describe an object of this kind and configuration.

    An object may also define two methods that are not members of this protocol,
    because not every component has anything to write besides its Functions; the
    checkpointer calls them if they are present (``hasattr``):

    ``write_sidecar(folder: Path, t_ms: float) -> None``
        Write extra files for the checkpoint at ``t_ms`` into ``folder``.
    ``read_sidecar(folder: Path, t_ms: float) -> None``
        Read them back.
    """

    namespace: str

    def restart_functions(self) -> list[tuple[str, dolfinx.fem.Function]]:
        """The ``(name, Function)`` pairs that hold the state, in a fixed order."""
        ...

    def restart_metadata(self) -> dict[str, Any]:
        """The rest of the state, and what identifies the configuration; JSON-able."""
        ...

    def load_restart(
        self,
        functions: Sequence[tuple[str, dolfinx.fem.Function]],
        metadata: Mapping[str, Any],
    ) -> None:
        """Take the state back from ``functions`` (filled) and ``metadata``."""
        ...


@dataclass
class EPState:
    """beat's splitting solver as a component: ``v`` and one ``state_<name>`` per ODE
    state, named by ``state_names`` in the order of the ODE's state array.

    The metadata is ``{"state_names": [...], "solver": ep_solver.restart_metadata()}``.
    The solver's part holds the time step the PDE's matrix was assembled at, without
    which a restored PDE would reassemble it at one that may differ in its last bit.
    """

    ep_solver: beat.MonodomainSplittingSolver
    state_names: Sequence[str]

    namespace = "ep"

    def restart_functions(self) -> list[tuple[str, dolfinx.fem.Function]]:
        """``("v", pde.v)``, the live potential, then fresh copies of the ODE states."""
        return self.ep_solver.restart_functions(self.state_names)

    def restart_metadata(self) -> dict[str, Any]:
        return {
            "state_names": list(self.state_names),
            "solver": self.ep_solver.restart_metadata(),
        }

    def load_restart(
        self,
        functions: Sequence[tuple[str, dolfinx.fem.Function]],
        metadata: Mapping[str, Any],
    ) -> None:
        """Raise ``ValueError`` unless the state names are this solver's, in its order;
        then restore the solver."""
        saved, own = list(metadata.get("state_names", [])), list(self.state_names)
        if saved != own:
            only_saved = sorted(set(saved) - set(own))
            only_own = sorted(set(own) - set(saved))
            difference = (
                f"only in the checkpoint: {only_saved}; only in this run: {only_own}"
                if only_saved or only_own
                else "the same names in another order"
            )
            raise ValueError(f"The EP state names differ ({difference})")
        self.ep_solver.load_restart(functions, metadata["solver"])


@dataclass
class MechanicsState:
    """A ``pulse.StaticProblem`` (or ``DynamicProblem``) as a component.

    Its Functions are the problem's own (``mechanics_*``), so they are restored by
    writing into them; its metadata is ``problem.restart_metadata()``.
    """

    problem: pulse.StaticProblem

    namespace = "mechanics"

    def restart_functions(self) -> list[tuple[str, dolfinx.fem.Function]]:
        return self.problem.restart_functions()

    def restart_metadata(self) -> dict[str, Any]:
        return self.problem.restart_metadata()

    def load_restart(
        self,
        functions: Sequence[tuple[str, dolfinx.fem.Function]],
        metadata: Mapping[str, Any],
    ) -> None:
        """The Functions are the problem's own, already filled; restore the metadata."""
        self.problem.load_restart_metadata(metadata)


@dataclass
class CycleState:
    """A ``pulse.cycle.CycleController`` as a component: no Functions, and its
    ``state_dict()`` as the metadata (in SI, as pulse writes it)."""

    cycle: pulse.cycle.CycleController

    namespace = "cycle"

    def restart_functions(self) -> list[tuple[str, dolfinx.fem.Function]]:
        return []

    def restart_metadata(self) -> dict[str, Any]:
        return self.cycle.state_dict()

    def load_restart(
        self,
        functions: Sequence[tuple[str, dolfinx.fem.Function]],
        metadata: Mapping[str, Any],
    ) -> None:
        self.cycle.load_state_dict(metadata)


@dataclass(frozen=True)
class Snapshot:
    """Copies of the state of a list of components, in memory.

    ``arrays`` maps each namespace to its ``(name, values)`` pairs, in the order of its
    ``restart_functions()``; ``metadata`` maps it to its ``restart_metadata()``.
    """

    arrays: dict[str, list[tuple[str, np.ndarray]]]
    metadata: dict[str, dict[str, Any]]


def take_snapshot(components: Sequence[Checkpointable]) -> Snapshot:
    """Copy every component's restart Functions' values and its metadata.

    Raises ``ValueError`` if two components share a namespace.
    """
    arrays: dict[str, list[tuple[str, np.ndarray]]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for component in components:
        namespace = component.namespace
        if namespace in arrays:
            raise ValueError(f"Two components share the namespace {namespace!r}")
        # Copies: restart_functions() may return the component's live Functions.
        arrays[namespace] = [(name, f.x.array.copy()) for name, f in component.restart_functions()]
        metadata[namespace] = copy.deepcopy(component.restart_metadata())
    return Snapshot(arrays, metadata)


def _difference(saved: Sequence[str], own: Sequence[str]) -> str:
    only_saved = [name for name in saved if name not in own]
    only_own = [name for name in own if name not in saved]
    if only_saved or only_own:
        return f"only in the snapshot: {only_saved}; only in the components: {only_own}"
    return f"the same names in another order: {list(saved)} and {list(own)}"


def restore_snapshot(components: Sequence[Checkpointable], snapshot: Snapshot) -> None:
    """Put a :class:`Snapshot` back into the components it was taken of, in order.

    Each component gets a fresh ``restart_functions()`` list, filled with the saved
    values, and a copy of its saved metadata, through ``load_restart``.

    Raises ``ValueError``, naming the difference and before writing anything, if the
    components' namespaces or any component's function names (or array sizes) differ
    from the snapshot's.
    """
    namespaces = [component.namespace for component in components]
    if namespaces != list(snapshot.arrays):
        raise ValueError(
            "The snapshot is of other components ("
            + _difference(list(snapshot.arrays), namespaces)
            + ")",
        )
    fresh = []
    for component in components:
        functions = component.restart_functions()
        names = [name for name, _ in functions]
        saved = snapshot.arrays[component.namespace]
        saved_names = [name for name, _ in saved]
        if names != saved_names:
            raise ValueError(
                f"The snapshot's {component.namespace!r} functions are not this "
                f"component's ({_difference(saved_names, names)})",
            )
        for (name, f), (_, values) in zip(functions, saved):
            if f.x.array.shape != values.shape:
                raise ValueError(
                    f"The snapshot holds {values.shape} values of {name!r}, the "
                    f"component {f.x.array.shape}",
                )
        fresh.append(functions)
    for component, functions in zip(components, fresh):
        for (_, f), (_, values) in zip(functions, snapshot.arrays[component.namespace]):
            f.x.array[:] = values
        component.load_restart(functions, copy.deepcopy(snapshot.metadata[component.namespace]))
