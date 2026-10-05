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

On disk, a :class:`Checkpointer` writes them in the upstream CLIs' layout: every
Function into :data:`RESTART` (``restart.bp``, through io4dolfinx, in input-mesh order)
and every metadata value into :data:`RESTART_META` (``restart.json``), written last and
atomically, so that it only ever names a complete checkpoint.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from mpi4py import MPI

import dolfinx
import io4dolfinx
import numpy as np

from .provenance import provenance

if TYPE_CHECKING:
    import beat
    import pulse
    import pulse.cycle

    from .controller import SimulationController


@runtime_checkable
class Checkpointable(Protocol):
    """A component whose state can be written to disk and read back.

    Attributes
    ----------
    namespace
        The component's key in a snapshot and in a checkpoint's metadata. It must be
        unique among the components saved together.

    Notes
    -----
    Restart function names must be unique across all the components saved together,
    not only within one. simcardemsx's own components prefix theirs with their
    namespace (``activation_...``, ``transfer_...``), and pulse's are
    ``mechanics_...``. EP's are beat's own (``v``, ``state_<name>``) and are not
    prefixed.

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


def _difference(
    saved: Sequence[str],
    own: Sequence[str],
    saved_in: str = "the snapshot",
    own_in: str = "the components",
) -> str:
    only_saved = [name for name in saved if name not in own]
    only_own = [name for name in own if name not in saved]
    if only_saved or only_own:
        return f"only in {saved_in}: {only_saved}; only in {own_in}: {only_own}"
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


# ----------------------------------------------------------------------
# On disk: restart.bp and restart.json
# ----------------------------------------------------------------------

#: The checkpoint's Functions, written by io4dolfinx into the run's output folder.
RESTART = "restart.bp"
#: The checkpoint's metadata. Written last, and atomically, so it names only a
#: checkpoint whose Functions are all in :data:`RESTART`.
RESTART_META = "restart.json"
#: The namespace in ``restart.json`` that holds the controller's metadata and the
#: checkpointer's own keys: :class:`~simcardemsx.controller.SimulationController`'s.
NAMESPACE = "simcardemsx"
#: The keys the checkpointer adds to :data:`NAMESPACE`'s entry. No component's metadata
#: may use them.
RESERVED_KEYS = ("physics_hash", "functions", "provenance", "history")

#: The directory provenance is taken in: simcardemsx's own source, for its git commit.
_PACKAGE_DIR = Path(__file__).resolve().parent


def write_json(path: Path, data: Mapping[str, Any], comm: MPI.Comm = MPI.COMM_WORLD) -> None:
    """Write ``data`` to ``path`` as JSON, atomically, on rank 0 of ``comm``.

    The JSON goes into a temporary file beside ``path``, which ``os.replace`` then moves
    onto it, so ``path`` holds either the old document or the new one, whole, even if
    the process is killed meanwhile. Rank 0 then broadcasts whether it succeeded, which
    is the barrier: no rank returns before the file is in place, and if rank 0 fails,
    every rank raises (rank 0 its own error, the others ``OSError``).
    """
    failure: Exception | None = None
    if comm.rank == 0:
        tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
        try:
            tmp.write_text(json.dumps(dict(data), indent=2))
            os.replace(tmp, path)
        except Exception as error:
            failure = error
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)
    message = comm.bcast(None if failure is None else repr(failure), root=0)
    if failure is not None:
        raise failure
    if message is not None:
        raise OSError(f"Rank 0 could not write {path}: {message}")


def physics_hash(physics: Mapping[str, Any]) -> str:
    """The sha256 of ``physics`` as JSON with sorted keys, as the upstream CLIs hash
    their configuration. Values JSON cannot hold are hashed as their ``str``."""
    return hashlib.sha256(
        json.dumps(dict(physics), sort_keys=True, default=str).encode(),
    ).hexdigest()


def check_restart(folder: Path, physics: Mapping[str, Any]) -> None:
    """Refuse to restart a run of ``physics`` from the checkpoint in ``folder`` unless
    that checkpoint was written with the same physics.

    Raises
    ------
    FileNotFoundError
        If ``folder / RESTART_META`` does not exist: ``folder`` holds no complete
        checkpoint.
    ValueError
        If the checkpoint's ``physics_hash`` is not :func:`physics_hash` of ``physics``,
        or it has none (it was not written by a :class:`Checkpointer`).
    """
    folder = Path(folder)
    path = folder / RESTART_META
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint to restart from: {path} does not exist")
    stored = json.loads(path.read_text()).get(NAMESPACE, {}).get("physics_hash")
    if stored is None:
        raise ValueError(
            f"{path} has no {NAMESPACE}.physics_hash: it was not written by simcardemsx's "
            "Checkpointer",
        )
    own = physics_hash(physics)
    if stored != own:
        raise ValueError(
            f"Cannot restart: the physics differ from the checkpoint's (physics_hash "
            f"{stored} in {path}, {own} for this run). The checkpointed run's "
            f"configuration is in {folder / 'config.resolved.toml'}.",
        )


def _names_difference(
    saved: Mapping[str, Sequence[str]],
    own: Mapping[str, Sequence[str]],
) -> str:
    """How two ``{namespace: function names}`` dicts differ, namespace by namespace."""
    parts = []
    for namespace in [*saved, *(namespace for namespace in own if namespace not in saved)]:
        if namespace not in own:
            parts.append(f"{namespace!r} only in the checkpoint")
        elif namespace not in saved:
            parts.append(f"{namespace!r} only in this run")
        elif list(saved[namespace]) != list(own[namespace]):
            difference = _difference(
                saved[namespace],
                own[namespace],
                "the checkpoint",
                "this run",
            )
            parts.append(f"{namespace!r}: {difference}")
    return "; ".join(parts)


class Checkpointer:
    """Write a coupled run to ``folder`` and read it back: ``restart.bp`` and
    ``restart.json``, in the upstream CLIs' layout.

    Its components are ``controller.components()`` followed by ``extra``. A checkpoint
    holds each component's restart Functions in :data:`RESTART`, at ``t =
    controller.t`` (ms), under their own names, and its metadata in
    :data:`RESTART_META`, one entry per namespace. The :data:`NAMESPACE` entry, the
    controller's, also holds the checkpointer's own keys (:data:`RESERVED_KEYS`):
    ``physics_hash``, ``functions`` (each namespace's function names), ``provenance``
    (of the process that wrote it) and ``history``.

    Parameters
    ----------
    controller:
        The coupled run.
    folder:
        Where the checkpoint is written and read.
    physics:
        Everything that defines the run's physics, and nothing about its length, its
        output or its solver options. Only its :func:`physics_hash` is stored, and a
        restore refuses a checkpoint written with another, so a restart may change the
        end time or the solver options, as with the CLIs.
    extra:
        More components, saved after the controller's.

    Attributes
    ----------
    components:
        ``controller.components()`` followed by ``extra``.
    provenance:
        This process's :func:`~simcardemsx.provenance.provenance`.
    history:
        The provenance of every process that wrote this run or resumed it, oldest
        first: ``[provenance]`` until :meth:`restore` puts the stored history before it.

    Raises
    ------
    ValueError
        If two components share a namespace or a restart-function name, or a
        component's metadata uses one of :data:`RESERVED_KEYS`.

    Notes
    -----
    Times that :data:`RESTART` holds for every function are complete checkpoints.
    :meth:`write` writes no Functions at a time already complete there (io4dolfinx
    would append a duplicate, and read the first one back), and only rewrites
    ``restart.json``. A fresh checkpointer knows of none; :meth:`restore` reads them.
    So a fresh one assumes that ``folder`` holds no ``restart.bp`` of another run.
    """

    def __init__(
        self,
        controller: SimulationController,
        folder: Path,
        *,
        physics: Mapping[str, Any],
        extra: Sequence[Checkpointable] = (),
    ):
        self.controller = controller
        self.folder = Path(folder)
        self.physics = physics
        self.components: list[Checkpointable] = [*controller.components(), *extra]
        self.comm = controller.mechanics.problem.geometry.mesh.comm
        self._check_components()
        self.provenance = provenance(_PACKAGE_DIR, self.comm)
        self.history: list[dict[str, Any]] = [self.provenance]
        self._complete: list[float] = []

    def _check_components(self) -> None:
        namespaces: set[str] = set()
        owners: dict[str, str] = {}
        shared: list[str] = []
        for component in self.components:
            namespace = component.namespace
            if namespace in namespaces:
                raise ValueError(f"Two components share the namespace {namespace!r}")
            namespaces.add(namespace)
            for name, _ in component.restart_functions():
                if name in owners:
                    shared.append(f"{name!r} ({owners[name]!r} and {namespace!r})")
                owners.setdefault(name, namespace)
            reserved = [key for key in RESERVED_KEYS if key in component.restart_metadata()]
            if reserved:
                raise ValueError(
                    f"The {namespace!r} component's metadata uses {reserved}, which the "
                    f"checkpointer reserves for its own keys: {list(RESERVED_KEYS)}",
                )
        if shared:
            raise ValueError(
                "Restart function names must be unique across components, but these are "
                f"shared: {', '.join(shared)}",
            )

    def _tolerance(self) -> float:
        """Two times (ms) this close are one checkpoint's, as the CLIs compare them."""
        return 1e-9 * self.controller.dt_mech

    def _holds(self, times: Sequence[float] | np.ndarray, t: float) -> bool:
        return bool(np.any(np.abs(np.asarray(times, dtype=float) - t) < self._tolerance()))

    def _stored_times(self, name: str) -> np.ndarray:
        """Every time :data:`RESTART` holds ``name`` at, duplicates included."""
        times = io4dolfinx.read_timestamps(
            filename=self.folder / RESTART,
            comm=self.comm,
            function_name=name,
        )
        return np.asarray(times, dtype=float)

    def _functions(
        self,
    ) -> list[tuple[Checkpointable, list[tuple[str, dolfinx.fem.Function]]]]:
        """Each component with a fresh call of its ``restart_functions()``."""
        return [(component, component.restart_functions()) for component in self.components]

    def write(self) -> None:
        """Write a checkpoint of the run at ``t = controller.t``.

        In order: every component's Functions into :data:`RESTART` at ``t``, unless
        ``t`` is already complete there; each component's ``write_sidecar(folder, t)``,
        where defined; and :data:`RESTART_META`, last and atomically
        (:func:`write_json`), so that it names this checkpoint only once all of it is
        on disk.

        Raises
        ------
        RuntimeError
            Before writing anything, if the backend has a step pending
            (``begin_step`` without ``post_solve``): checkpoint between steps.
        """
        t = self.controller.t
        # Both CoupledBackends have step_pending; the protocol does not say so.
        if cast(Any, self.controller.backend).step_pending:
            raise RuntimeError(
                f"Cannot checkpoint at t = {t} ms: the backend has a step pending "
                "(begin_step without post_solve). Checkpoint between steps.",
            )
        functions = self._functions()
        if not self._holds(self._complete, t):
            for _, pairs in functions:
                for name, f in pairs:
                    io4dolfinx.write_function_on_input_mesh(
                        self.folder / RESTART,
                        f,
                        time=t,
                        name=name,
                    )
            self._complete.append(t)
        for component in self.components:
            write_sidecar = getattr(component, "write_sidecar", None)
            if write_sidecar is not None:
                write_sidecar(self.folder, t)

        metadata = {
            component.namespace: component.restart_metadata() for component in self.components
        }
        metadata[NAMESPACE] = {
            **metadata.get(NAMESPACE, {}),
            "physics_hash": physics_hash(self.physics),
            "functions": {
                component.namespace: [name for name, _ in pairs] for component, pairs in functions
            },
            "provenance": self.provenance,
            "history": self.history,
        }
        write_json(self.folder / RESTART_META, metadata, self.comm)

    def restore(self) -> float:
        """Restore the run from the checkpoint :data:`RESTART_META` names, and return
        its ``t`` (ms).

        In order: the physics are checked (:func:`check_restart`), then the function
        names and the time; every component's Functions, from a fresh
        ``restart_functions()`` call, are read from :data:`RESTART` at ``t``; each
        component's ``load_restart`` is called, in order, with its metadata as written;
        then each component's ``read_sidecar(folder, t)``, where defined. Nothing is
        moved back to EP afterwards (``plan.backward()`` is not called): the rows of
        EP's arrays that come from the backend are the transfer plan's own restart
        Functions. :attr:`history` becomes the stored history followed by this
        process's provenance.

        Each Function is read at the stored time closest to ``t``: io4dolfinx reads at
        an exact time, and returns the first of duplicates.

        Raises
        ------
        FileNotFoundError, ValueError
            From :func:`check_restart`.
        ValueError
            If the checkpoint's function names differ from this run's, listing the
            differences, or if :data:`RESTART` does not hold ``t`` for every function,
            listing the times it does hold for every one. Both before anything is read.
        """
        check_restart(self.folder, self.physics)
        meta = json.loads((self.folder / RESTART_META).read_text())
        own = meta[NAMESPACE]

        functions = self._functions()
        names = {component.namespace: [name for name, _ in pairs] for component, pairs in functions}
        if own["functions"] != names:
            raise ValueError(
                f"The checkpoint in {self.folder} holds other functions than this run "
                f"({_names_difference(own['functions'], names)})",
            )

        t = float(own["t_ms"])
        stored = {name: self._stored_times(name) for pairs in names.values() for name in pairs}
        complete = self._complete_times(stored)
        missing = [name for name, times in stored.items() if not self._holds(times, t)]
        if missing:
            which = "any function" if len(missing) == len(stored) else str(missing)
            raise ValueError(
                f"{self.folder / RESTART} holds no checkpoint at t = {t} ms, which "
                f"{RESTART_META} names, for {which}. It holds complete checkpoints at "
                f"t = {complete} ms.",
            )

        for _, pairs in functions:
            for name, f in pairs:
                times = stored[name]
                io4dolfinx.read_function(
                    self.folder / RESTART,
                    f,
                    time=float(times[np.argmin(np.abs(times - t))]),
                    name=name,
                )
                f.x.scatter_forward()
        for component, pairs in functions:
            metadata = {
                key: value
                for key, value in meta[component.namespace].items()
                if key not in RESERVED_KEYS
            }
            component.load_restart(pairs, metadata)
        for component in self.components:
            read_sidecar = getattr(component, "read_sidecar", None)
            if read_sidecar is not None:
                read_sidecar(self.folder, t)

        self._complete = complete
        self.history = [*own["history"], self.provenance]
        return t

    def _complete_times(self, stored: Mapping[str, np.ndarray]) -> list[float]:
        """The times held by every one of ``stored``'s names, sorted."""
        if not stored:
            return []
        first, *rest = stored.values()
        return [
            float(t)
            for t in np.unique(first)
            if all(self._holds(times, float(t)) for times in rest)
        ]
