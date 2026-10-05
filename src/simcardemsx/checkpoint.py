"""Checkpoint and restart: what a component must offer to be saved and restored.

A component is :class:`Checkpointable` when it can name the ``dolfinx`` Functions that
hold its state, describe the rest of it as JSON, and take both back. A checkpointer
writes every component's :meth:`~Checkpointable.restart_functions` to one file and
their :meth:`~Checkpointable.restart_metadata` to another, and on restart hands each
component a fresh call of ``restart_functions()`` whose values it has overwritten.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

import dolfinx


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
