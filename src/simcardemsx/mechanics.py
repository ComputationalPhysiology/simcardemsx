"""How :class:`~simcardemsx.controller.SimulationController` advances mechanics.

The controller does not call ``pulse.StaticProblem.solve()`` directly; it calls
``advance`` on a *driver* instead. :class:`Solve` is the plain driver -- it just
calls ``problem.solve()`` -- kept as the default so existing callers see no change
in behaviour. Later drivers wrap a problem with something more: setting a
circulation's clock before solving, or running a cavity cycle controller around
the solve. Whatever a driver does, ``SimulationController`` only ever needs the
``pulse.StaticProblem`` underneath it (for its guards, and to read the Newton
iteration count) and a way to advance it by one mechanics time step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import pulse


@runtime_checkable
class MechanicsDriver(Protocol):
    """What :class:`~simcardemsx.controller.SimulationController` requires of a driver.

    ``problem`` is read by the controller's construction-time guards (backend
    identity, quadrature degree, the ``DynamicProblem`` time-step check) and, after
    each solve, for the Newton iteration count -- always the underlying
    ``pulse.StaticProblem``, never the driver itself.
    """

    @property
    def problem(self) -> pulse.StaticProblem:
        """The underlying mechanics problem. ``DynamicProblem`` is a subclass."""
        ...

    def advance(self, t_n: float, dt: float) -> bool:
        """Advance ``problem`` from ``t_n`` by ``dt`` (both in ms).

        Returns
        -------
        bool
            ``True`` if the step converged.
        """
        ...


@dataclass
class Solve:
    """The plain driver: ``advance`` just solves ``problem``, ignoring ``t_n``/``dt``.

    ``as_driver`` wraps a bare ``pulse.StaticProblem`` in this, so a caller that does
    not need a driver of its own sees the same behaviour as calling
    ``problem.solve()`` from the controller directly.
    """

    problem: pulse.StaticProblem

    def advance(self, t_n: float, dt: float) -> bool:
        return self.problem.solve()


def as_driver(mechanics: MechanicsDriver | pulse.StaticProblem) -> MechanicsDriver:
    """Return ``mechanics`` as a :class:`MechanicsDriver`.

    Anything that already looks like a driver (has both ``problem`` and
    ``advance``) is returned as is; a bare ``pulse.StaticProblem`` (or
    ``DynamicProblem``) is wrapped in :class:`Solve`.
    """
    if isinstance(mechanics, MechanicsDriver):
        return mechanics
    return Solve(mechanics)
