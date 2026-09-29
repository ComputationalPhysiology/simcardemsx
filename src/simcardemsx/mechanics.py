"""How :class:`~simcardemsx.controller.SimulationController` advances mechanics.

The controller does not call ``pulse.StaticProblem.solve()`` directly; it calls
``advance`` on a *driver* instead. :class:`Solve` is the plain driver -- it just
calls ``problem.solve()`` -- kept as the default so existing callers see no change
in behaviour. Other drivers wrap a problem with something more:
:class:`CirculationClock` sets a closed-loop circuit's clock before solving.
Whatever a driver does, ``SimulationController`` only ever needs the
``pulse.StaticProblem`` underneath it (for its guards, and to read the Newton
iteration count) and a way to advance it by one mechanics time step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    import dolfinx
    import pulse

#: The factor converting the controller's ms into each time unit a circuit may be
#: written in.
_PER_MS = {"s": 1e-3, "ms": 1.0}


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


@dataclass
class CirculationClock:
    """Drive a problem's closed-loop circuit on the controller's clock.

    pulse gives the circuit (``problem.circulation``) a clock of its own, in the
    ``.ode`` file's own time unit: ``problem.circulation_time`` and
    ``problem.circulation_dt`` are constants the caller sets. A circuit whose
    activation phase would be computed with ``Mod`` needs that phase supplied from
    outside instead, since UFL has no ``Mod``: Regazzoni's, with its ``timing``
    component dropped, takes it as ``beat_phase``.

    ``advance(t_n, dt)`` (both in ms, as everywhere in the controller) sets these for
    the step to ``t_{n+1} = t_n + dt``, then solves: ``circulation_time`` to
    ``t_{n+1}``, since the circuit's right-hand side is evaluated at the end of the
    step; ``circulation_dt`` to ``dt``; and ``beat_phase``, if given, to
    ``t_{n+1} mod period``. All three are converted to ``time_unit``.

    Parameters
    ----------
    problem:
        The mechanics problem. Must have a circulation.
    time_unit:
        The circuit's own time unit, ``"s"`` (Regazzoni's) or ``"ms"``.
    beat_phase:
        The ``Constant`` the problem's ``circulation_missing`` supplies as the
        circuit's beat phase, or ``None`` for a circuit that needs none.
    period:
        The beat length, in ms. Required if and only if ``beat_phase`` is given, and
        it must match the circuit's own (for Regazzoni's, ``RR = 1 / HR``).

    Raises
    ------
    ValueError
        If ``problem`` has no circulation, ``time_unit`` is neither ``"s"`` nor
        ``"ms"``, or only one of ``beat_phase`` and ``period`` is given.
    """

    problem: pulse.StaticProblem
    time_unit: str = "s"
    beat_phase: dolfinx.fem.Constant | None = None
    period: float | None = None

    def __post_init__(self) -> None:
        if self.problem.circulation is None:
            raise ValueError(
                "CirculationClock drives a closed-loop circulation, but the problem has "
                "none (problem.circulation is None); use Solve for a problem without one.",
            )
        if self.time_unit not in _PER_MS:
            raise ValueError(
                f"Unknown time_unit {self.time_unit!r}: the circuit's clock must be in one "
                f"of {sorted(_PER_MS)}.",
            )
        if (self.beat_phase is None) != (self.period is None):
            missing = "period" if self.period is None else "beat_phase"
            raise ValueError(
                "beat_phase and period must be given together, since the beat phase is "
                f"t mod period; {missing} is missing.",
            )

    def advance(self, t_n: float, dt: float) -> bool:
        scale = _PER_MS[self.time_unit]
        t_next = t_n + dt
        self.problem.circulation_time.value = np.asarray(t_next * scale)
        self.problem.circulation_dt.value = np.asarray(dt * scale)
        if self.beat_phase is not None:
            assert self.period is not None  # guarded in __post_init__
            self.beat_phase.value = np.asarray((t_next % self.period) * scale)
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
