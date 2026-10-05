"""The activation-backend interface.

An *activation backend* owns everything about how active tension is generated
and how it reaches the mechanics solve. Different backends make genuinely
different numerical choices -- where the EP/mechanics split is cut, and whether
the force-generation model is coupled monolithically or in a segregated way.

The controller drives a backend through the :class:`CoupledBackend` protocol below;
what crosses between EP and activation is derived from the EP module and the backend's
``missing``/``provides`` by :func:`simcardemsx.transfer_plan.resolve` and moved by
:class:`simcardemsx.transfer_plan.TransferPlan`. :class:`~simcardemsx.backends.generated.
GeneratedActivation` and :class:`~simcardemsx.backends.segregated.CrossbridgeSegregated`
are two; ``CrossbridgeSegregated`` also documents the units of its crossings with
:class:`Transfer` records. :class:`ZetaSplitUFL` is driven directly against ``pulse``,
and declares its crossings with :class:`Transfer` instead.

That matters more than usual here, because the choices are not equivalent.
A segregated coupling of force generation to mechanics is unstable, and in
fact not convergent, once active stiffness exceeds passive stiffness
(Regazzoni & Quarteroni 2021); a monolithic one is stable but expensive. Having
the alternatives behind one interface is what makes them comparable on the same
mesh, in the same process, from the same configuration, rather than by
comparing published numbers.

A backend is also a ``pulse.ActiveModel``: it supplies ``S``/``P`` and is handed
straight to ``pulse.CardiacModel``. Note that ``S`` is the primary contract, not
``strain_energy``. Only a segregated backend has a clean potential to
differentiate; for the zeta split ``Ta`` depends on the stretch through
``zeta_s(lambda_dot)``, so no closed-form potential exists, and for an
external-operator backend ``Ta`` is opaque by construction. This is fine:
``pulse.StaticProblem`` assembles ``model.S(C)`` and never touches
``strain_energy``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Protocol, runtime_checkable

import dolfinx


@dataclass(frozen=True)
class Transfer:
    """One variable crossing between the EP and mechanics subsystems.

    A bare name is not enough to drive a transfer. The direction is carried by
    which backend method returns the ``Transfer`` (``wants_from_ep`` or
    ``gives_to_ep``); the rest is here.

    Attributes
    ----------
    name:
        The variable's name in the generated ODE module, e.g. ``"cai"`` or
        ``"XS"``. Used to look up its index via ``state_index``/``monitor_index``.
    unit:
        The unit *as the producing side emits it*, e.g. ``"mM"`` for ToR-ORd's
        ``cai``. The consumer may want something else -- crossbridge wants
        micromolar -- and carrying the unit here is what lets the coupler
        convert explicitly instead of relying on a lookup table keyed
        invisibly by name.
    kind:
        Whether ``name`` is an ODE state or a monitored (derived) expression.
        They need different index lookups in the generated module.
    """

    name: str
    unit: str = "dimensionless"
    kind: Literal["state", "monitor"] = "state"


@runtime_checkable
class CoupledBackend(Protocol):
    """What :class:`~simcardemsx.controller.SimulationController` requires of a backend.

    Once per mechanics time step the controller calls, in this order::

        backend.begin_step(t_n, dt)   # t_n is the old time, dt the mechanics step (ms)
        mechanics.advance(t_n, dt)    # the Newton solve
        backend.post_solve()          # accept the step
        # then TransferPlan.backward() reads ``outputs``

    Nothing is accepted before ``post_solve``: until then the backend's stored states,
    ``outputs`` and ``active_tension`` are those of the last accepted step, so a step
    whose solve fails leaves them untouched. ``begin_step`` only prepares the step.

    What crosses between EP and the backend is derived by
    :func:`simcardemsx.transfer_plan.resolve` from ``missing`` and ``provides``, the
    name -> index dicts of the generated activation module, and moved through
    ``inputs`` (EP -> backend, filled before ``begin_step``) and ``outputs``
    (backend -> EP, read after ``post_solve``), all on ``space``.
    """

    @property
    def missing(self) -> Mapping[str, int]:
        """What the activation module needs from EP, name -> index."""
        ...

    @property
    def provides(self) -> Mapping[str, int]:
        """What the activation module hands to EP, name -> index."""
        ...

    @property
    def quadrature_degree(self) -> int | None:
        """The degree of the quadrature space the states live on, if they live on one."""
        ...

    inputs: dict[str, dolfinx.fem.Function]
    outputs: dict[str, dolfinx.fem.Function]
    space: dolfinx.fem.FunctionSpace
    active_tension: dolfinx.fem.Function

    def begin_step(self, t_n: float, dt: float) -> None:
        """Prepare the step from ``t_n`` to ``t_n + dt`` (ms)."""
        ...

    def post_solve(self) -> None:
        """Accept the converged step."""
        ...
