"""The activation-backend interface.

An *activation backend* owns everything about how active tension is generated
and how it reaches the mechanics solve. Different backends make genuinely
different numerical choices -- where the EP/mechanics split is cut, and whether
the force-generation model is coupled monolithically or in a segregated way --
but they present one interface, so a simulation can swap between them by
changing one line.

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
from typing import Mapping, Protocol

import dolfinx
import ufl


@dataclass(frozen=True)
class Transfer:
    """One variable crossing between the EP and mechanics subsystems.

    A bare name is not enough to drive a transfer. The direction is carried by
    which method returns the ``Transfer`` (:meth:`ActivationBackend.wants_from_ep`
    or :meth:`ActivationBackend.gives_to_ep`); the rest is here.

    Attributes
    ----------
    name:
        The variable's name in the generated ODE module, e.g. ``"cai"`` or
        ``"XS"``. The coupler resolves it against that module's ``missing``
        mapping, which is what gives the variable its row in the positional
        transfer buffers.
    unit:
        The unit *as the producing side emits it*, e.g. ``"mM"`` for ToR-ORd's
        ``cai``. The consumer may want something else -- crossbridge wants
        micromolar -- and carrying the unit here is what lets the coupler
        convert explicitly instead of relying on a lookup table keyed
        invisibly by name.

        Defaults to ``"1"``, dimensionless, spelled as the ``.ode`` files spell
        it. When the ODE source declares no unit for a variable the coupler
        assumes this one and warns, rather than proceeding silently.

        There is deliberately no state-or-monitor field. Neither direction
        needs one: the forward path goes through the generated
        ``missing_values`` function, which requires no index lookup, and the
        backward path writes positionally into the EP side's missing array. A
        variable crossing back is an EP *missing* variable regardless of how
        the producing side derived it.
    """

    name: str
    unit: str = "1"


class ActivationBackend(Protocol):
    """What the coupler and the mechanics problem require of a backend.

    Lifecycle, once per mechanics time step::

        backend.step(t, dt)     # advance activation using the EP inputs and
                                # the stretch from the *previous* solve
        problem.solve()         # Newton
        backend.post_solve()    # record the new stretch; advance anything
                                # that needed the updated displacement

    The ordering is not cosmetic. Whatever stretch a backend uses to advance
    its activation in ``step`` must be the same one it measures the increment
    against afterwards, or a stabilized scheme stops being consistent.
    """

    # -- pulse.ActiveModel ---------------------------------------------------

    def S(self, C: ufl.core.expr.Expr, dev: bool = False) -> ufl.core.expr.Expr:
        """Active second Piola-Kirchhoff stress."""
        ...

    def P(self, F: ufl.core.expr.Expr, dev: bool = False) -> ufl.core.expr.Expr:
        """Active first Piola-Kirchhoff stress."""
        ...

    def Fe(self, F: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        """Elastic part of the deformation gradient (identity for active stress)."""
        ...

    def register(self, u: dolfinx.fem.Function) -> None:
        """Receive the displacement field. Called by ``pulse.StaticProblem``."""
        ...

    # -- coupling ------------------------------------------------------------

    def wants_from_ep(self) -> tuple[Transfer, ...]:
        """Variables this backend needs transferred from the EP subsystem."""
        ...

    def gives_to_ep(self) -> tuple[Transfer, ...]:
        """Variables this backend supplies back to the EP subsystem.

        Rarely empty, and never safely ignored: a split that moves calcium
        buffering out of the EP model has to return the buffering flux, or the
        EP calcium transient is wrong with nothing raised.
        """
        ...

    @property
    def ep_inputs(self) -> Mapping[str, dolfinx.fem.Function]:
        """Functions the coupler fills with the :meth:`wants_from_ep` variables.

        The backend owns these, on whatever space it evaluates activation on,
        so the coupler interpolates into them rather than deciding where they
        live.
        """
        ...

    @property
    def ep_outputs(self) -> Mapping[str, dolfinx.fem.Function]:
        """Functions holding the :meth:`gives_to_ep` variables, for the coupler
        to transfer back to the EP mesh.

        The mirror of :attr:`ep_inputs`. Kept symmetric because the return path
        is as load-bearing as the forward one and is easier to forget.
        """
        ...

    def step(self, t: float, dt: float | None = None) -> None:
        """Advance activation to time ``t``, before the mechanics solve."""
        ...

    def post_solve(self) -> None:
        """Update state that depends on the just-computed displacement."""
        ...

    @property
    def active_tension(self) -> dolfinx.fem.Function:
        """Active tension of the last completed step, for output.

        Named in full rather than ``Ta`` because backends may already use that
        name for a UFL-valued method of the stretch.
        """
        ...
