"""The coupled time step: EP micro-steps, then one mechanics solve, values crossing both ways.

What crosses is derived from the EP module and the backend's ``missing``/``provides``
(:func:`~simcardemsx.transfer_plan.resolve`), and moved by a
:class:`~simcardemsx.transfer_plan.TransferPlan`. The backend is a
:class:`~simcardemsx.backends.base.CoupledBackend`:
:class:`~simcardemsx.backends.GeneratedActivation` or
:class:`~simcardemsx.backends.CrossbridgeSegregated`. The deprecated
:class:`~simcardemsx.backends.ZetaSplitUFL` remains usable directly against ``pulse``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

import beat
import pulse

from .mechanics import as_driver
from .transfer_plan import TransferPlan, resolve

if TYPE_CHECKING:
    from .backends.base import CoupledBackend
    from .mechanics import MechanicsDriver
    from .ode_model import ODEModules

logger = logging.getLogger(__name__)

#: Relative tolerance on ``dt_mech == n * dt_ep``: round-off, not a real remainder.
_DT_RTOL = 1e-9


class SimulationController:
    """Advance EP and mechanics together, one mechanics time step at a time.

    Parameters
    ----------
    mechanics:
        The mechanics driver, or a bare ``pulse.StaticProblem``/``DynamicProblem``
        (wrapped in :class:`~simcardemsx.mechanics.Solve`, so its solver options
        remain the caller's: the plain driver only calls ``problem.solve()``).
    ep_solver:
        beat's splitting solver; ``ep_solver.ode`` is the ODE solver whose arrays the
        plan reads and writes.
    backend:
        The activation backend. Must be ``mechanics.problem.model.active``.
    ode_modules:
        The two modules generated from the ``.ode`` file ``ep_solver`` was built from
        (and a ``GeneratedActivation`` backend, from its ``mechanics``). Only
        ``ode_modules.ep`` is read: what crosses is resolved between it and the
        backend, and ``ode_modules.mechanics`` is no longer read.
    dt_mech, dt_ep:
        Mechanics and EP time steps, in ms. ``dt_mech`` must be a whole multiple of
        ``dt_ep``.

    Raises
    ------
    ValueError
        If ``backend`` is not the problem's active model, if ``dt_mech`` is not a
        multiple of ``dt_ep``, if the problem is a ``pulse.DynamicProblem`` whose
        ``parameters["dt"]`` does not equal ``dt_mech``, or if the backend stores its
        states on a quadrature space whose degree differs from the mechanics form's.
    TypeError
        If ``ep_solver.ode`` is not a :class:`beat.odesolver.DolfinODESolver`.
    NotImplementedError
        From :class:`~simcardemsx.transfer_plan.TransferPlan`, if the EP ODE space is
        neither P1 nor DG0.
    """

    def __init__(
        self,
        mechanics: MechanicsDriver | pulse.StaticProblem,
        ep_solver: beat.MonodomainSplittingSolver,
        backend: CoupledBackend,
        ode_modules: ODEModules,
        dt_mech: float,
        dt_ep: float,
    ):
        self.mechanics = as_driver(mechanics)
        problem = self.mechanics.problem

        if backend is not problem.model.active:
            raise ValueError(
                "backend must be the active model of the mechanics problem "
                "(mechanics.problem.model.active): the controller steps the backend and "
                "the problem solves with its own, so they must be the same object.",
            )

        ep_steps_per_mech = round(dt_mech / dt_ep)
        if ep_steps_per_mech < 1 or abs(ep_steps_per_mech * dt_ep - dt_mech) > _DT_RTOL * dt_mech:
            raise ValueError(
                f"dt_mech ({dt_mech}) must be a whole multiple of dt_ep ({dt_ep})",
            )

        if isinstance(problem, pulse.DynamicProblem):
            # dt is a pint Variable in s; the controller's clock is in ms throughout.
            problem_dt_ms = problem.parameters["dt"].to_base_units() * 1e3
            if abs(problem_dt_ms - dt_mech) > _DT_RTOL * dt_mech:
                raise ValueError(
                    f"problem.parameters['dt'] ({problem_dt_ms} ms) must equal dt_mech "
                    f"({dt_mech} ms): pulse.DynamicProblem's inertia term is assembled "
                    "against its own dt, so a mismatch would silently step the mechanics "
                    "clock and the controller's clock apart.",
                )

        if backend.space.ufl_element().family_name == "quadrature":
            # The measure every integral of the pulse form uses, so its metadata is the
            # degree the form asks for.
            geometry_degree = problem.geometry.dx.metadata().get("quadrature_degree")
            if backend.quadrature_degree != geometry_degree:
                raise ValueError(
                    f"The backend stores its states at quadrature degree "
                    f"{backend.quadrature_degree}, but the mechanics form integrates at "
                    f"quadrature degree {geometry_degree} (geometry.dx). They must "
                    "be equal: FFCx evaluates the whole integrand at a quadrature-element "
                    "coefficient's own degree and ignores the measure's quadrature_degree, so "
                    "a mismatch would silently change the quadrature of the whole momentum "
                    "integral, not just which states are looked up.",
                )

        # beat types this as its generic ODE-solver protocol. The transfer plan needs
        # the one-model solver: it writes into its missing_variables and parameters arrays.
        ode = ep_solver.ode
        if not isinstance(ode, beat.odesolver.DolfinODESolver):
            raise TypeError(
                f"ep_solver.ode must be a beat.odesolver.DolfinODESolver, got {type(ode).__name__}",
            )

        self.ep_solver = ep_solver
        self.backend = backend
        self.ode_modules = ode_modules
        self.dt_mech = dt_mech
        self.dt_ep = dt_ep
        self.ep_steps_per_mech = ep_steps_per_mech

        self.plan = TransferPlan(
            resolve(ode_modules.ep, backend),
            ode_modules.ep,
            ode,
            backend,
        )

        self.t = 0.0
        self.ep_step_idx = 0
        self.mech_step_idx = 0

    def step(
        self,
        ep_callback: Callable[[float, int], None] | None = None,
        mech_callback: Callable[[float, int, int], None] | None = None,
    ) -> None:
        """Advance the coupled system by one mechanics time step, from ``t`` to ``t + dt_mech``.

        In order: the EP micro-steps (``ep_callback(t, ep_step_idx)`` after each); EP
        values forward into the backend's inputs, at the new ``t``; ``self.mechanics.
        advance()``, with the backend stepping from the old ``t`` by ``dt_mech``;
        ``post_solve()``; the backend's outputs back into EP's arrays; ``mech_callback(t,
        mech_step_idx, newton_iterations)``.

        Raises ``RuntimeError`` if the mechanics driver's ``advance`` does not converge.
        """
        t_n = self.t
        logger.info(f"--- Solving coupled step from t = {t_n} ---")

        for i in range(self.ep_steps_per_mech):
            t0 = self.t
            self.t = t_n + (i + 1) * self.dt_ep
            self.ep_solver.step((t0, self.t))
            self.ep_step_idx += 1
            if ep_callback:
                ep_callback(self.t, self.ep_step_idx)

        self.plan.forward(self.t)

        self.backend.begin_step(t_n, self.dt_mech)
        ok = self.mechanics.advance(t_n, self.dt_mech)
        if not ok:
            raise RuntimeError(
                f"The mechanics solve did not converge for the step from t = {t_n} to t = {self.t}",
            )
        self.backend.post_solve()
        self.plan.backward()
        self.mech_step_idx += 1

        if mech_callback:
            mech_callback(
                self.t,
                self.mech_step_idx,
                self.mechanics.problem.problem.solver.getIterationNumber(),
            )
