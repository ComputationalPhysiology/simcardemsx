"""The coupled time step: EP micro-steps, then one mechanics solve, values crossing both ways.

What crosses is derived from the two generated modules (:func:`~simcardemsx.transfer_plan.
resolve`), and moved by a :class:`~simcardemsx.transfer_plan.TransferPlan`. Only
:class:`~simcardemsx.backends.GeneratedActivation` is supported here; the older backends
remain usable directly against ``pulse``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

from .transfer_plan import TransferPlan, resolve

if TYPE_CHECKING:
    import beat
    import pulse

    from .backends import GeneratedActivation
    from .ode_model import ODEModules

logger = logging.getLogger(__name__)

#: Relative tolerance on ``dt_mech == n * dt_ep``: round-off, not a real remainder.
_DT_RTOL = 1e-9


class SimulationController:
    """Advance EP and mechanics together, one mechanics time step at a time.

    Parameters
    ----------
    mechanics_problem:
        The mechanics problem. Its solver options are the caller's: the controller
        only calls ``solve()``.
    ep_solver:
        beat's splitting solver; ``ep_solver.ode`` is the ODE solver whose arrays the
        plan reads and writes.
    backend:
        The activation backend. Must be ``mechanics_problem.model.active``.
    ode_modules:
        The two modules generated from the ``.ode`` file ``ep_solver`` and ``backend``
        were built from.
    dt_mech, dt_ep:
        Mechanics and EP time steps, in ms. ``dt_mech`` must be a whole multiple of
        ``dt_ep``.

    Raises
    ------
    ValueError
        If ``backend`` is not the problem's active model, if ``dt_mech`` is not a
        multiple of ``dt_ep``, or if the backend stores its states on a quadrature
        space whose degree differs from the mechanics form's.
    NotImplementedError
        From :class:`~simcardemsx.transfer_plan.TransferPlan`, if the EP ODE space is
        neither P1 nor DG0.
    """

    def __init__(
        self,
        mechanics_problem: pulse.StaticProblem,
        ep_solver: beat.MonodomainSplittingSolver,
        backend: GeneratedActivation,
        ode_modules: ODEModules,
        dt_mech: float,
        dt_ep: float,
    ):
        if backend is not mechanics_problem.model.active:
            raise ValueError(
                "backend must be the active model of the mechanics problem "
                "(mechanics_problem.model.active): the controller steps the backend and "
                "the problem solves with its own, so they must be the same object.",
            )

        ep_steps_per_mech = round(dt_mech / dt_ep)
        if ep_steps_per_mech < 1 or abs(ep_steps_per_mech * dt_ep - dt_mech) > _DT_RTOL * dt_mech:
            raise ValueError(
                f"dt_mech ({dt_mech}) must be a whole multiple of dt_ep ({dt_ep})",
            )

        if backend.space.ufl_element().family_name == "quadrature":
            geometry_degree = mechanics_problem.geometry.metadata.get("quadrature_degree")
            if backend.quadrature_degree != geometry_degree:
                raise ValueError(
                    f"The backend stores its states at quadrature degree "
                    f"{backend.quadrature_degree}, but the mechanics form integrates at "
                    f"geometry.metadata['quadrature_degree'] = {geometry_degree}. They must "
                    "be equal: FFCx evaluates the whole integrand at a quadrature-element "
                    "coefficient's own degree and ignores the measure's quadrature_degree, so "
                    "a mismatch would silently change the quadrature of the whole momentum "
                    "integral, not just which states are looked up.",
                )

        self.mechanics_problem = mechanics_problem
        self.ep_solver = ep_solver
        self.backend = backend
        self.ode_modules = ode_modules
        self.dt_mech = dt_mech
        self.dt_ep = dt_ep
        self.ep_steps_per_mech = ep_steps_per_mech

        self.plan = TransferPlan(
            resolve(ode_modules.ep, ode_modules.mechanics),
            ode_modules.ep,
            ep_solver.ode,
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
        values forward into the backend's inputs, at the new ``t``; the mechanics solve,
        with the backend stepping from the old ``t`` by ``dt_mech``; ``post_solve()``;
        the backend's outputs back into EP's arrays; ``mech_callback(t, mech_step_idx,
        newton_iterations)``.

        Raises ``RuntimeError`` if the mechanics solve does not converge.
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

        self.backend.t.value = t_n
        self.backend.dt.value = self.dt_mech
        ok = self.mechanics_problem.solve()
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
                self.mechanics_problem.problem.solver.getIterationNumber(),
            )
