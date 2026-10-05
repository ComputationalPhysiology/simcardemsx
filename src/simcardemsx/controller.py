"""The coupled time step: EP micro-steps, then one mechanics solve, values crossing both ways.

What crosses is derived from the EP module and the backend's ``missing``/``provides``
(:func:`~simcardemsx.transfer_plan.resolve`), and moved by a
:class:`~simcardemsx.transfer_plan.TransferPlan`. The backend is a
:class:`~simcardemsx.backends.base.CoupledBackend`:
:class:`~simcardemsx.backends.GeneratedActivation` or
:class:`~simcardemsx.backends.CrossbridgeSegregated`. The deprecated
:class:`~simcardemsx.backends.ZetaSplitUFL` remains usable directly against ``pulse``.

A step that fails is rolled back: the controller takes a
:class:`~simcardemsx.checkpoint.Snapshot` of all its components at the start of each
step, and restores it if the step does not complete.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Callable, cast

import beat
import pulse

from .checkpoint import (
    Checkpointable,
    CycleState,
    EPState,
    MechanicsState,
    Snapshot,
    restore_snapshot,
    take_snapshot,
)
from .mechanics import Cycle, as_driver
from .transfer_plan import TransferPlan, resolve

if TYPE_CHECKING:
    import dolfinx

    from .backends.base import CoupledBackend
    from .mechanics import MechanicsDriver
    from .ode_model import ODEModules

logger = logging.getLogger(__name__)

#: Relative tolerance on ``dt_mech == n * dt_ep``: round-off, not a real remainder.
_DT_RTOL = 1e-9


class SimulationController:
    """Advance EP and mechanics together, one mechanics time step at a time.

    The controller is itself a :class:`~simcardemsx.checkpoint.Checkpointable`
    (namespace ``"simcardemsx"``): its state is its clock and step counters.

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

    Attributes
    ----------
    t:
        The time reached, in ms: the end of the last step that completed. A step that
        fails leaves it at that step's start.
    t_failed:
        The end, in ms, of the step that :meth:`step` last rolled back: the ``t`` it
        would have reached, ``t_n + ep_steps_per_mech * dt_ep``. ``None`` if the last
        step did not fail, or failed and its restore raised too. Every step resets it
        to ``None`` first.

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

        #: The EP states' names, in the order of the ODE's state array.
        self._state_names = sorted(ode_modules.ep.state, key=ode_modules.ep.state.__getitem__)

        self.t = 0.0
        self.ep_step_idx = 0
        self.mech_step_idx = 0
        self.t_failed: float | None = None

    # ------------------------------------------------------------------
    # Components, snapshots and restore
    # ------------------------------------------------------------------

    namespace = "simcardemsx"

    def components(self) -> list[Checkpointable]:
        """Everything whose state the coupled run carries from step to step, in the
        order it is restored: EP, the transfer plan (the rows of EP's arrays that come
        from the backend), the mechanics problem, the cycle controller (only when the
        driver is a :class:`~simcardemsx.mechanics.Cycle`), the backend, and the
        controller itself."""
        components: list[Checkpointable] = [
            EPState(self.ep_solver, self._state_names),
            self.plan,
            MechanicsState(self.mechanics.problem),
        ]
        if isinstance(self.mechanics, Cycle):
            components.append(CycleState(self.mechanics.controller))
        # Both CoupledBackends are Checkpointable; the protocol does not say so.
        components += [cast(Checkpointable, self.backend), self]
        return components

    def restart_functions(self) -> list[tuple[str, dolfinx.fem.Function]]:
        """None: the controller's state is all metadata."""
        return []

    def restart_metadata(self) -> dict[str, Any]:
        """The clock and the step counters, and the time steps they count (ms)."""
        return {
            "t_ms": self.t,
            "ep_step_idx": self.ep_step_idx,
            "mech_step_idx": self.mech_step_idx,
            "dt_mech_ms": self.dt_mech,
            "dt_ep_ms": self.dt_ep,
        }

    def load_restart(
        self,
        functions: Sequence[tuple[str, dolfinx.fem.Function]],
        metadata: Mapping[str, Any],
    ) -> None:
        """Set the clock and the step counters.

        Raises ``ValueError`` if the metadata's ``dt_mech`` or ``dt_ep`` differ from
        this controller's, since the counters count steps of those sizes.
        """
        for key, own in (("dt_mech_ms", self.dt_mech), ("dt_ep_ms", self.dt_ep)):
            if metadata[key] != own:
                raise ValueError(
                    f"The checkpoint was written with {key.removesuffix('_ms')} = "
                    f"{metadata[key]} ms, this controller has {own} ms",
                )
        self.t = float(metadata["t_ms"])
        self.ep_step_idx = int(metadata["ep_step_idx"])
        self.mech_step_idx = int(metadata["mech_step_idx"])

    def snapshot(self) -> Snapshot:
        """Copies of the state of every one of :meth:`components`."""
        return take_snapshot(self.components())

    def restore(self, snapshot: Snapshot) -> None:
        """Put a :meth:`snapshot` back into every one of :meth:`components`.

        Nothing is moved back to EP (``plan.backward()`` is not called): the rows of
        EP's ``missing_variables`` and ``parameters`` that come from the backend are
        the transfer plan's own state, restored with the rest. Moving the backend's
        outputs back instead would be wrong before the first accepted step, when they
        are zero and EP holds its initial values.
        """
        restore_snapshot(self.components(), snapshot)

    def step(
        self,
        ep_callback: Callable[[float, int], None] | None = None,
        mech_callback: Callable[[float, int, int], None] | None = None,
    ) -> None:
        """Advance the coupled system by one mechanics time step, from ``t`` to ``t + dt_mech``.

        In order: a :meth:`snapshot`; the EP micro-steps (``ep_callback(t,
        ep_step_idx)`` after each); EP values forward into the backend's inputs, at the
        new ``t``; ``backend.begin_step(t_n, dt_mech)``, which prepares the step from
        the old ``t``; ``self.mechanics.advance(t_n, dt_mech)``;
        ``backend.post_solve()``, which accepts the step; the backend's outputs back
        into EP's arrays; the step counter; ``mech_callback(t, mech_step_idx,
        newton_iterations)``.

        If anything before ``mech_callback`` raises (any ``BaseException``,
        ``KeyboardInterrupt`` included), or ``advance`` returns ``False``, the step is
        rolled back: the snapshot is restored (:meth:`restore`), so every component,
        EP's arrays included, is as it was before the step and ``t`` is ``t_n``;
        :attr:`t_failed` is set to the ``t`` the step would have reached, ``t_n +
        ep_steps_per_mech * dt_ep`` (the EP loop's arithmetic, which ``t_n + dt_mech``
        can differ from in the last bit); and the error is raised. That is
        ``RuntimeError`` if ``advance`` returned ``False``, and otherwise the error
        itself, unchanged. If the restore raises in turn, its error is raised with the
        step's as its cause. An error from ``mech_callback`` is not rolled back: the
        step is complete by then.
        """
        self.t_failed = None
        t_n = self.t
        # As the last EP micro-step below computes it, bit for bit.
        t_end = t_n + self.ep_steps_per_mech * self.dt_ep
        logger.info(f"--- Solving coupled step from t = {t_n} ---")
        snapshot = self.snapshot()

        try:
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
                    "The mechanics solve did not converge for the step from "
                    f"t = {t_n} to t = {t_end}",
                )
            self.backend.post_solve()
            self.plan.backward()
            self.mech_step_idx += 1
        except BaseException as error:
            try:
                self.restore(snapshot)
            except BaseException as restore_error:
                raise restore_error from error
            self.t_failed = t_end
            logger.info(f"Rolled the coupled step back to t = {self.t}")
            raise

        if mech_callback:
            mech_callback(
                self.t,
                self.mech_step_idx,
                self.mechanics.problem.problem.solver.getIterationNumber(),
            )
