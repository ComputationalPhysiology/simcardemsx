"""Orchestration of one coupled EP/mechanics step."""

import logging

import numpy as np

from .transfers import resolve

logger = logging.getLogger(__name__)


class SimulationController:
    """Advances the coupled system, driving an activation backend.

    One mechanics step is:

    1. run ``dt_mech / dt_ep`` EP micro-steps,
    2. move what the backend needs from EP into the Functions it owns,
    3. advance the backend's activation,
    4. solve the mechanics problem once,
    5. let the backend update whatever depended on the new displacement,
    6. move what the backend sends back onto the EP mesh, for the next round.

    Which variables move in steps 2 and 6 is the backend's to declare, not this
    class's to assume -- that is what lets one controller drive either split.
    """

    def __init__(
        self,
        mechanics_problem,
        ep_solver,
        ode_model,
        backend,
        dt_mech: float,
        dt_ep: float,
    ):
        # The backend is passed rather than reached for. It is available as
        # mechanics_problem.model.active, but going through two objects to find
        # the controller's most important collaborator hides it from the
        # signature -- and nothing then stops a caller advancing one backend
        # while the solve uses another.
        in_form = mechanics_problem.model.active
        if backend is not in_form:
            raise ValueError(
                "backend is not the activation backend assembled into the "
                f"mechanics form: got {type(backend).__name__}, but the form "
                f"holds {type(in_form).__name__}. Advancing one while solving "
                "with the other silently decouples the two.",
            )

        self.mechanics_problem = mechanics_problem
        self.ep_solver = ep_solver
        self.ode_model = ode_model
        self.backend = backend

        self.dt_mech = dt_mech
        self.dt_ep = dt_ep

        self.ep_steps_per_mech = int(np.round(dt_mech / dt_ep))
        if not np.isclose(self.ep_steps_per_mech * dt_ep, dt_mech):
            raise ValueError("dt_mech must be an exact multiple of dt_ep")

        self.plan = resolve(
            backend,
            ep_missing=ode_model.ep_missing,
            mech_missing=ode_model.mech_missing,
            units=ode_model.units,
            ep_sources=ode_model.ep_transfer_sources(),
            ep_targets=ode_model.ep_transfer_targets(),
        )

        self.t = 0.0
        self.ep_step_idx = 0
        self.mech_step_idx = 0

    def step(self, ep_callback=None, mech_callback=None):
        """Advances the fully coupled system by one mechanics time step."""
        logger.info(f"--- Solving Coupled Step at t={self.t} ---")

        # 1. EP micro-steps
        for _ in range(self.ep_steps_per_mech):
            self.ep_solver.step((self.t, self.t + self.dt_ep))
            self.t += self.dt_ep
            self.ep_step_idx += 1

            if ep_callback:
                ep_callback(self.t, self.ep_step_idx)

        # 2. EP -> the backend's own Functions
        self.ode_model.update_ep_missing_values(
            self.t,
            self.ep_solver.ode._values,
            self.ep_solver.ode.parameters,
        )
        self.plan.push_to_backend()

        # 3. Advance activation, using the stretch from the previous solve
        self.backend.step(self.t, dt=self.dt_mech)

        # 4. One mechanics solve
        nit = self.mechanics_problem.solve()

        # 5. The backend owns whatever depends on the new displacement
        self.backend.post_solve()
        self.mech_step_idx += 1

        # 6. Mechanics -> EP, for the next round's micro-steps
        self.plan.pull_from_backend()
        self.ode_model.missing_ep.ep_function_to_values()

        if mech_callback:
            mech_callback(self.t, self.mech_step_idx, nit)
