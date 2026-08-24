import logging

import numpy as np

logger = logging.getLogger(__name__)


class SimulationController:
    def __init__(
        self,
        mechanics_problem,
        ep_solver,
        ode_model,
        dt_mech: float,
        dt_ep: float,
    ):
        self.mechanics_problem = mechanics_problem
        self.ep_solver = ep_solver
        self.ode_model = ode_model

        self.dt_mech = dt_mech
        self.dt_ep = dt_ep

        self.ep_steps_per_mech = int(np.round(dt_mech / dt_ep))
        if not np.isclose(self.ep_steps_per_mech * dt_ep, dt_mech):
            raise ValueError("dt_mech must be an exact multiple of dt_ep")

        self.t = 0.0
        self.ep_step_idx = 0
        self.mech_step_idx = 0

    def step(self, ep_callback=None, mech_callback=None):
        """Advances the fully coupled system by one mechanics time step."""
        logger.info(f"--- Solving Coupled Step at t={self.t} ---")

        # 1. Step EP solver forward by dt_mech using micro-steps
        for _ in range(self.ep_steps_per_mech):
            self.ep_solver.step((self.t, self.t + self.dt_ep))
            self.t += self.dt_ep
            self.ep_step_idx += 1

            if ep_callback:
                ep_callback(self.t, self.ep_step_idx)

        # 2. Transfer state from EP to Mechanics
        self.ode_model.update_ep_missing_values(
            self.t,
            self.ep_solver.ode._values,
            self.ep_solver.ode.parameters,
        )
        self.ode_model.missing_mech.interpolate_ep_to_mechanics()
        self.ode_model.missing_mech.mechanics_function_to_values()

        # 3. Solve Mechanics Problem
        self.mechanics_problem.model.active.t.value = self.t
        nit = self.mechanics_problem.solve()
        # The activation backend owns the post-solve update: it records the new
        # stretch and advances its own state. It also advances the previous
        # values, so the separate update_prev() call this used to make was
        # redundant.
        self.mechanics_problem.model.active.post_solve()
        self.mech_step_idx += 1

        # 4. Transfer state from Mechanics back to EP
        if self.ode_model.missing_ep is not None:
            self.ode_model.missing_ep.interpolate_mechanics_to_ep()
            self.ode_model.missing_ep.ep_function_to_values()
        self.ode_model.update_prev_missing_mech()

        if mech_callback:
            mech_callback(self.t, self.mech_step_idx, nit)
