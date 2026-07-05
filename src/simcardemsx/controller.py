# src/simcardemsx/controller.py
import logging

import numpy as np

logger = logging.getLogger(__name__)


class SimulationController:
    def __init__(
        self,
        mechanics_problem,
        ep_solver,
        transfer_ep_to_mech,
        transfer_mech_to_ep,
        dt_mech: float,
        dt_ep: float,
    ):
        self.mechanics_problem = mechanics_problem
        self.ep_solver = ep_solver
        self.transfer_ep2mech = transfer_ep_to_mech
        self.transfer_mech2ep = transfer_mech_to_ep

        self.dt_mech = dt_mech
        self.dt_ep = dt_ep

        # Calculate how many EP steps per Mechanics step
        self.ep_steps_per_mech = int(np.round(dt_mech / dt_ep))
        if not np.isclose(self.ep_steps_per_mech * dt_ep, dt_mech):
            raise ValueError("dt_mech must be an exact multiple of dt_ep")

        self.t = 0.0

    def step(self):
        """Advances the fully coupled system by one mechanics time step (dt_mech)."""
        logger.info(f"--- Solving Coupled Step at t={self.t} ---")

        # 1. Step EP solver forward by dt_mech using micro-steps
        for _ in range(self.ep_steps_per_mech):
            self.ep_solver.step()  # Assuming beat solver has a step method
            self.t += self.dt_ep

        # 2. Transfer state from EP to Mechanics
        # e.g., transfer intracellular calcium or strong cross-bridges
        self.transfer_ep2mech.interpolate(
            self.ep_solver.state_function,  # Replace with actual EP state function
            self.mechanics_problem.model.active.XS,  # Or whatever LandModel needs
        )

        # 3. Solve Mechanics Problem
        self.mechanics_problem.model.active.t.value = self.t
        self.mechanics_problem.solve()
        self.mechanics_problem.post_solve()

        # 4. Transfer state from Mechanics back to EP (Mechano-Electric Feedback)
        # e.g., transfer stretch (lmbda) or stretch rate
        self.transfer_mech2ep.interpolate(
            self.mechanics_problem.model.active.lmbda,
            self.ep_solver.stretch_function,  # Replace with actual EP stretch parameter
        )
