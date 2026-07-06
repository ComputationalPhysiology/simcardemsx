import dolfinx
import pulse
import ufl


class MechanicsProblem(pulse.StaticProblem):
    def iteration_update(self, snes, step, rnorm):
        """
        This callback fires during every single Newton iteration.
        """
        # Extract the current displacement guess from the Newton solver
        current_vec = snes.getSolution()

        # Sync the FEniCSx displacement function with the PETSc vector
        u_local_size = self.u.x.index_map.size_local * self.u.x.block_size
        self.u.x.array[:u_local_size] = current_vec.array_r[:u_local_size]

        # 3. Update ghost nodes (crucial for parallel MPI runs)
        self.u.x.scatter_forward()

        # ---------------------------------------------------------
        # YOUR CUSTOM UPDATE LOGIC GOES HERE
        # Now that problem.u has the current iteration's displacement,
        # you can calculate new stretches, project them to Functions,
        # or update ODE states dynamically!
        # ---------------------------------------------------------

        print(f"--- Newton Iteration {step} ---")
        # Example: print the max stretch at this specific iteration
        # print("Max displacement guess:", np.max(problem.u.x.array))
        F = ufl.grad(self.u) + ufl.Identity(3)
        C = F.T * F

        f0 = self.model.material.f0
        f2 = ufl.inner(C * f0, f0)
        lmbda = ufl.sqrt(f2)

        self.model.active.update_dLambda(lmbda=lmbda)

    def __post_init__(self):
        super().__post_init__()
        # snes_solver = self.problem.solver

        # setMonitor triggers our function so we can inspect/update state
        # snes_solver.setMonitor(self.iteration_update)

    def _material_form(self, u: dolfinx.fem.Function, p: dolfinx.fem.Function):
        import logging

        logger = logging.getLogger(__name__)
        logger.debug("Creating custom simcardemsx material form with Sa...")

        # 1. Setup kinematics
        I = ufl.Identity(3)
        F = I + ufl.grad(u)
        C = ufl.variable(F.T * F)
        J = ufl.det(F)

        # Automatic differentiation for the variation of C
        var_C = ufl.derivative(C, u, self.u_test)

        forms = self._empty_form()

        # 2. Get Passive & Compressibility Contributions
        S_passive = self.model.material.S(C, dev=True)
        S_comp = self.model.compressibility.S(C)

        # 3. Compute Active Second Piola-Kirchhoff Stress (Sa)
        f0 = self.model.material.f0

        # Calculate stretch squared using C: f0^T * C * f0
        f2 = ufl.inner(C * f0, f0)
        lmbda = ufl.sqrt(f2)

        # Sa = Ta * (f0 x f0)
        Sa = self.model.active.Ta(lmbda) * ufl.outer(f0, f0)

        # 4. Total Stress and Weak Form Integration
        S_total = S_passive + S_comp + Sa
        forms[0] += ufl.inner(S_total, 0.5 * var_C) * self.geometry.dx

        # 5. Incompressibility constraint (if applicable)
        if self.is_incompressible:
            forms[-1] += (J - 1.0) * self.p_test * self.geometry.dx

        return forms

    def post_solve(self):
        # Keep the updated post_solve from the previous step
        F = ufl.grad(self.u) + ufl.Identity(3)
        C = F.T * F

        f0 = self.model.material.f0
        f2 = ufl.inner(C * f0, f0)
        lmbda = ufl.sqrt(f2)

        # self.model.active.lmbda.interpolate(
        #     dolfinx.fem.Expression(
        #         lmbda,
        #         self.model.active.function_space.element.interpolation_points,
        #     ),
        # )

        # if self.model.active.dt > 0:
        #     self.model.active._dLambda.interpolate(
        #         dolfinx.fem.Expression(
        #             (lmbda - self.model.active.lmbda_prev) / self.model.active.dt,
        #             self.model.active.function_space.element.interpolation_points,
        #         ),
        #     )
        self.model.active.Ta_current.interpolate(
            dolfinx.fem.Expression(
                self.model.active.Ta(lmbda),
                self.model.active.function_space.element.interpolation_points,
            ),
        )

        self.model.active.update(lmbda=lmbda)
        self.model.active.u_prev.x.array[:] = self.u.x.array[:]
        print(self.model.active._dLambda.x.array)
