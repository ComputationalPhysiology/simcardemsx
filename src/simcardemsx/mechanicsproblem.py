from dataclasses import dataclass
import fenicsx_pulse
import ufl
import dolfinx


class MechanicsProblem(fenicsx_pulse.StaticProblem):
    def _material_form(self, u: dolfinx.fem.Function, p: dolfinx.fem.Function):
        F = ufl.grad(u) + ufl.Identity(3)
        internal_energy = self.model.strain_energy(F, p=p) * self.geometry.dx

        forms = [
            ufl.derivative(internal_energy, f, f_test)
            for f, f_test in zip(self.states, self.test_functions)
        ]

        f0 = self.model.material.f0
        f = F * f0
        lmbda = ufl.sqrt(f**2)
        Pa = self.model.active.Ta(lmbda) * ufl.outer(f, f0)
        forms[0] += ufl.inner(Pa, ufl.grad(self.u_test)) * self.geometry.dx

        return forms

    def post_solve(self):
        F = ufl.grad(self.u) + ufl.Identity(3)
        f = F * self.model.material.f0
        lmbda = ufl.sqrt(f**2)

        self.model.active.lmbda.interpolate(
            dolfinx.fem.Expression(
                lmbda, self.model.active.function_space.element.interpolation_points()
            )
        )

        if self.model.active.dt > 0:
            self.model.active._dLambda.interpolate(
                dolfinx.fem.Expression(
                    (lmbda - self.model.active.lmbda_prev) / self.model.active.dt,
                    self.model.active.function_space.element.interpolation_points(),
                )
            )

        self.model.active.Ta_current.interpolate(
            dolfinx.fem.Expression(
                self.model.active.Ta(lmbda),
                self.model.active.function_space.element.interpolation_points(),
            )
        )

        self.model.active.update_current(lmbda=lmbda)
        self.model.active.update_prev()
