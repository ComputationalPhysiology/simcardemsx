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

        # return self._create_residual_form(internal_energy)
        return forms
