from textwrap import dedent

import gotranx
import sympy
from gotranx.codegen.python import PythonCodeGenerator
from sympy.printing.pycode import PythonCodePrinter

from . import template

rel_op_2_ufl = {
    "<": "ufl.lt",
    "<=": "ufl.le",
    "==": "ufl.eq",
    "!=": "ufl.ne",
    ">": "ufl.gt",
    ">=": "ufl.ge",
}

LAND_MODEL = dedent(
    """
import ufl
import dolfinx
import pulse


class LandModel(pulse.active_model.ActiveModel):
    def __init__(
        self,
        function_space: dolfinx.fem.FunctionSpace,
        missing_values: list[dolfinx.fem.Function],
        f0: dolfinx.fem.Function | dolfinx.fem.Constant,
        Ta_key: str = "Ta",
        lmbda_key: str = "lmbda",
        dLambda_key: str = "dLambda",
        s0: dolfinx.fem.Function | dolfinx.fem.Constant | None = None,
        n0: dolfinx.fem.Function | dolfinx.fem.Constant | None = None,
        eta: dolfinx.fem.Constant = dolfinx.default_scalar_type(0.0),
        isotropy: pulse.active_stress.ActiveStressModels = pulse.active_stress.ActiveStressModels.transversely,
        ode_states: dict[str, float] | None = None,
        ode_parameters: dict[str, float] | None = None,
    ):
        self.function_space = function_space
        self.missing_values = missing_values
        self.f0 = f0
        self.Ta_key = Ta_key
        self.lmbda_key = lmbda_key
        self.dLambda_key = dLambda_key
        self.s0 = s0
        self.n0 = n0
        self.isotropy = isotropy
        self.eta = dolfinx.fem.Constant(ufl.domain.extract_unique_domain(self.f0), eta)

        self._dLambda = dolfinx.fem.Function(self.function_space, name="dLambda")
        self.lmbda_prev = dolfinx.fem.Function(self.function_space, name="lmbda_prev")
        self.lmbda_prev.x.array[:] = 1.0
        self.lmbda = dolfinx.fem.Function(self.function_space, name="lmbda")

        self.Ta_current = dolfinx.fem.Function(self.function_space, name="Ta")

        self.t = dolfinx.fem.Constant(self.mesh, 0.0)
        self._t_prev = dolfinx.fem.Constant(self.mesh, 0.0)

        if ode_states is None:
            ode_states = {}
        y_ = init_state_values(**ode_states)
        if ode_parameters is None:
            ode_parameters = {}
        p = init_parameter_values(**ode_parameters)

        self._zetas_index = state_index("Zetas")
        self._zetaw_index = state_index("Zetaw")

        num_points = self._dLambda.x.array.size
        y = numpy.zeros((len(y_), num_points))
        y.T[:] = y_

        self.p = []
        for pname, pi in zip(parameter, p):
            const = dolfinx.fem.Constant(self.mesh, pi)
            const.name = pname
            self.p.append(const)

        self.y = [dolfinx.fem.Function(self.function_space, name=name) for name in state]
        self.y_prev = [dolfinx.fem.Function(self.function_space, name=f"{name}_prev") for name in state]

        for i, yi in enumerate(self.y):
            yi.x.array[:] = y[i, :]

    @property
    def dt(self) -> float:
        return float(self.t - self._t_prev)

    def dLambda(self, lmbda):
        if self.dt < 1e-12:
            return self._dLambda
        else:
            return (lmbda - self.lmbda_prev) / self.dt

    @property
    def mesh(self):
        return self.function_space.mesh

    def Fe(self, F: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        return F

    def update(self, lmbda):
        self.p[parameter_index("lmbda")] = lmbda
        self.p[parameter_index("dLambda")] = self.dLambda(lmbda)

        y_new = generalized_rush_larsen(
            self.y,
            self.t.value,
            self.dt,
            self.p,
            self.missing_values,
        )
        for y_current, new_value in zip(self.y, y_new):
            y_current.interpolate(
                dolfinx.fem.Expression(
                    new_value, self.function_space.element.interpolation_points()
                )
            )

        self.update_prev()

    def update_prev(self):
        # logger.debug("update previous")
        for next, prev in zip(self.y, self.y_prev):
            prev.x.array[:] = next.x.array

        self.lmbda_prev.x.array[:] = self.lmbda.x.array

        self._t_prev.value = self.t.value.copy()

    def Ta(self, lmbda):
        self.p[parameter_index(self.lmbda_key)] = lmbda
        self.p[parameter_index(self.dLambda_key)] = self.dLambda(lmbda)

        Ta_index = monitor_index(self.Ta_key)
        mv = monitor_values(
            self.t.value,
            self.y,
            self.p,
            self.missing_values,
        )
        kPa2Pa = 1000.0
        return mv[Ta_index] * kPa2Pa

    def strain_energy(self, C: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        \"""Active strain energy density

        Parameters
        ----------
        C : ufl.core.expr.Expr
            The deformation gradient

        Returns
        -------
        ufl.core.expr.Expr
            The active strain energy density

        Raises
        ------
        NotImplementedError
            If the active stress model is not implemented
        \"""
        lmbda = ufl.sqrt(ufl.inner(C * self.f0, self.f0))
        if self.isotropy == pulse.active_stress.ActiveStressModels.transversely:
            return pulse.active_stress.transversely_active_stress_strain_energy(
                Ta=self.Ta(lmbda),
                C=C,
                f0=self.f0,
                eta=self.eta,
            )
        else:
            raise NotImplementedError

    def S(self, C: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        \"""Cauchy stress tensor for the active stress model.

        Parameters
        ----------
        C : ufl.core.expr.Expr
            The right Cauchy-Green deformation tensor

        Returns
        -------
        ufl.core.expr.Expr
            The Cauchy stress tensor
        \"""
        lmbda = ufl.sqrt(ufl.inner(C * self.f0, self.f0))
        if self.isotropy == pulse.active_stress.ActiveStressModels.transversely:
            return pulse.active_stress.transversely_active_stress(Ta=self.Ta(lmbda), f0=self.f0, eta=self.eta)
        else:
            raise NotImplementedError

""",
)


class SimcardemsPrinter(PythonCodePrinter):
    _kf = {
        **{k: f"ufl.{v.replace('math.', '')}" for k, v in PythonCodePrinter._kf.items()},
    }
    _kc = {k: f"ufl.{v.replace('math.', '')}" for k, v in PythonCodePrinter._kc.items()}

    def _print_Relational(self, expr):
        return f"{rel_op_2_ufl[expr.rel_op]}({self._print(expr.lhs)}, {self._print(expr.rhs)})"

    def _print_Piecewise(self, expr):
        result = []

        if isinstance(expr.args[0][0], sympy.codegen.ast.Assignment):
            lhs = super()._print(expr.args[0][0].lhs)
            result.append(f"{super()._print(lhs)} = ")
            all_lsh_equal = True
            for arg in expr.args:
                result.append("ufl.conditional(")
                result.append(f"{super()._print(arg[1])}")
                result.append(", ")
                result.append(f"{super()._print(arg[0].rhs)}")
                result.append(", ")
                all_lsh_equal = all_lsh_equal and super()._print(arg[0].lhs) == lhs

            assert all_lsh_equal, "All assignments in Piecewise must have the same lhs"

            if super()._print(arg[1]) == "True":
                result = result[:-6]
                result.append(f", {super()._print(arg[0].rhs)}")
            else:
                raise ValueError("Last condition in Piecewise must be True")

            result.append(")" * (len(expr.args) - 1))

        else:
            from gotranx.codegen.base import _print_Piecewise

            conds, exprs = _print_Piecewise(self, expr)

            for c, e in zip(conds, exprs):
                result.append("ufl.conditional(")
                result.append(f"{c}")
                result.append(", ")
                result.append(f"{e}")
                result.append(", ")

            if c == "True":
                result = result[:-6]
                result.append(f", {e}")
            else:
                raise ValueError("Last condition in Piecewise must be True")

            result.append(")" * (len(conds) - 1))

        return "".join(result)

    def _print_And(self, expr):
        if len(expr.args) == 2:
            value = f"ufl.And({self._print(expr.args[0])}, {self._print(expr.args[1])})"
        else:
            raise NotImplementedError

        return value

    def _print_Or(self, expr):
        # value = super()._print_Or(expr)
        if len(expr.args) == 2:
            value = f"ufl.Or({self._print(expr.args[0])}, {self._print(expr.args[1])})"
        else:
            raise NotImplementedError

        return value

    def _print_Equality(self, expr):
        lhs, rhs = expr.args
        return f"ufl.eq({self._print(lhs)}, {self._print(rhs)})"

    def _print_Assignment(self, expr):
        sym, value = expr.lhs, expr.rhs
        if isinstance(sym, sympy.tensor.indexed.Indexed):
            if sym.base.name == "values":
                index = self._print(sym.indices[0])
                return f"_{sym.base.name}_{index} = {self._print(value)}"

        return super()._print_Assignment(expr)


class SimcardemsCodeGenerator(PythonCodeGenerator):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._printer = SimcardemsPrinter()

    def imports(self) -> str:
        return (
            "\n".join(
                [
                    "import dolfinx",
                    "import ufl",
                    "import numpy",
                ],
            )
            + "\n"
            + LAND_MODEL
        )

    @property
    def template(self):
        return template


def ode2mechanics(ode, missing_values=None):
    codegen = SimcardemsCodeGenerator(ode)

    comp = [
        codegen.imports(),
        codegen.parameter_index(),
        codegen.state_index(),
        codegen.monitor_index(),
        codegen.missing_index(),
        codegen.initial_parameter_values(),
        codegen.initial_state_values(),
        codegen.rhs(),
        codegen.monitor_values(),
        codegen.scheme(gotranx.schemes.get_scheme("generalized_rush_larsen")),
        codegen.missing_values(missing_values),
    ]

    return codegen._format("\n".join(comp))
