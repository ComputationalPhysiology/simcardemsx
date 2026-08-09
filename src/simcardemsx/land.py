import logging
from enum import Enum

import dolfinx
import numpy as np
import pulse
import ufl

logger = logging.getLogger(__name__)


class Scheme(str, Enum):
    fd = "fd"
    bd = "bd"
    analytic = "analytic"


def _Zeta(Zeta_prev, A, c, dLambda, dt, scheme):
    dZetas_dt = A * dLambda - Zeta_prev * c
    dZetas_dt_linearized = -c
    if abs(c) > 1e-8:
        ans = Zeta_prev + dZetas_dt * (np.exp(-c * dt) - 1.0) / dZetas_dt_linearized
    else:
        ans = Zeta_prev + dZetas_dt * dt

    if hasattr(ans, "ufl_shape"):
        return ufl.max_value(ans, -1.0)
    return max(ans, -1.0)


_parameters = {
    "Beta0": 2.3,
    "Tot_A": 25.0,
    "Tref": 120,
    "kuw": 0.182,
    "kws": 0.012,
    "phi": 2.23,
    "rs": 0.25,
    "rw": 0.5,
}


class LandModel(pulse.active_model.ActiveModel):
    def __init__(
        self,
        f0,
        s0,
        n0,
        XS,
        XW,
        mesh,
        parameters=None,
        Zetas=None,
        Zetaw=None,
        lmbda=None,
        eta=0.0,
        scheme: Scheme = Scheme.analytic,
        dLambda_tol: float = 1e-12,
        **kwargs,
    ):
        logger.debug("Initialize Land Model")

        self._eta = eta
        self.function_space = dolfinx.fem.functionspace(mesh, ("DG", 1))
        self.u_space = dolfinx.fem.functionspace(mesh, ("P", 2, (3,)))
        self.u = dolfinx.fem.Function(self.u_space)
        self.u_prev = dolfinx.fem.Function(self.u_space)

        self.XS = XS
        self.XW = XW
        if parameters is None:
            parameters = _parameters
        self._parameters = parameters

        self._scheme = scheme

        self._dLambda = dolfinx.fem.Function(self.function_space)
        self.lmbda_prev = dolfinx.fem.Function(self.function_space)
        self.lmbda_prev.x.array[:] = 1.0
        self.lmbda = dolfinx.fem.Function(self.function_space)

        self._Zetas = dolfinx.fem.Function(self.function_space)
        self.Zetas_prev = dolfinx.fem.Function(self.function_space)
        self._Zetaw = dolfinx.fem.Function(self.function_space)
        self.Zetaw_prev = dolfinx.fem.Function(self.function_space)
        self.Ta_current = dolfinx.fem.Function(self.function_space, name="Ta")

        self._dLambda_tol = dLambda_tol
        self.t = dolfinx.fem.Constant(mesh, 0.0)
        self._t_prev = dolfinx.fem.Constant(mesh, 0.0)

    def dLambda(self, lmbda):
        logger.debug("Evaluate dLambda")

        if self.dt == 0:
            return self._dLambda
        else:
            return (lmbda - self.lmbda_prev) / self.dt

    @property
    def Aw(self):
        Tot_A = self._parameters["Tot_A"]
        rs = self._parameters["rs"]
        rw = self._parameters["rw"]
        scale_popu_rw = 1.0  # self._parameters["scale_popu_rw"]
        scale_popu_rs = 1.0  # self._parameters["scale_popu_rs"]
        return (
            Tot_A
            * rs
            * scale_popu_rs
            / (rs * scale_popu_rs + rw * scale_popu_rw * (1.0 - (rs * scale_popu_rs)))
        )

    @property
    def As(self):
        return self.Aw

    @property
    def cw(self):
        phi = self._parameters["phi"]
        kuw = self._parameters["kuw"]
        rw = self._parameters["rw"]

        scale_popu_kuw = 1.0  # self._parameters["scale_popu_kuw"]
        scale_popu_rw = 1.0  # self._parameters["scale_popu_rw"]
        return kuw * scale_popu_kuw * phi * (1.0 - (rw * scale_popu_rw)) / (rw * scale_popu_rw)

    @property
    def cs(self):
        phi = self._parameters["phi"]
        kws = self._parameters["kws"]
        rs = self._parameters["rs"]
        rw = self._parameters["rw"]
        scale_popu_kws = 1.0  # self._parameters["scale_popu_kws"]
        scale_popu_rw = 1.0  # self._parameters["scale_popu_rw"]
        scale_popu_rs = 1.0  # self._parameters["scale_popu_rs"]
        return (
            kws
            * scale_popu_kws
            * phi
            * rw
            * scale_popu_rw
            * (1.0 - (rs * scale_popu_rs))
            / (rs * scale_popu_rs)
        )

    def update_Zetas(self, lmbda):
        logger.debug("update Zetas")
        zetas_expr = dolfinx.fem.Expression(
            _Zeta(
                self.Zetas_prev,
                self.As,
                self.cs,
                self.dLambda(lmbda),
                self.dt,
                self._scheme,
            ),
            self.function_space.element.interpolation_points,
        )
        self._Zetas.interpolate(zetas_expr)

    def Zetas(self, lmbda):
        # return self._Zetas
        return _Zeta(
            self.Zetas_prev,
            self.As,
            self.cs,
            self.dLambda(lmbda),
            self.dt,
            self._scheme,
        )

    def update_Zetaw(self, lmbda):
        logger.debug("update Zetaw")
        zetaw_expr = dolfinx.fem.Expression(
            _Zeta(
                self.Zetaw_prev,
                self.Aw,
                self.cw,
                self.dLambda(lmbda),
                self.dt,
                self._scheme,
            ),
            self.function_space.element.interpolation_points,
        )
        self._Zetaw.interpolate(zetaw_expr)

    def Zetaw(self, lmbda):
        return _Zeta(
            self.Zetaw_prev,
            self.Aw,
            self.cw,
            self.dLambda(lmbda),
            self.dt,
            self._scheme,
        )

    @property
    def dt(self) -> float:
        return float(self.t - self._t_prev)

    def update(self, lmbda=None):
        self.update_current(lmbda=lmbda)
        self.update_prev()

    def update_current(self, lmbda):
        self.update_Zetas(lmbda=lmbda)
        self.update_Zetaw(lmbda=lmbda)

    def update_prev(self):
        logger.debug("update previous")
        self.Zetas_prev.x.array[:] = self._Zetas.x.array
        self.Zetaw_prev.x.array[:] = self._Zetaw.x.array
        self.lmbda_prev.x.array[:] = self.lmbda.x.array

        self._t_prev.value = self.t.value.copy()

    def Ta(self, lmbda):
        logger.debug("Evaluate Ta (Exact Implicit)")

        Tref = self._parameters["Tref"]
        rs = self._parameters["rs"]
        scale_popu_Tref = 1.0
        scale_popu_rs = 1.0
        Beta0 = self._parameters["Beta0"]

        _min = ufl.min_value
        _max = ufl.max_value
        if isinstance(lmbda, (int, float)):
            _min = min
            _max = max

        # 1. Exact length dependence
        lmbda_capped = _min(1.2, lmbda)
        h_lambda_prima = 1.0 + Beta0 * (lmbda_capped + _min(lmbda_capped, 0.87) - 1.87)
        h_lambda = _max(0.0, h_lambda_prima)

        # 2. Evaluate dynamic active state
        Zetas = self.Zetas(lmbda)
        Zetaw = self.Zetaw(lmbda)
        # 3. Exactly bound active fraction to >= 0.0 to prevent negative tension
        active_fraction = self.XS * (Zetas + 1.0) + self.XW * Zetaw
        active_fraction = ufl.max_value(0.0, active_fraction)

        return 1000 * h_lambda * (Tref * scale_popu_Tref / (rs * scale_popu_rs)) * active_fraction

    def strain_energy(self, F: ufl.core.expr.Expr):
        return 0.0

    def Fe(self, F: ufl.core.expr.Expr):
        return F
