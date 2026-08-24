"""The zeta split: Land active contraction, coupled monolithically in Newton.

This is the scheme validated in Myklebust et al., *Impact of Segregation Scheme
on Performance of a Strongly Coupled Cardiac Electromechanical Solver*, where it
came out the most accurate of the three splits compared. The thin-filament
populations ``XS``/``XW`` are solved on the EP side and transferred in; the
distortion states ``zeta_s``/``zeta_w`` -- the ones that depend on the stretch
*rate* -- stay here and are rebuilt symbolically in UFL every time the form is
assembled.

That last point is the whole reason this backend exists alongside the
crossbridge ones. Because ``Ta`` is a UFL expression of the current
displacement, Newton re-linearizes it at every iteration and the stretch is
treated implicitly: force generation and mechanics are effectively coupled
*monolithically*, which is what makes the scheme stable. The paper says as much
-- the velocity-dependent ODEs "were always included in the mechanics problem in
order to avoid solver instabilities". A NumPy contraction model cannot do this,
which is why the crossbridge backends need the stabilization of Regazzoni &
Quarteroni instead.
"""

from __future__ import annotations

import logging
from enum import Enum

import dolfinx
import numpy as np
import pulse
import ufl

from .base import Transfer

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


class ZetaSplitUFL(pulse.active_model.ActiveModel):
    """Land active contraction with the zeta states solved symbolically in UFL.

    Parameters
    ----------
    f0, s0, n0:
        Fibre, sheet and sheet-normal directions.
    mesh:
        The mechanics mesh.
    element:
        The function space the activation state lives on, as a
        ``dolfinx.fem.functionspace`` specification. The default matches what
        this package used before the space became configurable, and what the
        simcardems paper used. A quadrature element matching the form's
        ``quadrature_degree`` removes the interpolation error between the
        activation state and what the assembler integrates, at greater cost.

        Changing it changes results, so it is not something to vary while
        establishing that a coupling reproduces a reference.

    Notes
    -----
    The thin-filament populations ``XS`` and ``XW`` are owned by this backend
    and exposed through :attr:`ep_inputs`, which is where the coupler writes
    the values it transfers from the EP subsystem.
    formulation:
        Which active-stress convention to use. The default ``stretch`` is the
        Regazzoni & Quarteroni normalization,

            P_a = Ta * F f0 (x) f0 / |F f0|,

        under which ``Ta`` is the tension per unit *deformed* fibre
        cross-section: ``|P_a f0| = Ta`` exactly, whatever the stretch. That is
        what ``Ta`` means in the Land model's own calibration, and it is the
        convention the active stiffness of the crossbridge backends is defined
        against, so all backends agree on what the number means.

        ``invariant`` is the historical simcardems form, ``P_a = Ta * F f0 (x)
        f0``, which is larger by a factor of the fibre stretch and therefore
        makes the delivered tension depend on how far the fibre has shortened.
        It is retained only to reproduce results generated before this was
        changed; new work should not use it.

        **The two differ by a factor of lambda**, so this is a modelling
        choice, not a refactor.
    """

    def __init__(
        self,
        f0,
        s0,
        n0,
        mesh,
        element=("DG", 1),
        parameters=None,
        eta=0.0,
        scheme: Scheme = Scheme.analytic,
        dLambda_tol: float = 1e-12,
        formulation: pulse.ActiveStressFormulation = pulse.ActiveStressFormulation.stretch,
    ):
        logger.debug("Initialize ZetaSplitUFL")

        self.f0 = f0
        self.s0 = s0
        self.n0 = n0
        self._eta = eta
        self._scheme = scheme
        self._dLambda_tol = dLambda_tol
        self.formulation = formulation

        self.function_space = dolfinx.fem.functionspace(mesh, element)
        self.u_space = dolfinx.fem.functionspace(mesh, ("P", 2, (3,)))
        self.u = dolfinx.fem.Function(self.u_space)
        self.u_prev = dolfinx.fem.Function(self.u_space)

        # Owned here, not injected. The coupler interpolates into these; a
        # caller wiring in its own Functions is how positional assumptions
        # about the transfer buffers used to leak into calling code.
        self.XS = dolfinx.fem.Function(self.function_space, name="XS")
        self.XW = dolfinx.fem.Function(self.function_space, name="XW")

        self._parameters = parameters if parameters is not None else _parameters

        self._dLambda = dolfinx.fem.Function(self.function_space)
        self.lmbda_prev = dolfinx.fem.Function(self.function_space)
        self.lmbda_prev.x.array[:] = 1.0
        self.lmbda = dolfinx.fem.Function(self.function_space)

        self._Zetas = dolfinx.fem.Function(self.function_space)
        self.Zetas_prev = dolfinx.fem.Function(self.function_space)
        self._Zetaw = dolfinx.fem.Function(self.function_space)
        self.Zetaw_prev = dolfinx.fem.Function(self.function_space)
        self.Ta_current = dolfinx.fem.Function(self.function_space, name="Ta")

        self.t = dolfinx.fem.Constant(mesh, 0.0)
        self._t_prev = dolfinx.fem.Constant(mesh, 0.0)

    # ------------------------------------------------------------------
    # Coupling interface
    # ------------------------------------------------------------------

    def wants_from_ep(self) -> tuple[Transfer, ...]:
        """``XS`` and ``XW``, matching ``mechanics.missing`` of a zeta-split ODE file."""
        return (Transfer(name="XS"), Transfer(name="XW"))

    def gives_to_ep(self) -> tuple[Transfer, ...]:
        """``Zetas`` and ``Zetaw``, matching ``ep.missing`` of a zeta-split ODE file.

        Unlike the calcium splits, nothing here is a buffering flux -- the EP
        side keeps its own troponin -- so this is purely the distortion
        feedback.
        """
        return (Transfer(name="Zetas"), Transfer(name="Zetaw"))

    @property
    def ep_inputs(self) -> dict[str, dolfinx.fem.Function]:
        return {"XS": self.XS, "XW": self.XW}

    @property
    def ep_outputs(self) -> dict[str, dolfinx.fem.Function]:
        return {"Zetas": self._Zetas, "Zetaw": self._Zetaw}

    def register(self, u: dolfinx.fem.Function) -> None:
        """Receive the displacement from ``pulse.StaticProblem``."""
        self.u = u

    def step(self, t: float, dt: float | None = None) -> None:
        """Advance to time ``t``.

        The zeta states are not integrated here: they are rebuilt from
        ``t - t_prev`` inside the form at every Newton iteration, which is what
        makes this backend monolithic. All this does is move the clock.
        """
        self.t.value = t

    def post_solve(self) -> None:
        """Record the stretch and advance the zeta states, after the solve.

        Previously the body of ``MechanicsProblem.post_solve``. It lives here
        now, so that ``pulse.StaticProblem`` can be used unmodified.
        """
        F = ufl.grad(self.u) + ufl.Identity(3)
        C = F.T * F
        lmbda = ufl.sqrt(ufl.inner(C * self.f0, self.f0))
        points = self.function_space.element.interpolation_points

        self.lmbda.interpolate(dolfinx.fem.Expression(lmbda, points))

        if self.dt > 0:
            self._dLambda.interpolate(
                dolfinx.fem.Expression((lmbda - self.lmbda_prev) / self.dt, points),
            )
        self.Ta_current.interpolate(dolfinx.fem.Expression(self.Ta(lmbda), points))

        self.update(lmbda=lmbda)

    @property
    def active_tension(self) -> dolfinx.fem.Function:
        """Active tension of the last completed step, as a Function.

        Deliberately *not* called ``Ta``: on this class ``Ta(lmbda)`` is a
        method returning a UFL expression, and has been since before the
        backend refactor. Overloading the same name with a Function would make
        ``model.Ta(lmbda)`` silently evaluate a Function at a point instead of
        raising -- which is a hang, not an error.
        """
        return self.Ta_current

    # ------------------------------------------------------------------
    # pulse.ActiveModel
    # ------------------------------------------------------------------

    def Fe(self, F: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        return F

    def strain_energy(self, C: ufl.core.expr.Expr):
        """Not available for this backend.

        ``Ta`` depends on the stretch through ``zeta_s``/``zeta_w``, which are
        themselves functions of the stretch *rate*, so there is no closed-form
        potential to differentiate. ``S`` is supplied directly instead, which
        is all ``pulse.StaticProblem`` needs.
        """
        raise NotImplementedError(
            "ZetaSplitUFL has no active strain-energy potential; it supplies S directly. "
            "pulse.StaticProblem assembles model.S(C), so this is not needed.",
        )

    def S(self, C: ufl.core.expr.Expr, dev: bool = False) -> ufl.core.expr.Expr:
        """Active second Piola-Kirchhoff stress.

        ``dev`` is accepted for protocol compatibility and ignored, as in
        ``pulse.ActiveStress``: a purely fibre-directed active stress is not
        split into deviatoric and volumetric parts.
        """
        lmbda = ufl.sqrt(ufl.inner(C * self.f0, self.f0))
        Ta = self.Ta(lmbda)
        if self.formulation == pulse.ActiveStressFormulation.stretch:
            Ta = Ta / lmbda
        return Ta * ufl.outer(self.f0, self.f0)

    def P(self, F: ufl.core.expr.Expr, dev: bool = False) -> ufl.core.expr.Expr:
        return F * self.S(F.T * F, dev=dev)

    # ------------------------------------------------------------------
    # Land model
    # ------------------------------------------------------------------

    @property
    def dt(self) -> float:
        return float(self.t - self._t_prev)

    def dLambda(self, lmbda):
        logger.debug("Evaluate dLambda")
        if self.dt == 0:
            return self._dLambda
        return (lmbda - self.lmbda_prev) / self.dt

    @property
    def Aw(self):
        Tot_A = self._parameters["Tot_A"]
        rs = self._parameters["rs"]
        rw = self._parameters["rw"]
        scale_popu_rw = 1.0
        scale_popu_rs = 1.0
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
        scale_popu_kuw = 1.0
        scale_popu_rw = 1.0
        return kuw * scale_popu_kuw * phi * (1.0 - (rw * scale_popu_rw)) / (rw * scale_popu_rw)

    @property
    def cs(self):
        phi = self._parameters["phi"]
        kws = self._parameters["kws"]
        rs = self._parameters["rs"]
        rw = self._parameters["rw"]
        scale_popu_kws = 1.0
        scale_popu_rw = 1.0
        scale_popu_rs = 1.0
        return (
            kws
            * scale_popu_kws
            * phi
            * rw
            * scale_popu_rw
            * (1.0 - (rs * scale_popu_rs))
            / (rs * scale_popu_rs)
        )

    def Zetas(self, lmbda):
        return _Zeta(self.Zetas_prev, self.As, self.cs, self.dLambda(lmbda), self.dt, self._scheme)

    def Zetaw(self, lmbda):
        return _Zeta(self.Zetaw_prev, self.Aw, self.cw, self.dLambda(lmbda), self.dt, self._scheme)

    def update_Zetas(self, lmbda):
        logger.debug("update Zetas")
        self._Zetas.interpolate(
            dolfinx.fem.Expression(
                self.Zetas(lmbda),
                self.function_space.element.interpolation_points,
            ),
        )

    def update_Zetaw(self, lmbda):
        logger.debug("update Zetaw")
        self._Zetaw.interpolate(
            dolfinx.fem.Expression(
                self.Zetaw(lmbda),
                self.function_space.element.interpolation_points,
            ),
        )

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
        """Active tension as a UFL expression of the fibre stretch.

        Rebuilt from ``Zetas``/``Zetaw`` on every call, so that the assembled
        form carries the dependence on the current displacement and Newton
        differentiates through it.
        """
        logger.debug("Evaluate Ta")

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

        lmbda_capped = _min(1.2, lmbda)
        h_lambda_prima = 1.0 + Beta0 * (lmbda_capped + _min(lmbda_capped, 0.87) - 1.87)
        h_lambda = _max(0.0, h_lambda_prima)

        Zetas = self.Zetas(lmbda)
        Zetaw = self.Zetaw(lmbda)
        active_fraction = self.XS * (Zetas + 1.0) + self.XW * Zetaw
        active_fraction = ufl.max_value(0.0, active_fraction)

        return 1000 * h_lambda * (Tref * scale_popu_Tref / (rs * scale_popu_rs)) * active_fraction
