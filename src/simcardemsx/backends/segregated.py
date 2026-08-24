"""The Ca_i split: a crossbridge contraction model, segregated but stabilized.

The contraction model here is one of the NumPy models from ``crossbridge``, so
unlike :class:`~simcardemsx.backends.zeta_split.ZetaSplitUFL` it cannot be
re-integrated inside the Newton iteration and cannot be differentiated by UFL.
That forces the force-generation/mechanics interface to be **segregated**, and
that is not a free choice: Regazzoni & Quarteroni (2021) show the naive
staggered scheme has amplification factor tending to :math:`-K_a/K_p` as
:math:`\\Delta t \\to 0`, so once active stiffness exceeds passive stiffness --
routine in contracting myocardium -- it is not merely inaccurate but *not
convergent*. Refining the time step makes it worse.

This backend therefore uses their stabilized-segregated scheme. The active
stress carries a consistent extra term,

.. math::
    \\mathbf{P}_{act} = \\left[T_a + K_a(\\lambda - \\lambda_{prev})\\right]
        \\frac{\\mathbf{F} f_0 \\otimes f_0}{|\\mathbf{F} f_0|}

which is :math:`\\mathcal{O}(\\Delta t)` and vanishes in the limit -- the same
continuous problem -- but is unconditionally stable. Physically it stops
treating active tension as a dead load during the solve and treats it as what
it is: a population of crossbridges acting as springs.

Because crossbridge owns the whole contraction model, troponin included, this
is the Ca_i split: calcium crosses in, and the troponin buffering flux
``J_TRPN`` must cross back, or the EP model's calcium transient runs unbuffered.
"""

from __future__ import annotations

import logging

import crossbridge
import dolfinx
import numpy as np
import pulse
import ufl

from .. import units
from .base import Transfer

logger = logging.getLogger(__name__)


class CrossbridgeSegregated(pulse.active_model.ActiveModel):
    """A ``crossbridge`` contraction model coupled by the stabilized scheme.

    Parameters
    ----------
    f0:
        Fibre direction.
    mesh:
        The mechanics mesh.
    model:
        Name of a model in ``crossbridge.MODEL_REGISTRY`` (``"Land2017"``,
        ``"Lewalle2024"``, ``"RDQ18"``, ``"RDQ20MF"``), or a class.
    element:
        Function space for the activation state, as a
        ``dolfinx.fem.functionspace`` spec. Defaults to ``("DG", 1)``, matching
        the zeta-split backend and the discretization used in the simcardems
        paper. Everything -- the ODE state, ``Ta``, ``Ka``, and
        ``lambda_prev`` -- lives on this one space, which is what keeps the
        stabilization consistent.
    SL_ref:
        Sarcomere length [um] of the reference configuration, i.e. where
        ``lmbda == 1``. Defaults to the model's own slack length ``SL0``, which
        makes the stiffness rescaling in :mod:`simcardemsx.units` a no-op.
        Required for models that do not define ``SL0`` -- currently ``RDQ18``,
        which uses ``SL`` directly rather than a normalized stretch.
    trpnmax:
        Total troponin concentration [mM] used to turn the model's calcium
        binding rate into ``J_TRPN``. Defaults to ToR-ORd's 0.07 mM.
    params:
        Parameter overrides forwarded to the crossbridge model.
    stabilized:
        Whether to include the ``Ka`` term. **Leave this True.** It exists so
        that tests can demonstrate the instability it removes; a False run is
        the non-convergent scheme described above.

    Notes
    -----
    Time is in **ms** here, matching the EP side and the rest of simcardemsx;
    crossbridge works in seconds, and the conversion happens in
    :meth:`step`.
    """

    def __init__(
        self,
        f0,
        mesh,
        model: str | type = "Land2017",
        *,
        element=("DG", 1),
        SL_ref: float | None = None,
        trpnmax: float = units.TRPNMAX_MM,
        params: dict | None = None,
        stabilized: bool = True,
    ):
        self.f0 = f0
        self.trpnmax = trpnmax
        self.stabilized = stabilized

        self.function_space = dolfinx.fem.functionspace(mesh, element)

        # One value per local dof, ghosts included, so the arrays line up with
        # x.array without a scatter before assembly. Mixing this convention with
        # size_local elsewhere is the classic way to break this in parallel.
        num_cells = self.function_space.dofmap.index_map.size_local
        num_cells += self.function_space.dofmap.index_map.num_ghosts
        num_cells *= self.function_space.dofmap.index_map_bs
        self.num_cells = num_cells

        ModelClass = crossbridge.get_model(model) if isinstance(model, str) else model
        self.model = ModelClass(num_cells=num_cells, params=params)

        # Most models normalize length as Lambda = SL / SL0 and expose SL0.
        # RDQ18 does not: it uses SL directly in its overlap function Chi(SL),
        # so it has filament geometry (LA/LM/LB) but no slack length. There is
        # no way to infer one from its parameters, and picking a number would
        # silently place the model at an arbitrary point on its force-length
        # curve -- which scales the force it generates. So require SL_ref.
        model_SL0 = self.model.p.get("SL0")
        if SL_ref is None:
            if model_SL0 is None:
                raise ValueError(
                    f"{ModelClass.__name__} does not define SL0, so the sarcomere "
                    "length of the reference configuration cannot be inferred. "
                    "Pass SL_ref explicitly, e.g. SL_ref=2.0 [um].",
                )
            SL_ref = model_SL0
        self.SL_ref = SL_ref
        # With no Lambda normalization there is no chain rule to apply, so the
        # rescaling factor is 1. (Safe regardless here: the only such model,
        # RDQ18, has no strain-rate feedback and reports Ka = 0.)
        self.SL0 = SL_ref if model_SL0 is None else model_SL0

        self.u: dolfinx.fem.Function | None = None

        V = self.function_space
        self.cai = dolfinx.fem.Function(V, name="cai")
        self.J_TRPN = dolfinx.fem.Function(V, name="J_TRPN")
        self.Ta_current = dolfinx.fem.Function(V, name="Ta")
        self.Ka_current = dolfinx.fem.Function(V, name="Ka")
        self.lmbda = dolfinx.fem.Function(V, name="lambda")
        self.lmbda.x.array[:] = 1.0
        #: The stretch the ODE was last advanced with. The stabilization term is
        #: measured against exactly this, which is why both are set together in
        #: :meth:`step` and nowhere else.
        self.lmbda_prev = dolfinx.fem.Function(V, name="lambda_prev")
        self.lmbda_prev.x.array[:] = 1.0

        # Stretch at the step before last, for the shortening velocity.
        self._lmbda_old = np.ones(num_cells)

        # The stress form itself is pulse's, not reimplemented here: one
        # implementation of Psi_a = Ta*dl + Ka*dl^2/2, tested upstream. When
        # unstabilized the form reads a separate Function that stays zero, so
        # that `active_stiffness` still reports the true Ka for inspection.
        self._Ka_form = self.Ka_current if stabilized else dolfinx.fem.Function(V, name="Ka_off")
        self._active = pulse.StabilizedActiveStress(
            f0=f0,
            activation=pulse.units.Variable(self.Ta_current, "kPa"),
            active_stiffness=pulse.units.Variable(self._Ka_form, "kPa"),
            lmbda_prev=self.lmbda_prev,
        )

        logger.debug("Created CrossbridgeSegregated with %s on %d points", model, num_cells)

    # ------------------------------------------------------------------
    # Coupling interface
    # ------------------------------------------------------------------

    def wants_from_ep(self) -> tuple[Transfer, ...]:
        """Intracellular calcium, in the EP model's own millimolar."""
        return (Transfer(name="cai", unit="mM"),)

    def gives_to_ep(self) -> tuple[Transfer, ...]:
        """The troponin buffering flux.

        Not optional. crossbridge owns troponin in this split, so the EP model
        must have had its own removed; its calcium balance is then missing this
        term and will run unbuffered if it is not transferred back. Nothing
        raises -- the calcium transient is just too large and too fast.
        """
        return (Transfer(name="J_TRPN", unit="mM/ms", kind="monitor"),)

    @property
    def ep_inputs(self) -> dict[str, dolfinx.fem.Function]:
        return {"cai": self.cai}

    @property
    def ep_outputs(self) -> dict[str, dolfinx.fem.Function]:
        return {"J_TRPN": self.J_TRPN}

    def register(self, u: dolfinx.fem.Function) -> None:
        """Receive the displacement from ``pulse.StaticProblem``."""
        self.u = u

    def step(self, t: float, dt: float | None = None) -> None:
        """Advance the contraction model by ``dt`` [ms], before the solve.

        Everything the stabilization depends on is set here, together:

        1. read the stretch produced by the *previous* solve,
        2. advance the ODE with it, and with the velocity implied by it,
        3. record it as ``lmbda_prev``, the point the stress increment is
           measured from.

        Keeping (2) and (3) in one place is deliberate. If the stretch used to
        advance the ODE and the one the stabilizer measures against ever drift
        apart, the extra term stops being a consistent O(dt) perturbation and
        can destabilize the solve it was added to stabilize.
        """
        if dt is None:
            raise ValueError("CrossbridgeSegregated.step requires an explicit dt [ms]")

        lmbda = self._current_stretch()

        # Velocity from the two most recent stretches. Passed explicitly rather
        # than letting crossbridge finite-difference its own call history, which
        # need not share this dt and would use a different lambda than the one
        # the stabilization term is built on.
        dt_s = units.ms_to_s(dt)
        SL = units.stretch_to_sarcomere_length(lmbda, self.SL_ref)
        dSL = units.stretch_to_sarcomere_length(lmbda - self._lmbda_old, self.SL_ref) / dt_s

        Ca = units.calcium_to_crossbridge(self.cai.x.array)

        self.model.advance_step(dt_s, Ca, SL, dSL_vals=dSL)

        self.Ta_current.x.array[:] = self.model.get_active_tension()
        self.Ka_current.x.array[:] = units.active_stiffness_to_mechanics(
            self.model.get_active_stiffness(),
            SL_ref=self.SL_ref,
            SL0=self.SL0,
        )
        self.J_TRPN.x.array[:] = units.troponin_flux(
            self.model.get_calcium_binding_rate(),
            trpnmax_mM=self.trpnmax,
        )

        self.lmbda.x.array[:] = lmbda
        self.lmbda_prev.x.array[:] = lmbda
        self._lmbda_old = lmbda

    def post_solve(self) -> None:
        """Nothing to do.

        Deliberately empty. Unlike the zeta split, this backend captures the
        stretch at the *start* of a step, so that the value the ODE was advanced
        with and the value the stabilization measures against are provably the
        same array. Recording anything here would reintroduce the possibility of
        those two diverging.
        """

    @property
    def active_tension(self) -> dolfinx.fem.Function:
        return self.Ta_current

    @property
    def active_stiffness(self) -> dolfinx.fem.Function:
        return self.Ka_current

    # ------------------------------------------------------------------
    # pulse.ActiveModel
    # ------------------------------------------------------------------

    def Fe(self, F: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        return self._active.Fe(F)

    def strain_energy(self, C: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        r"""The stabilized active energy,
        :math:`\Psi_a = T_a \Delta\lambda + \frac{1}{2} K_a \Delta\lambda^2`.

        Unlike the other backends this one *does* have a potential, because
        ``Ta`` and ``Ka`` are frozen data during the solve and only
        :math:`\lambda` depends on the displacement.
        """
        return self._active.strain_energy(C)

    def S(self, C: ufl.core.expr.Expr, dev: bool = False) -> ufl.core.expr.Expr:
        r"""Active second Piola-Kirchhoff stress,
        :math:`\mathbf{S}_a = (T_a + K_a\Delta\lambda)\, f_0 \otimes f_0 / \lambda`
        -- R&Q Eq. (19) in the reference configuration.
        """
        return self._active.S(C, dev=dev)

    def P(self, F: ufl.core.expr.Expr, dev: bool = False) -> ufl.core.expr.Expr:
        return F * self.S(F.T * F, dev=dev)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _current_stretch(self) -> np.ndarray:
        """Fibre stretch of the current displacement, on the activation space."""
        if self.u is None:
            return np.ones(self.num_cells)
        F = ufl.Identity(3) + ufl.grad(self.u)
        expr = dolfinx.fem.Expression(
            ufl.sqrt(ufl.inner((F.T * F) * self.f0, self.f0)),
            self.function_space.element.interpolation_points,
        )
        out = dolfinx.fem.Function(self.function_space)
        out.interpolate(expr)
        return out.x.array.copy()
