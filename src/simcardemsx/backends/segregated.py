"""The Ca_i split: a crossbridge contraction model, segregated but stabilized.

The contraction model here is one of the NumPy models from ``crossbridge``, so
unlike :class:`~simcardemsx.backends.generated.GeneratedActivation` it cannot be
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

The backend is a :class:`~simcardemsx.backends.base.CoupledBackend`, so
:class:`~simcardemsx.controller.SimulationController` drives it. A step is a trial
until it is accepted: :meth:`~CrossbridgeSegregated.begin_step` advances a deep copy
of the model with the accepted stretch and writes its ``Ta`` and ``Ka`` into the
stress, and :meth:`~CrossbridgeSegregated.post_solve` commits it once the solve has
converged. A step whose solve fails is never accepted.
"""

from __future__ import annotations

import copy
import logging
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import basix.ufl
import crossbridge
import dolfinx
import numpy as np
import pulse
import ufl

from .. import units
from ..averaging import make_averager
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
        ``lmbda_prev`` -- lives on this one space, which is what keeps the
        stabilization consistent.
    quadrature_degree:
        Instead of ``element``, a scalar quadrature space at this degree, as
        :class:`~simcardemsx.backends.generated.GeneratedActivation` uses. Under
        :class:`~simcardemsx.controller.SimulationController` it must equal the
        quadrature degree of the mechanics form, which the controller checks.
        Giving both ``element`` and ``quadrature_degree`` is a ``ValueError``.
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

    Attributes
    ----------
    evaluate_at_end_of_step:
        ``True`` (overriding :class:`pulse.active_model.ActiveModel`'s default): the
        increment ``λ(u) - λ_n`` is measured from the stretch the model was advanced
        with, so ``pulse.DynamicProblem`` must assemble ``S`` at the true end-of-step
        displacement rather than at its ``alpha_f`` point. No effect under
        ``pulse.StaticProblem``.
    space, quadrature_degree, mesh:
        The space everything below lives on, its quadrature degree (``None`` unless it
        is a quadrature space), and the mechanics mesh.
    inputs, outputs:
        ``{"cai": cai}`` and ``{"J_TRPN": J_TRPN, "lmbda": lmbda_prev}``, in EP's mM,
        mM/ms and the dimensionless stretch. ``outputs["lmbda"]`` is ``lmbda_prev``
        itself, the accepted λ(u) that :meth:`post_solve` and :meth:`reset_stretch`
        update in place, as in :class:`~simcardemsx.backends.generated.GeneratedActivation`:
        an EP remainder that keeps ``lmbda`` as a parameter is sent it.
    lmbda_prev:
        The accepted fibre stretch λ_n: the stretch the next step advances the model
        with, and the one the stabilization measures its increment from. It starts at
        1, the reference configuration; for a problem that starts deformed, see
        :meth:`reset_stretch`.
    Ta_current, Ka_current:
        ``Ta`` and ``Ka`` (kPa; ``Ka`` per unit of the solver's λ) of the step being
        solved, read by the stress form. :meth:`begin_step` writes them.
    active_tension:
        The accepted ``Ta`` averaged onto P1, in kPa.

    Notes
    -----
    Time is in **ms** here, matching the EP side and the rest of simcardemsx;
    crossbridge works in seconds, and the conversion happens in
    :meth:`begin_step`.
    """

    evaluate_at_end_of_step = True

    def __init__(
        self,
        f0,
        mesh: dolfinx.mesh.Mesh,
        model: str | type = "Land2017",
        *,
        element: tuple[str, int] | None = None,
        quadrature_degree: int | None = None,
        SL_ref: float | None = None,
        trpnmax: float = units.TRPNMAX_MM,
        params: dict | None = None,
        stabilized: bool = True,
    ):
        if element is not None and quadrature_degree is not None:
            raise ValueError(
                "Give the activation space as element or as quadrature_degree, not both "
                f"(got element={element!r} and quadrature_degree={quadrature_degree})",
            )
        self.f0 = f0
        self.mesh = mesh
        self.trpnmax = trpnmax
        self.stabilized = stabilized
        self.quadrature_degree = quadrature_degree

        if quadrature_degree is not None:
            self.space = dolfinx.fem.functionspace(
                mesh,
                basix.ufl.quadrature_element(
                    mesh.basix_cell(),
                    value_shape=(),
                    degree=quadrature_degree,
                ),
            )
        else:
            self.space = dolfinx.fem.functionspace(mesh, element or ("DG", 1))

        # One value per local dof, ghosts included, so the arrays line up with
        # x.array without a scatter before assembly. Mixing this convention with
        # size_local elsewhere is the classic way to break this in parallel.
        num_cells = self.space.dofmap.index_map.size_local
        num_cells += self.space.dofmap.index_map.num_ghosts
        num_cells *= self.space.dofmap.index_map_bs
        self.num_cells = num_cells

        ModelClass = crossbridge.get_model(model) if isinstance(model, str) else model
        self.model = ModelClass(num_cells=num_cells, params=params)
        #: The step :meth:`begin_step` advanced, and its dt [ms], until
        #: :meth:`post_solve` commits it.
        self._trial: crossbridge.base.CardiacActivationModel | None = None
        self._trial_dt = 0.0

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
        self._lmbda_expression: dolfinx.fem.Expression | None = None

        V = self.space
        self.cai = dolfinx.fem.Function(V, name="cai")
        self.J_TRPN = dolfinx.fem.Function(V, name="J_TRPN")
        self.inputs = {"cai": self.cai}
        self.outputs = {"J_TRPN": self.J_TRPN}
        self.Ta_current = dolfinx.fem.Function(V, name="Ta")
        self.Ka_current = dolfinx.fem.Function(V, name="Ka")
        self._tension_kPa = dolfinx.fem.Function(V, name="tension_kPa")
        self._stiffness_kPa = dolfinx.fem.Function(V, name="stiffness_kPa")
        self.active_tension = dolfinx.fem.Function(
            dolfinx.fem.functionspace(mesh, ("P", 1)),
            name="active_tension",
        )
        self._average_tension = make_averager(self._tension_kPa, self.active_tension)

        #: λ_n. The trial is advanced with exactly this array and the stabilization
        #: measures its increment from this Function, which only :meth:`post_solve`
        #: and :meth:`reset_stretch` write, between steps.
        self.lmbda_prev = dolfinx.fem.Function(V, name="lambda_prev")
        self.lmbda_prev.x.array[:] = 1.0
        self.outputs["lmbda"] = self.lmbda_prev
        # λ_{n-1}, for the shortening velocity.
        self._lmbda_old = self.lmbda_prev.x.array.copy()

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

    @property
    def missing(self) -> Mapping[str, int]:
        """What the backend needs from EP, name -> index: ``cai``."""
        return {"cai": 0}

    @property
    def provides(self) -> Mapping[str, int]:
        """What the backend hands to EP, name -> index: ``J_TRPN``."""
        return {"J_TRPN": 0}

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
        return self.inputs

    @property
    def ep_outputs(self) -> dict[str, dolfinx.fem.Function]:
        return self.outputs

    def register(self, u: dolfinx.fem.Function) -> None:
        """Receive the displacement (``pulse.StaticProblem`` calls this) and compile
        λ(u), which :meth:`post_solve` evaluates."""
        self.u = u
        F = ufl.Identity(self.mesh.geometry.dim) + ufl.grad(u)
        self._lmbda_expression = dolfinx.fem.Expression(
            ufl.sqrt(ufl.inner((F.T * F) * self.f0, self.f0)),
            self.space.element.interpolation_points,
        )

    def begin_step(self, t_n: float, dt: float) -> None:
        """Advance a trial copy of the model from ``t_n`` by ``dt`` [ms]; accept nothing.

        The trial is a deep copy of the accepted model, so calling this again before
        :meth:`post_solve` (a retried step) starts from the accepted state again. It
        is advanced with ``inputs["cai"]``, the accepted stretch λ_n
        (``lmbda_prev``), and the velocity (λ_n - λ_{n-1}) / dt. Its ``Ta`` and
        ``Ka`` are written into ``Ta_current``/``Ka_current``, which the stress reads.

        The stabilization measures its increment ``λ(u) - λ_n`` from ``lmbda_prev``,
        the very array the trial was advanced with, and only :meth:`post_solve` and
        :meth:`reset_stretch` write it, between steps: the two cannot drift apart. If
        they did, the extra term would stop being a consistent O(dt) perturbation and
        could destabilize the solve it was added to stabilize.

        At ``dt == 0`` the trial is not advanced, so the step is the identity, and
        :meth:`post_solve` then accepts ``u`` as the rest state.
        """
        self._trial = None  # a retry's first trial need not outlive the copy below
        trial = copy.deepcopy(self.model)
        if dt > 0:
            lmbda = self.lmbda_prev.x.array
            # Velocity from the two most recent stretches. Passed explicitly rather
            # than letting crossbridge finite-difference its own call history, which
            # need not share this dt and would use a different lambda than the one
            # the stabilization term is built on.
            dt_s = units.ms_to_s(dt)
            SL = units.stretch_to_sarcomere_length(lmbda, self.SL_ref)
            dSL = units.stretch_to_sarcomere_length(lmbda - self._lmbda_old, self.SL_ref) / dt_s
            Ca = units.calcium_to_crossbridge(self.cai.x.array)
            trial.advance_step(dt_s, Ca, SL, dSL_vals=dSL)
        self._trial = trial
        self._trial_dt = dt

        self.Ta_current.x.array[:] = trial.get_active_tension()
        self.Ka_current.x.array[:] = units.active_stiffness_to_mechanics(
            trial.get_active_stiffness(),
            SL_ref=self.SL_ref,
            SL0=self.SL0,
        )

    def step(self, t: float, dt: float | None = None) -> None:
        """Deprecated alias of :meth:`begin_step`, which accepts nothing: call
        :meth:`post_solve` after the solve to accept the step."""
        warnings.warn(
            "CrossbridgeSegregated.step is deprecated: call begin_step(t_n, dt), and "
            "post_solve() after the solve to accept the step.",
            DeprecationWarning,
            stacklevel=2,
        )
        if dt is None:
            raise ValueError("CrossbridgeSegregated.step requires an explicit dt [ms]")
        self.begin_step(t, dt)

    def post_solve(self) -> None:
        """Accept the step :meth:`begin_step` prepared, after its solve converged.

        Commits the trial; takes λ_n as λ_{n-1} and λ(u) at the converged
        displacement as the new λ_n (without :meth:`register` there is no
        displacement, and λ stays as it is); sets ``J_TRPN`` from the committed
        model's calcium binding over the step; and records its ``Ta``/``Ka`` in
        :attr:`tension_kPa`/:attr:`stiffness_kPa` and the P1 :attr:`active_tension`.

        After a step at ``dt == 0`` (the identity) it accepts ``u`` as the rest state,
        as :meth:`reset_stretch` does: λ_{n-1} and λ_n are both λ(u), so a ``u`` that
        moved during that solve, e.g. an unloaded one, gives the next step no
        shortening velocity.

        Raises
        ------
        RuntimeError
            If no :meth:`begin_step` has been called since the last ``post_solve``.
        """
        if self._trial is None:
            raise RuntimeError(
                "post_solve() accepts the step begin_step() prepared, and begin_step() "
                "has not been called since the last post_solve()",
            )
        self.model = self._trial
        self._trial = None

        lmbda_n = self.lmbda_prev.x.array.copy()
        if self._lmbda_expression is not None:
            self.lmbda_prev.interpolate(self._lmbda_expression)
        self._lmbda_old = lmbda_n if self._trial_dt > 0 else self.lmbda_prev.x.array.copy()

        self.J_TRPN.x.array[:] = units.troponin_flux(
            self.model.get_calcium_binding_rate(),
            trpnmax_mM=self.trpnmax,
        )
        self._tension_kPa.x.array[:] = self.Ta_current.x.array
        self._stiffness_kPa.x.array[:] = self.Ka_current.x.array
        self._average_tension()

    def reset_stretch(self) -> None:
        """Take λ(u) of the registered displacement as the stretch at rest.

        For a problem that starts deformed: otherwise the first step is advanced at
        λ = 1, and the stabilization measures its increment from 1. ``lmbda_prev``
        (λ_n) and λ_{n-1} are both set to λ(u), so the first step is advanced at λ(u)
        with no shortening velocity. The model, ``outputs`` (but for ``outputs["lmbda"]``,
        which is ``lmbda_prev``) and the reported tension are untouched, and nothing is
        sent to EP. Call it between steps. A trial pending from :meth:`begin_step` was
        advanced with the old λ_n, so it is discarded: a fresh :meth:`begin_step` is
        needed, and a :meth:`post_solve` without one raises ``RuntimeError``.

        The other way to start deformed is one accepted step at ``dt == 0``, where the
        step is the identity.

        Raises
        ------
        RuntimeError
            If :meth:`register` has not been called: there is no displacement.
        """
        if self._lmbda_expression is None:
            raise RuntimeError("register(u) must be called before reset_stretch()")
        self._trial = None
        self.lmbda_prev.interpolate(self._lmbda_expression)
        self._lmbda_old = self.lmbda_prev.x.array.copy()

    @property
    def tension_kPa(self) -> dolfinx.fem.Function:
        """``Ta`` of the last accepted step, in kPa, on :attr:`space` (read-only).

        The ``Function`` :meth:`post_solve` writes in place and :attr:`active_tension`
        is averaged from. ``Ta_current`` holds the same values once the step is
        accepted, but :meth:`begin_step` overwrites it with the next trial's.
        """
        return self._tension_kPa

    @property
    def stiffness_kPa(self) -> dolfinx.fem.Function:
        """``Ka`` of the last accepted step, in kPa per unit of the solver's λ, on
        :attr:`space` (read-only): the continuous rate path of R&Q's Eq. (42), as
        crossbridge reports it, rescaled by ``SL_ref / SL0``. With ``stabilized=False``
        it is still reported, although the stress does not use it."""
        return self._stiffness_kPa

    @property
    def active_stiffness(self) -> dolfinx.fem.Function:
        """``Ka_current``: ``Ka`` of the step being solved, from :meth:`begin_step`."""
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

        ``dev`` is accepted for protocol compatibility and ignored: pulse's
        ``StabilizedActiveStress.S`` (unlike ``HyperElasticMaterial.S``) has no
        ``dev`` argument, since active tension along a fiber is not split into
        deviatoric/volumetric parts (pulse commit 87b973e).
        """
        return self._active.S(C)

    def P(self, F: ufl.core.expr.Expr, dev: bool = False) -> ufl.core.expr.Expr:
        return F * self.S(F.T * F, dev=dev)

    # ------------------------------------------------------------------
    # Checkpoint / restart (simcardemsx.checkpoint.Checkpointable)
    # ------------------------------------------------------------------

    namespace = "activation"

    @property
    def step_pending(self) -> bool:
        """Whether :meth:`begin_step` has prepared a trial that :meth:`post_solve` has
        not yet accepted."""
        return self._trial is not None

    def _holder_space(self, bs: int) -> dolfinx.fem.FunctionSpace:
        """The space of a Function holding ``bs`` values per point: :attr:`space` itself
        for ``bs == 1``, else the same element with ``value_shape=(bs,)``. Cached."""
        if bs == 1:
            return self.space
        cache = self.__dict__.setdefault("_holder_spaces", {})
        if bs not in cache:
            if self.quadrature_degree is not None:
                element: Any = basix.ufl.quadrature_element(
                    self.mesh.basix_cell(),
                    value_shape=(bs,),
                    degree=self.quadrature_degree,
                )
            else:
                element = ("DG", 1, (bs,))
            cache[bs] = dolfinx.fem.functionspace(self.mesh, element)
        return cache[bs]

    def _holder(self, name: str, array: np.ndarray) -> dolfinx.fem.Function:
        """A Function holding ``array`` of shape ``(..., num_cells)``, cell axis last."""
        bs = int(np.prod(array.shape[:-1]))
        f = dolfinx.fem.Function(self._holder_space(bs), name=name)
        f.x.array.reshape(array.shape[-1], bs)[:] = array.reshape(bs, array.shape[-1]).T
        return f

    @staticmethod
    def _unpack(f: dolfinx.fem.Function, shape: tuple[int, ...]) -> np.ndarray:
        bs = int(np.prod(shape[:-1]))
        return f.x.array.reshape(shape[-1], bs).T.reshape(shape).copy()

    def restart_functions(self) -> list[tuple[str, dolfinx.fem.Function]]:
        """The accepted state as Functions, in a fixed order.

        The model's arrays are copied into freshly allocated holders on every call, so
        this is valid while a step is pending (the model is still the accepted one);
        the checkpointer refuses to write then, through :attr:`step_pending`.
        """
        lmbda_old = dolfinx.fem.Function(self.space, name="lmbda_old")
        lmbda_old.x.array[:] = self._lmbda_old
        functions = [
            ("activation_lmbda_prev", self.lmbda_prev),
            ("activation_lmbda_old", lmbda_old),
            ("activation_tension_kPa", self._tension_kPa),
            ("activation_stiffness_kPa", self._stiffness_kPa),
            ("activation_output_J_TRPN", self.J_TRPN),
        ]
        state = self.model.get_state()
        for key in sorted(k for k, v in state.items() if isinstance(v, np.ndarray)):
            name = f"activation_model_{key}"
            functions.append((name, self._holder(name, state[key])))
        return functions

    def restart_metadata(self) -> dict[str, Any]:
        state = self.model.get_state()
        return {
            "backend": "CrossbridgeSegregated",
            "model": type(self.model).__name__,
            "stabilized": self.stabilized,
            "model_scalars": {k: v for k, v in state.items() if not isinstance(v, np.ndarray)},
        }

    def load_restart(
        self,
        functions: Sequence[tuple[str, dolfinx.fem.Function]],
        metadata: Mapping[str, Any],
    ) -> None:
        """Take the accepted state back; a pending step is discarded."""
        own = self.restart_metadata()
        for key in ("backend", "model", "stabilized"):
            if metadata.get(key) != own[key]:
                raise ValueError(
                    f"Checkpoint has {key}={metadata.get(key)!r}, this backend has {own[key]!r}",
                )
        by_name = dict(functions)
        state = self.model.get_state()
        new: dict[str, Any] = dict(metadata["model_scalars"])
        for key, value in state.items():
            if isinstance(value, np.ndarray):
                new[key] = self._unpack(by_name[f"activation_model_{key}"], value.shape)
        self._lmbda_old = by_name["activation_lmbda_old"].x.array.copy()
        self.model.set_state(new)
        self._trial = None
        self._average_tension()
