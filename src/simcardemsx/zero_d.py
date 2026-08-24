"""A 0D reference for the force-generation / mechanics coupling.

This replaces the finite-element mechanics with the zero-dimensional tissue
model of Regazzoni & Quarteroni (2021), Eq. (54),

.. math::
    M \\ddot{\\lambda} + \\sigma \\dot{\\lambda}
    + \\frac{\\partial W}{\\partial \\lambda}(\\lambda) + T_a = p(t)

so that the *same* contraction models used by the FEM backends can be driven
through the *same* three coupling schemes -- monolithic, segregated, and
stabilized-segregated -- with no FEM anywhere. That makes it the arbiter for
whether the stabilization in
:class:`~simcardemsx.backends.segregated.CrossbridgeSegregated` is doing what
it claims: the monolithic run here is a genuine reference solution, not an
approximation of one, and the setup reproduces a published test case.

Units
-----
**Seconds and pascals**, following the paper, not the milliseconds used
elsewhere in simcardemsx. This module exists to reproduce R&Q's Test Case 1,
and translating their parameters into other units would obscure exactly the
comparison it is for. ``crossbridge`` also works in seconds, so nothing needs
converting here.

Strain vs stretch
-----------------
R&Q write :math:`\\lambda` for the fibre *strain*, zero at rest. The rest of
simcardemsx uses the *stretch*, one at rest. This module follows the paper and
calls its state ``e`` for strain, converting at the crossbridge boundary via
``SL = (1 + e) * SL_ref``. :attr:`Result.stretch` reports ``1 + e`` for
comparison with the FEM backends.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, Literal

import crossbridge
import numpy as np
import numpy.typing as npt
import scipy.optimize

Scheme = Literal["monolithic", "segregated", "stabilized"]

#: The three coupling schemes compared in R&Q Sec. 5.2.
SCHEMES: tuple[Scheme, ...] = ("monolithic", "segregated", "stabilized")


@dataclass
class Tissue:
    """The 0D tissue, R&Q Eq. (54) with their nonlinear elastic potential.

    ``Kp`` is the passive stiffness. It is the parameter that decides whether
    the naive segregated scheme is stable at all: R&Q show the scheme's
    amplification factor tends to :math:`-K_a/K_p`, so it diverges once the
    active stiffness exceeds the passive one. Lowering ``Kp`` is therefore how
    you enter the unstable regime deliberately.
    """

    Kp: float = 1e6  # [Pa]
    sigma: float = 10.0  # [Pa s]   viscous modulus
    M: float = 0.1  # [Pa s^2] normalized mass

    def stress(self, e: float) -> float:
        """dW/de for W(e) = Kp * e * log(1 + e) / 2."""
        return 0.5 * self.Kp * (np.log1p(e) + e / (1.0 + e))

    def stiffness(self, e: float) -> float:
        """d2W/de2 -- the passive stiffness Ka is compared against."""
        return 0.5 * self.Kp * (2.0 / (1.0 + e) - e / (1.0 + e) ** 2)


def calcium_transient(
    t: npt.NDArray | float,
    c0: float = 0.1,
    cmax: float = 1.6,
    t0: float = 0.1,
    tau1: float = 0.02,
    tau2: float = 0.05,
) -> npt.NDArray:
    """R&Q Eq. (57): the idealized calcium transient of their Test Case 1.

    Returns micromolar, which is what crossbridge expects.
    """
    t = np.asarray(t, dtype=float)
    beta = (tau1 / tau2) ** (-1.0 / (tau1 / tau2 - 1.0)) - (tau1 / tau2) ** (
        -1.0 / (1.0 - tau2 / tau1)
    )
    ca = np.full_like(t, c0)
    active = t >= t0
    dt = t[active] - t0 if t.ndim else np.array([t - t0])
    ca_active = c0 + (cmax - c0) / beta * (np.exp(-dt / tau1) - np.exp(-dt / tau2))
    ca[active] = ca_active
    return ca


@dataclass
class Result:
    """Time series from a :func:`solve` run."""

    t: npt.NDArray
    strain: npt.NDArray
    Ta: npt.NDArray
    Ka: npt.NDArray
    scheme: str
    dt: float
    newton_iterations: int = 0

    @property
    def stretch(self) -> npt.NDArray:
        """``1 + strain``, for comparison with the FEM backends."""
        return 1.0 + self.strain

    @property
    def oscillation_metric(self) -> float:
        """Fraction of steps at which the strain reverses direction.

        The instability R&Q describe shows up as the solution alternating every
        step rather than as a blow-up, so counting sign changes of the
        increment detects it far more reliably than any amplitude threshold.
        A smooth twitch contracts then relaxes, giving a value near zero.
        """
        d = np.diff(self.strain)
        if d.size < 2:
            return 0.0
        return float(np.count_nonzero(np.diff(np.sign(d)) != 0) / (d.size - 1))


def solve(
    scheme: Scheme = "stabilized",
    *,
    model: str = "Land2017",
    tissue: Tissue | None = None,
    dt: float = 1e-3,
    T: float = 0.8,
    load: float = 0.0,
    SL_ref: float | None = None,
    params: dict | None = None,
    ca: Callable = calcium_transient,
) -> Result:
    """Run the 0D coupled problem under one of the three schemes.

    Parameters
    ----------
    scheme:
        ``"monolithic"`` solves the contraction model and the tissue balance
        simultaneously, by driving the strain at which the ODE is advanced to
        satisfy the balance -- a true reference solution.
        ``"segregated"`` advances the ODE with the previous step's strain and
        then solves the balance with the resulting tension held fixed. This is
        the scheme R&Q show to be non-convergent for ``Ka > Kp``; it is here to
        be measured, not used.
        ``"stabilized"`` is ``"segregated"`` plus the consistent term
        ``Ka (e_new - e_old)`` in the balance -- R&Q Eq. (19).
    load:
        Applied load ``p`` [Pa]. Zero is the isotonic twitch of Test Case 1.

    Notes
    -----
    All three schemes advance the contraction model identically; they differ
    only in which strain it sees and in what the tissue balance is solved
    against. That is the point: any difference in the result is attributable to
    the coupling scheme rather than to the model or the integrator.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}; expected one of {SCHEMES}")

    tissue = tissue or Tissue()
    ModelClass = crossbridge.get_model(model)
    cell = ModelClass(num_cells=1, params=params)
    SL_ref = SL_ref if SL_ref is not None else cell.p.get("SL0")
    if SL_ref is None:
        raise ValueError(f"{model} defines no SL0; pass SL_ref explicitly")

    n_steps = int(round(T / dt))
    ts = np.arange(n_steps + 1) * dt

    e_hist = np.zeros(n_steps + 1)
    Ta_hist = np.zeros(n_steps + 1)
    Ka_hist = np.zeros(n_steps + 1)

    e_old = 0.0  # e^(k-1)
    e_cur = 0.0  # e^(k)
    total_iterations = 0

    def advance(state, e_for_ode: float, e_prev: float, Ca: float):
        """Advance a *copy* of the contraction model, returning (Ta, Ka) in Pa.

        Copied rather than advanced in place so the monolithic root-find can
        evaluate trial strains without consuming the step.
        """
        trial = copy.deepcopy(state)
        SL = (1.0 + e_for_ode) * SL_ref
        dSL = (e_for_ode - e_prev) * SL_ref / dt
        trial.advance_step(dt, Ca, np.array([SL]), dSL_vals=np.array([dSL]))
        # crossbridge reports kPa; the tissue model is in Pa
        return (
            trial,
            1e3 * float(trial.get_active_tension()[0]),
            1e3
            * float(
                trial.get_active_stiffness()[0],
            ),
        )

    def inertia(e_new: float) -> float:
        return (
            tissue.M * (e_new - 2.0 * e_cur + e_old) / dt**2 + tissue.sigma * (e_new - e_cur) / dt
        )

    for k in range(n_steps):
        t_next = ts[k + 1]
        Ca = float(np.atleast_1d(ca(t_next))[0])

        if scheme == "monolithic":
            # The ODE sees the *new* strain, so the balance and the contraction
            # model are solved together. This is the whole difference.
            def residual(e_new: float) -> float:
                _, Ta, _ = advance(cell, e_new, e_cur, Ca)
                return inertia(e_new) + tissue.stress(e_new) + Ta - load

            e_new = float(scipy.optimize.newton(residual, e_cur, tol=1e-12, maxiter=100))
            total_iterations += 1
            cell, Ta, Ka = advance(cell, e_new, e_cur, Ca)
        else:
            # The ODE sees the previous strain; only the balance differs below.
            cell, Ta, Ka = advance(cell, e_cur, e_old, Ca)

            if scheme == "stabilized":
                # R&Q Eq. (19): the active tension enters as a spring, not a
                # dead load. Linear in e_new, so no iteration is needed.
                def residual(e_new: float) -> float:
                    return inertia(e_new) + tissue.stress(e_new) + Ta + Ka * (e_new - e_cur) - load
            else:

                def residual(e_new: float) -> float:
                    return inertia(e_new) + tissue.stress(e_new) + Ta - load

            # The naive scheme is the one under test *for divergence* (see
            # module docstring and test_naive_scheme_does_not_converge): at
            # fine dt in the unstable regime it can genuinely blow past the
            # strain domain (e <= -1, where tissue.stress's log1p is undefined)
            # before 100 iterations are up. That is the expected failure mode,
            # not a bug -- so let it return whatever it lands on (nan or not)
            # instead of raising, and keep failing loudly for "stabilized",
            # which is supposed to be well-posed here.
            e_new = float(
                scipy.optimize.newton(
                    residual,
                    e_cur,
                    tol=1e-12,
                    maxiter=100,
                    disp=(scheme != "segregated"),
                ),
            )
            total_iterations += 1

        e_old, e_cur = e_cur, e_new
        e_hist[k + 1] = e_new
        Ta_hist[k + 1] = Ta
        Ka_hist[k + 1] = Ka

    return Result(
        t=ts,
        strain=e_hist,
        Ta=Ta_hist,
        Ka=Ka_hist,
        scheme=scheme,
        dt=dt,
        newton_iterations=total_iterations,
    )


@dataclass
class Comparison:
    """All three schemes run under identical conditions."""

    results: dict[str, Result] = field(default_factory=dict)

    def __getitem__(self, scheme: str) -> Result:
        return self.results[scheme]

    def error_against_monolithic(self, scheme: str) -> float:
        """Max absolute strain difference from the monolithic reference."""
        ref = self.results["monolithic"]
        return float(np.max(np.abs(self.results[scheme].strain - ref.strain)))


def compare(**kwargs) -> Comparison:
    """Run all three schemes with identical settings."""
    return Comparison({s: solve(scheme=s, **kwargs) for s in SCHEMES})
