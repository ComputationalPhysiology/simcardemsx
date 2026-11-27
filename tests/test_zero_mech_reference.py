"""simcardemsx against zero-mech's 0D mechanics, where the two must agree to solver tolerance.

With rollers on the three faces through the origin and a uniform load, the one-element
cube deforms homogeneously, ``F = diag(λ1, λ2, λ3)``, on any mesh. The finite element
solution of the one-element gates is then the solution of a 0D problem: the stress
``P(F)`` vanishes. The 0D side here solves that problem with the stress of
`zero-mech <https://github.com/ComputationalPhysiology/zero-mech>`_: Holzapfel-Ogden and
the ``stretch`` active stress written as sympy strain energies and differentiated there,
independently of pulse's UFL forms, with the material parameters read off pulse and Land's
tension converted from kPa by a literal of its own.

The time stepping is the same on both sides by design, since it is what is being compared
against: the ``mechanics`` component generated as numpy (``conftest._numpy_mech``) and
advanced by GRL at the trial stretch inside the root find, which is what
``scheme="monolithic"`` does inside Newton. Whether the 0D solution is the right answer is
a separate question. It was checked once by hand when this test was written, not by a
test: for the Ca_i split at dt 1 to 0.002 ms it converges at first order (observed orders
1.02 to 1.00) to a Radau solution of the continuous problem, the index-reduced DAE.

- The one-element gates (gate 1's calcium, gate 4's zeta inputs): the mean of λ after each
  step equals the 0D stretch to ``LMBDA_ATOL``. Measured: 1.8e-14 (Ca_i) and 4.2e-14 (zeta).
  The 0D side with the ``invariant`` active stress instead misses by 1.4e-4 and 5e-3, and
  with Land's tension 1 % too large by 4e-5 and 4.5e-4.
- Gate 5's harness (beat EP on the 3x3x3 cube, ToR-ORd firing everywhere, one element of
  mechanics, through ``SimulationController``): everything is uniform, so the coupled
  problem is 0D too, and the 0D twin repeats the controller's documented order (EP
  micro-steps with the crossings frozen, each taken as beat's Strang step takes it, forward
  at the new time, monolithic mechanics, backward). λ, ``v``, ``cai`` and ``Ta`` agree to
  ``RTOL`` of their largest value. Measured: at most 4.4e-13 (the Ca_i split's ``Ta``); a
  twin taking one whole cell step per micro-step instead misses ``v`` by 6 %. Gate 5 itself
  only asserts that the values are nonzero.
"""

from collections.abc import Callable, Mapping
from types import ModuleType
from typing import cast

from mpi4py import MPI

import dolfinx
import numpy as np
import pulse
import pytest
import scipy.optimize
import sympy as sp
import zero_mech
from conftest import (
    _caisplit_inputs,
    _mechanics,
    _numpy_mech,
    _run,
    _zetasplit_inputs,
)
from test_round_trip import DT_EP, DT_MECH, NUM_STEPS

from simcardemsx.controller import SimulationController
from simcardemsx.ode_model import EP_SCHEME

#: Pa per kPa: Land's tension is in kPa, pulse's stresses in Pa
PA_PER_KPA = 1000.0
#: The 0D balance is accepted at this residual, in Pa
RESIDUAL_ATOL = 1e-7
#: Bound on |mean λ (FEM) - λ (0D)| on the one element
LMBDA_ATOL = 1e-10
#: Bound on the coupled loop's differences, relative to each quantity's largest value
RTOL = 1e-9


def _homogeneous_stress() -> Callable[..., list[float]]:
    """``(λ1, λ2, p, Ta [Pa]) -> [P11, P22, P33]`` in Pa, at ``F = diag(λ1, λ2, 1/(λ1 λ2))``.

    The material of ``conftest._mechanics`` (Holzapfel-Ogden, transversely isotropic,
    incompressible, fibres along x and sheets along y) and simcardemsx's default active
    stress, ``Ta F f0⊗f0 / λ_f`` (the ``stretch`` formulation), all from zero-mech, whose
    ``StretchActiveStress`` writes it as the strain energy ``Ta (λ_f - 1)``.
    """
    model = zero_mech.Model(
        material=zero_mech.material.HolzapfelOgden(
            use_heaviside=True,
            f0=sp.Matrix([1, 0, 0]),
            s0=sp.Matrix([0, 1, 0]),
        ),
        compressibility=zero_mech.compressibility.Incompressible(),
        active=zero_mech.StretchActiveStress(f0=sp.Matrix([1, 0, 0])),
    )
    parameters = {
        model[name]: float(cast(pulse.Variable, value).to_base_units())
        for name, value in pulse.HolzapfelOgden.transversely_isotropic_parameters().items()
    }
    l1, l2 = sp.symbols("λ1 λ2", positive=True)
    experiment = zero_mech.experiments.MechanicalExperiment(F11=l1, F22=l2, F33=1 / (l1 * l2))
    P = model.first_piola_kirchhoff(experiment.F).xreplace(parameters)
    return sp.lambdify((l1, l2, model["p"], model["Ta"]), [P[i, i] for i in range(3)], "numpy")


class _Homogeneous0D:
    """A split's ``mechanics`` component (numpy, GRL) on a homogeneous element, monolithic.

    Each step finds ``(λ1, λ2, p)`` at which the stress vanishes, with the component
    advanced from the last accepted state at the trial ``λ1`` and the stretch rate
    ``(λ1 - λ_prev) / dt``, then accepts the states of the solution.
    """

    def __init__(self, mech: ModuleType, stress: Callable[..., list[float]]) -> None:
        self.mech = mech
        self.stress = stress
        self.states = mech.init_state_values()
        self.parameters = mech.init_parameter_values()
        self.i_lmbda = mech.parameter_index("lmbda")
        self.i_dlambda = mech.parameter_index("dLambda")
        self.i_Ta = mech.monitor_index("Ta")
        self.x = np.array([1.0, 1.0, 0.0])
        self.lmbda = 1.0
        self.Ta = 0.0
        self.missing = np.zeros(len(mech.missing))
        self.provided = np.zeros(0)

    def _advance(self, lmbda: float, t_n: float, dt: float):
        parameters = self.parameters.copy()
        parameters[self.i_lmbda] = lmbda
        parameters[self.i_dlambda] = (lmbda - self.lmbda) / dt
        states = self.mech.generalized_rush_larsen(
            self.states.copy(),
            t_n,
            dt,
            parameters,
            self.missing,
        )
        Ta = self.mech.monitor_values(t_n, states, parameters, self.missing)[self.i_Ta]
        return states, parameters, Ta

    def _residual(self, x: np.ndarray, t_n: float, dt: float) -> np.ndarray:
        _, _, Ta = self._advance(x[0], t_n, dt)
        return np.asarray(self.stress(x[0], x[1], x[2], PA_PER_KPA * Ta), dtype=float)

    def step(self, t_n: float, dt: float, missing: np.ndarray) -> float:
        self.missing = np.asarray(missing, dtype=float)
        solution = scipy.optimize.root(
            self._residual,
            self.x,
            args=(t_n, dt),
            method="hybr",
            options={"xtol": 1e-13},
        )
        # Judged by the residual: hybr can stop on its step size at a converged point
        residual = np.max(np.abs(self._residual(solution.x, t_n, dt)))
        assert residual < RESIDUAL_ATOL, f"t = {t_n}: {solution.message} ({residual:.2e} Pa)"
        self.x = solution.x
        # The states of the solution, never of a trial evaluation
        self.states, parameters, self.Ta = self._advance(self.x[0], t_n, dt)
        self.lmbda = float(self.x[0])
        self.provided = self.mech.missing_values(t_n, self.states, parameters, self.missing)
        return self.lmbda


@pytest.fixture(scope="module")
def homogeneous_stress():
    return _homogeneous_stress()


@pytest.mark.parametrize(
    "split, inputs_of_t, t_end",
    [("caisplit", _caisplit_inputs, 40.0), ("zetasplit", _zetasplit_inputs, 80.0)],
)
def test_one_element_equals_the_zero_mech_solution(
    split_modules,
    homogeneous_stress,
    split,
    inputs_of_t,
    t_end,
):
    dt = 1.0
    fem = _run(split_modules, _mechanics, split, "monolithic", dt, t_end, inputs_of_t)
    # Every step converged. Not t_fail == t_end, which a failure on the last step also gives.
    assert len(fem.trace) == round(t_end / dt)

    mech = _numpy_mech(split)
    zero_d = _Homogeneous0D(mech, homogeneous_stress)
    reference = []
    for n in range(round(t_end / dt)):
        missing = np.zeros(len(mech.missing))
        for name, value in inputs_of_t((n + 1) * dt).items():
            missing[mech.missing[name]] = value
        reference.append(zero_d.step(n * dt, dt, missing))

    assert min(reference) < 0.96  # a contraction, not the rest state
    assert np.max(np.abs(fem.trace - np.array(reference))) < LMBDA_ATOL


def _by_name(values: np.ndarray, indices: Mapping[str, int], into: Mapping[str, int]) -> np.ndarray:
    """``values``, laid out by ``indices``, rearranged by the same names into ``into``'s."""
    out = np.zeros(len(into))
    for name, index in into.items():
        out[index] = values[indices[name]]
    return out


@pytest.mark.parametrize("split", ["zetasplit", "caisplit"])
def test_coupled_loop_equals_its_zero_mech_twin(
    split_modules,
    make_ep_solver,
    homogeneous_stress,
    split,
):
    modules = split_modules[split]
    ep = modules.ep
    i_v, i_cai = ep.state_index("v"), ep.state_index("cai")

    ep_solver = make_ep_solver(ep, dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 3, 3, 3))
    problem, backend = _mechanics(
        modules.mechanics,
        dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1),
    )
    controller = SimulationController(problem, ep_solver, backend, modules, DT_MECH, DT_EP)
    fem: dict[str, list[float]] = {"lmbda": [], "v": [], "cai": [], "Ta": []}
    spread = 0.0
    for _ in range(NUM_STEPS):
        controller.step()
        states = ep_solver.ode.values
        fem["lmbda"].append(float(np.mean(backend.lmbda_prev.x.array)))
        fem["v"].append(float(np.mean(states[i_v])))
        fem["cai"].append(float(np.mean(states[i_cai])))
        fem["Ta"].append(float(np.mean(backend.tension_kPa.x.array)))
        spread = max(spread, float(np.ptp(states[i_v])), float(np.ptp(backend.lmbda_prev.x.array)))
    assert spread < 1e-9  # uniform, so the comparison with 0D is exact

    # The twin. The crossings are paired by name, as transfer_plan.resolve pairs them.
    mech = _numpy_mech(split)
    zero_d = _Homogeneous0D(mech, homogeneous_stress)
    y = ep.init_state_values()
    p = ep.init_parameter_values()
    to_ep = np.zeros(len(ep.missing))  # beat's missing variables start at zero
    i_lmbda = ep.parameter.get("lmbda")
    step = getattr(ep, EP_SCHEME)
    reference: dict[str, list[float]] = {key: [] for key in fem}
    t = 0.0
    for _ in range(NUM_STEPS):
        t_n = t
        for i in range(round(DT_MECH / DT_EP)):  # the controller's own time arithmetic
            t_start, t = t, t_n + (i + 1) * DT_EP
            # beat's Strang step: half an ODE step, the PDE step (the identity on this
            # uniform field) and the closing half ODE step, at beat's own arguments.
            dt = t - t_start
            y = step(y, t_start, 0.5 * dt, p, to_ep)
            y = step(y, t_start + 0.5 * dt, 0.5 * dt, p, to_ep)
        forward = _by_name(ep.missing_values(t, y, p, to_ep), ep.provides, mech.missing)
        lmbda = zero_d.step(t_n, DT_MECH, forward)
        to_ep = _by_name(zero_d.provided, modules.mechanics.provides, ep.missing)
        if i_lmbda is not None:
            p[i_lmbda] = lmbda
        reference["lmbda"].append(lmbda)
        reference["v"].append(float(y[i_v]))
        reference["cai"].append(float(y[i_cai]))
        reference["Ta"].append(float(zero_d.Ta))

    assert min(reference["lmbda"]) < 0.99  # the beat contracts the element
    for key, values in reference.items():
        scale = np.max(np.abs(values))
        assert np.max(np.abs(np.array(fem[key]) - values)) <= RTOL * scale, key
