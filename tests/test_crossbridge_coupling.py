"""The coupling gates for crossbridge's contraction models, ``CrossbridgeSegregated``.

Gate X4: crossbridge's Land2017 and the ``.ode``'s Land agree, to first order in dt.
The two are different implementations of one model (crossbridge's own scheme against
gotranx's generalized Rush-Larsen on the ``.ode``), so they differ at O(dt), and the
gate is that the difference falls with dt rather than that it is small.
``land2017_from_ode`` maps the ``.ode``'s parameters onto crossbridge, and the initial
states are matched.

Measured maxima over whole ms, the first 60 ms, at dt 1 / 0.5 / 0.25 ms (orders in
brackets):

- isometric, prescribed calcium: max |dTa| 0.900 / 0.444 / 0.219 kPa (1.02, 1.02), 0.88 %
  of peak at dt 0.25; max |dJ_TRPN| 3.08e-4 / 1.56e-4 / 7.89e-5 mM/ms (0.98, 0.99);
- through the controller: mean tension 5.93e-2 / 2.20e-2 / 8.46e-3 kPa (1.43, 1.38); EP's
  mean cai 1.06e-5 / 5.18e-6 / 2.63e-6 mM (1.04, 0.98); EP's mean J_TRPN 2.51e-5 / 1.18e-5 /
  5.90e-6 mM/ms (1.09, 0.99).

Gate X2, gate 5 for every model in ``crossbridge.MODEL_REGISTRY``. Measured max |J_TRPN|
in EP after the first step / the last (mM/ms): RDQ18 5.45e-4 / 3.37e-6, Lewalle2024
4.11e-9 / 2.05e-5, Land2017 1.01e-4 / 7.75e-5, RDQ20MF 1.85e-3 / 1.80e-5. RDQ20MF needs
crossbridge 0.3.3: before it, ``advance_step`` took one explicit Euler RU step of the dt it
was handed, correct only at its own ``dt_RU`` of 0.025 ms, and at dt_mech 1 ms EP's cai
was negative at 2.1 ms and NaN from 3 ms.

Gate X5, gate 1 and S1 for crossbridge's Land2017. Measured: the naive scheme's onsets
35.0 / 17.5 / 13.4 ms at dt 1 / 0.25 / 0.05 (Newton failed at 35.0 / 22.0 / 17.15 ms);
the stabilized scheme's spread of λ at most 1.2e-15, and its max |mean λ - reference|
1.13e-2 / 3.43e-3 / 7.67e-4 (orders 0.86, 0.93).

Gate X6, D1 for crossbridge's Land2017. Measured: max |flagged - reference| in mean λ
0.0 (bitwise), max |flagged - alpha_f| 7.2e-3; mean λ falls to 0.899.
"""

from types import ModuleType

from mpi4py import MPI

import crossbridge
import dolfinx
import numpy as np
import pytest
from conftest import (
    _caisplit_inputs,
    _lmbda_error,
    _numpy_mech,
    _observed_orders,
    _run,
    _run_dynamic,
    calcium,
    land2017_from_ode,
    match_land2017_initial_states,
)

from simcardemsx.backends import CrossbridgeSegregated
from simcardemsx.controller import SimulationController

T_END = 60.0
DTS = (1.0, 0.5, 0.25)
#: crossbridge's total troponin concentration [mM], the backend's default.
TRPNMAX = 0.07

#: The reference sarcomere length [um] of the models that define no ``SL0``.
SL_REF = {"RDQ18": 2.0}


def _crossbridge_factory(model: str, mech: ModuleType, *, stabilized: bool = True):
    """A ``backend_factory`` for ``conftest._mechanics``/``_dynamic_mechanics``.

    It builds ``CrossbridgeSegregated`` of ``model`` on quadrature at the degree it is
    handed, the mechanics form's. Land2017 gets the ``.ode``'s parameters and initial
    states (``land2017_from_ode``, ``match_land2017_initial_states``); the other models
    their own defaults, with :data:`SL_REF` where they define no ``SL0``.
    """

    def factory(mesh, f0, quadrature_degree):
        land = model == "Land2017"
        backend = CrossbridgeSegregated(
            f0,
            mesh,
            model,
            quadrature_degree=quadrature_degree,
            SL_ref=SL_REF.get(model),
            params=land2017_from_ode(mech) if land else None,
            stabilized=stabilized,
        )
        if land:
            match_land2017_initial_states(backend.model, mech)
        return backend

    return factory


def _run_ode_land(ref, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Ta [kPa] and J_TRPN [mM/ms] per step of the ``.ode``'s Land at lambda = 1."""
    state = ref.init_state_values().copy()
    p = ref.init_parameter_values(lmbda=1.0, dLambda=0.0)
    Ta, J = [], []
    for n in range(round(T_END / dt)):
        m = np.array([calcium((n + 1) * dt)])
        state = ref.generalized_rush_larsen(state, n * dt, dt, p, m)
        monitor = ref.monitor_values(n * dt, state, p, m)
        Ta.append(monitor[ref.monitor["Ta"]])
        J.append(ref.missing_values(n * dt, state, p, m)[0])
    return np.array(Ta), np.array(J)


def _run_crossbridge_land(ref, dt: float) -> tuple[np.ndarray, np.ndarray]:
    model = crossbridge.Land2017(num_cells=1, params=land2017_from_ode(ref))
    match_land2017_initial_states(model, ref)
    Ta, J = [], []
    for n in range(round(T_END / dt)):
        model.advance_step(
            dt / 1000,
            calcium((n + 1) * dt) * 1000,
            np.array([1.8]),
            dSL_vals=np.array([0.0]),
        )
        Ta.append(model.get_active_tension()[0])
        J.append(model.get_calcium_binding_rate()[0] * TRPNMAX / 1000)
    return np.array(Ta), np.array(J)


def test_crossbridge_land_matches_the_ode_land_isometric():
    """The probe as a test: lambda = 1, prescribed Ca_i, no mesh.

    Measured max |dTa| at whole ms: see the assertion message of a failing run; the
    probe found 0.90, 0.44, 0.22 kPa (0.88 % of peak at dt 0.25).
    """
    ref = _numpy_mech("caisplit")
    dTa: dict[float, float] = {}
    dJ: dict[float, float] = {}
    peak: dict[float, float] = {}
    for dt in DTS:
        Ta_ref, J_ref = _run_ode_land(ref, dt)
        Ta_cb, J_cb = _run_crossbridge_land(ref, dt)
        every = round(1 / dt)
        whole = slice(every - 1, None, every)
        dTa[dt] = float(np.max(np.abs(Ta_ref[whole] - Ta_cb[whole])))
        dJ[dt] = float(np.max(np.abs(J_ref[whole] - J_cb[whole])))
        peak[dt] = float(Ta_ref.max())
    orders_Ta, orders_J = _observed_orders(dTa), _observed_orders(dJ)
    print("isometric dTa", dTa, "dJ", dJ, "orders", orders_Ta, orders_J)

    assert min(orders_Ta) >= 0.8, (dTa, orders_Ta)
    assert min(orders_J) >= 0.8, (dJ, orders_J)
    assert dTa[0.25] / peak[0.25] < 0.02, (dTa, peak)


def _controller_run(split_modules, make_ep_solver, make_mechanics, dt_mech: float, kind: str):
    """Mean tension [kPa], EP's mean cai and mean J_TRPN at whole ms, for one backend."""
    modules = split_modules["caisplit"]
    ep, mech = modules
    ep_solver = make_ep_solver(ep, dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 3, 3, 3))
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    problem, backend = make_mechanics(
        mech,
        mesh,
        quadrature_degree=2,
        scheme="stabilized",
        backend_factory=_crossbridge_factory("Land2017", mech) if kind == "crossbridge" else None,
    )
    controller = SimulationController(problem, ep_solver, backend, modules, dt_mech, 0.05)
    ode = ep_solver.ode
    cai = ep.state_index("cai")
    j_trpn = ep.missing["J_TRPN"]
    every = round(1 / dt_mech)
    Ta, cai_mean, J = [], [], []
    for n in range(1, round(T_END / dt_mech) + 1):
        controller.step()
        if n % every == 0:
            Ta.append(backend.tension_kPa.x.array.mean())
            cai_mean.append(ode.values[cai].mean())
            J.append(ode.missing_variables[j_trpn].mean())
    return np.array(Ta), np.array(cai_mean), np.array(J)


@pytest.mark.slow
def test_crossbridge_land_matches_the_generated_land_through_the_controller(
    split_modules,
    make_ep_solver,
    make_mechanics,
):
    """Both backends through their own ``SimulationController`` on the same inputs.

    The generated Land is ``GeneratedActivation(scheme="stabilized")``, the other
    ``CrossbridgeSegregated`` with matched parameters and initial states. EP is beat on
    the 3x3x3 cube with the Ca_i remainder; mechanics is the one-element StaticProblem.
    """
    names = ("tension_kPa", "cai", "J_TRPN")
    diffs: dict[str, dict[float, float]] = {n: {} for n in names}
    for dt in DTS:
        generated = _controller_run(split_modules, make_ep_solver, make_mechanics, dt, "generated")
        cross = _controller_run(split_modules, make_ep_solver, make_mechanics, dt, "crossbridge")
        for name, g, c in zip(names, generated, cross, strict=True):
            diffs[name][dt] = float(np.max(np.abs(g - c)))
    for name in names:
        print(name, diffs[name], _observed_orders(diffs[name]))
    for name in names:
        assert min(_observed_orders(diffs[name])) >= 0.8, (name, diffs[name])


#: X2's set-up, gate 5's: the EP and mechanics time steps [ms] and the number of steps.
DT_EP = 0.05
DT_MECH = 1.0
NUM_STEPS = 60


@pytest.mark.slow
@pytest.mark.parametrize("model", list(crossbridge.MODEL_REGISTRY))
def test_every_crossbridge_model_round_trips_through_the_controller(
    split_modules,
    make_ep_solver,
    make_mechanics,
    model,
):
    """X2, gate 5 for every crossbridge model: what it computes reaches EP.

    Gate 5's set-up: beat with the Ca_i remainder on the 3x3x3 cube, ToR-ORd firing on
    its own stimulus at t = 0, and the one-element ``StaticProblem``, coupled only
    through ``SimulationController``. Nothing is injected: the assertions read beat's
    own arrays. After the first step a second ``plan.backward()`` must leave them
    bit-identical, which holds only if ``post_solve`` ran before ``backward``; the
    step's ``J_TRPN`` is nonzero by then, so the check is not vacuous. All 60 steps must
    then complete, with the controller's clock at 60 ms, ``J_TRPN`` must have reached
    EP, and EP's states must still be finite.
    """
    modules = split_modules["caisplit"]
    ep, mech = modules
    ep_solver = make_ep_solver(ep, dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 3, 3, 3))
    problem, backend = make_mechanics(
        mech,
        dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1),
        quadrature_degree=2,
        backend_factory=_crossbridge_factory(model, mech),
    )
    controller = SimulationController(problem, ep_solver, backend, modules, DT_MECH, DT_EP)
    # The arrays beat captured at construction: what EP integrates with.
    ode = ep_solver.ode
    mv = ode.missing_variables
    parameters = ode.parameters
    j_trpn = ep.missing["J_TRPN"]

    ep_calls: list[float] = []
    mech_calls: list[tuple[float, int]] = []

    def step() -> None:
        controller.step(
            ep_callback=lambda t, i: ep_calls.append(t),
            mech_callback=lambda t, i, nit: mech_calls.append((t, i)),
        )

    step()
    first = (mv.copy(), parameters.copy())
    controller.plan.backward()
    np.testing.assert_array_equal(mv, first[0])
    np.testing.assert_array_equal(parameters, first[1])
    assert np.abs(first[0][j_trpn]).max() > 0.0

    for _ in range(NUM_STEPS - 1):
        step()
    print(model, "max |J_TRPN|", np.abs(first[0][j_trpn]).max(), np.abs(mv[j_trpn]).max())

    assert len(ep_calls) == NUM_STEPS * round(DT_MECH / DT_EP)
    assert mech_calls[-1] == pytest.approx((NUM_STEPS * DT_MECH, NUM_STEPS))
    assert controller.t == pytest.approx(NUM_STEPS * DT_MECH)
    assert np.abs(mv[j_trpn]).max() > 0.0
    # Not implied by the above: a model can return a finite J_TRPN from NaN calcium.
    assert np.all(np.isfinite(ode.values)), np.count_nonzero(~np.isfinite(ode.values))


#: X5's time steps [ms] and end time [ms]: gate 1's and S1's.
X5_DTS = (1.0, 0.25, 0.05)
X5_T_END = 40.0


@pytest.mark.slow
def test_crossbridge_stabilized_converges_and_naive_does_not(
    split_modules,
    make_mechanics,
    caisplit_reference,
):
    """X5, gate 1 and S1 for crossbridge: R&Q's result in FEM.

    Gate 1's regime, loop and onset criterion (``conftest._run``): the soft one-element
    ``StaticProblem``, quasistatic, with crossbridge's Land2017 at the ``.ode``'s
    parameters and initial states and the prescribed ``calcium`` at ``t_{n+1}``. The
    naive scheme (``stabilized=False``) must become unstable earlier as dt shrinks; the
    stabilized one must never trip the onset, and must converge at first order against
    gate 1's reference, the generated Land monolithic at dt 0.01 ms, by S1's error.
    """
    _, mech = split_modules["caisplit"]
    assert caisplit_reference.t_unstable == X5_T_END

    def run(stabilized: bool, dt: float):
        return _run(
            split_modules,
            make_mechanics,
            "caisplit",
            "stabilized" if stabilized else "segregated",
            dt,
            X5_T_END,
            _caisplit_inputs,
            backend_factory=_crossbridge_factory("Land2017", mech, stabilized=stabilized),
        )

    naive = {dt: run(False, dt) for dt in X5_DTS}
    onsets = [naive[dt].t_unstable for dt in X5_DTS]
    stabilized = {dt: run(True, dt) for dt in X5_DTS}
    print(
        "naive onsets",
        onsets,
        "t_fail",
        [naive[dt].t_fail for dt in X5_DTS],
        "stabilized max spread",
        [float(stabilized[dt].spread.max()) for dt in X5_DTS],
    )

    assert onsets[0] > onsets[1] > onsets[2], onsets
    for dt, r in stabilized.items():
        assert r.t_unstable == X5_T_END, (dt, r.t_fail, r.spread.max())

    e = {dt: _lmbda_error(r, caisplit_reference) for dt, r in stabilized.items()}
    orders = _observed_orders(e)
    print("stabilized errors", e, "orders", orders)
    assert orders[0] >= 0.8, (e, orders)
    assert orders[1] >= 0.8, (e, orders)


@pytest.mark.slow
def test_crossbridge_is_evaluated_at_the_end_of_the_step(split_modules, make_dynamic_mechanics):
    """X6, D1 for crossbridge: the flag reproduces the by-hand end-of-step reference.

    D1's element and loop (``conftest._dynamic_mechanics``, ``_run_dynamic``): the
    damped 1 cm cube under ``DynamicProblem``, dt 1 ms for 60 ms, the prescribed
    ``calcium``, crossbridge's Land2017 at the ``.ode``'s parameters and initial states.
    Three runs, the backend stepped the same way in each: flagged; the ``_TrueUActive``
    reference, with ``Passive()`` as ``model.active`` and the backend's ``S`` added by
    hand at the end-of-step displacement; and alpha_f, the flag unset on the instance
    before the problem is built. The flagged run must match the reference to round-off
    and differ from alpha_f by more than round-off (about 1e-15).
    """
    _, mech = split_modules["caisplit"]
    factory = _crossbridge_factory("Land2017", mech)

    flagged = _run_dynamic(
        make_dynamic_mechanics,
        mech,
        reference=False,
        end_of_step=True,
        backend_factory=factory,
    )
    reference = _run_dynamic(
        make_dynamic_mechanics,
        mech,
        reference=True,
        end_of_step=True,
        backend_factory=factory,
    )
    alpha_f = _run_dynamic(
        make_dynamic_mechanics,
        mech,
        reference=False,
        end_of_step=False,
        backend_factory=factory,
    )
    print(
        "max |flagged - reference|",
        np.max(np.abs(flagged - reference)),
        "max |flagged - alpha_f|",
        np.max(np.abs(flagged - alpha_f)),
        "min mean lambda",
        flagged.min(),
    )

    np.testing.assert_allclose(flagged, reference, rtol=1e-10, atol=0)
    assert np.max(np.abs(flagged - alpha_f)) > 1e-8
