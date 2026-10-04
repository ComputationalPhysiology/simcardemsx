"""Gate X4: crossbridge's Land2017 and the ``.ode``'s Land agree, to first order in dt.

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
"""

from mpi4py import MPI

import crossbridge
import dolfinx
import numpy as np
import pytest
from conftest import _numpy_mech, calcium, land2017_from_ode, match_land2017_initial_states

from simcardemsx.backends import CrossbridgeSegregated
from simcardemsx.controller import SimulationController

T_END = 60.0
DTS = (1.0, 0.5, 0.25)
#: crossbridge's total troponin concentration [mM], the backend's default.
TRPNMAX = 0.07


def _orders(values: dict[float, float]) -> tuple[float, float]:
    return (
        float(np.log2(values[1.0] / values[0.5])),
        float(np.log2(values[0.5] / values[0.25])),
    )


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
    peak = 0.0
    for dt in DTS:
        Ta_ref, J_ref = _run_ode_land(ref, dt)
        Ta_cb, J_cb = _run_crossbridge_land(ref, dt)
        every = round(1 / dt)
        whole = slice(every - 1, None, every)
        dTa[dt] = float(np.max(np.abs(Ta_ref[whole] - Ta_cb[whole])))
        dJ[dt] = float(np.max(np.abs(J_ref[whole] - J_cb[whole])))
        peak = Ta_ref.max()
    print("isometric dTa", dTa, "dJ", dJ, "orders", _orders(dTa), _orders(dJ))

    assert min(_orders(dTa)) >= 0.8
    assert min(_orders(dJ)) >= 0.8
    assert dTa[0.25] / peak < 0.02


def _controller_run(split_modules, make_ep_solver, make_mechanics, dt_mech: float, kind: str):
    """Mean tension [kPa], EP's mean cai and mean J_TRPN at whole ms, for one backend."""
    modules = split_modules["caisplit"]
    ep, mech = modules
    ep_solver = make_ep_solver(ep, dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 3, 3, 3))
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)

    def factory(mesh, f0, quadrature_degree):
        backend = CrossbridgeSegregated(
            f0,
            mesh,
            "Land2017",
            quadrature_degree=quadrature_degree,
            params=land2017_from_ode(mech),
        )
        match_land2017_initial_states(backend.model, mech)
        return backend

    problem, backend = make_mechanics(
        mech,
        mesh,
        quadrature_degree=2,
        scheme="stabilized",
        backend_factory=factory if kind == "crossbridge" else None,
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
        print(name, diffs[name], _orders(diffs[name]))
    for name in names:
        assert min(_orders(diffs[name])) >= 0.8, (name, diffs[name])
