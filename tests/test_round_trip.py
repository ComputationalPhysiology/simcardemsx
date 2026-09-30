"""Gate 5: what the mechanics side computes reaches EP, through the controller.

On the old coupling the mechanics -> EP path transferred zeros (the buffer it
interpolated from was read and never written) and EP's ``lmbda`` parameter was
never set, so length-dependent calcium sensitivity ran at λ = 1. A smoke test
passed throughout, because it injected the value it then asserted on.

Here nothing is injected: EP (real beat, ToR-ORd firing on its own stimulus at
t = 0) and mechanics (one element, :class:`GeneratedActivation`, monolithic) are
coupled only through :class:`SimulationController`, and the assertions read
beat's own arrays -- what EP will actually integrate with next.
"""

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest

from simcardemsx.controller import SimulationController

DT_EP = 0.05
DT_MECH = 1.0
NUM_STEPS = 60


@pytest.mark.slow
@pytest.mark.parametrize("scheme", ["monolithic", "stabilized"])
@pytest.mark.parametrize("split", ["zetasplit", "caisplit"])
def test_values_return_to_ep_through_the_controller(
    split_modules,
    make_ep_solver,
    make_mechanics,
    split,
    scheme,
):
    modules = split_modules[split]
    ep, mech = modules
    ep_solver = make_ep_solver(ep, dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 3, 3, 3))
    problem, backend = make_mechanics(
        mech,
        dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1),
        quadrature_degree=2,
        scheme=scheme,
    )
    controller = SimulationController(problem, ep_solver, backend, modules, DT_MECH, DT_EP)
    # The arrays beat captured at construction: what EP integrates with.
    ode = ep_solver.ode
    mv = ode.missing_variables
    parameters = ode.parameters

    ep_calls: list[float] = []
    mech_calls: list[tuple[float, int]] = []
    for _ in range(NUM_STEPS):
        controller.step(
            ep_callback=lambda t, i: ep_calls.append(t),
            mech_callback=lambda t, i, nit: mech_calls.append((t, i)),
        )

    assert len(ep_calls) == NUM_STEPS * round(DT_MECH / DT_EP)
    assert mech_calls[-1] == pytest.approx((NUM_STEPS * DT_MECH, NUM_STEPS))

    if split == "zetasplit":
        assert np.abs(mv[ep.missing["Zetas"]]).max() > 1e-8
        assert np.abs(mv[ep.missing["Zetaw"]]).max() > 1e-8
        assert parameters[ep.parameter["lmbda"]].min() < 1.0 - 1e-6
    else:
        assert np.abs(mv[ep.missing["J_TRPN"]]).max() > 0.0


@pytest.mark.parametrize("scheme", ["monolithic", "segregated", "stabilized"])
@pytest.mark.parametrize("split", ["zetasplit", "caisplit"])
def test_values_cross_back_after_the_step_is_accepted(
    split_modules,
    make_ep_solver,
    make_mechanics,
    split,
    scheme,
):
    """``step()`` accepts the step (``post_solve``) before moving values back to EP
    (``backward``), so EP is left holding the accepted step's outputs, and moving them
    again changes nothing. In the other order EP would get the previous step's
    outputs, one step late throughout: after the first step, the zeros the backend's
    outputs start from.
    """
    modules = split_modules[split]
    ep_solver = make_ep_solver(
        modules.ep,
        dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 3, 3, 3),
    )
    problem, backend = make_mechanics(
        modules.mechanics,
        dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1),
        quadrature_degree=2,
        scheme=scheme,
    )
    controller = SimulationController(problem, ep_solver, backend, modules, DT_MECH, DT_EP)
    ode = ep_solver.ode

    controller.step()
    missing_variables = ode.missing_variables.copy()
    parameters = ode.parameters.copy()
    controller.plan.backward()

    np.testing.assert_array_equal(ode.missing_variables, missing_variables)
    np.testing.assert_array_equal(ode.parameters, parameters)
