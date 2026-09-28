"""Tests for :func:`simcardemsx.transfer_plan.resolve`.

`resolve` is the pure half of the transfer plan: given the two modules
generated from one gotranx `.ode` file, it says what crosses between EP and
activation without touching meshes or function spaces. `test_shipped_splits_resolve`
checks it against the table in `.scratch/monolithic-activation/spec.md` (S2) for
all three splits shipped in `numerical_experiments/odefiles`;
`test_a_name_nothing_produces_raises` checks the guard against modules that
don't actually come from the same file.

:class:`simcardemsx.transfer_plan.TransferPlan` adds the spaces and operators on
top: its construction guards, and each direction checked on its own with the
value set at the source and read where the other side will use it. That the two
directions work together through the controller is gate 5
(``tests/test_round_trip.py``).
"""

import types

from mpi4py import MPI

import beat
import dolfinx
import numpy as np
import pytest
import ufl

from simcardemsx.backends import GeneratedActivation
from simcardemsx.transfer_plan import Crossings, TransferPlan, resolve


@pytest.mark.parametrize(
    "split, forward, backward, stretch",
    [
        ("caisplit", ("cai",), ("J_TRPN",), False),
        ("zetasplit", ("XS", "XW"), ("Zetas", "Zetaw"), True),
        ("catrpnsplit", ("CaTrpn",), (), True),
    ],
)
def test_shipped_splits_resolve(split_modules, split, forward, backward, stretch):
    ep, mech = split_modules[split]
    assert resolve(ep, mech) == Crossings(forward, backward, stretch)


def test_a_name_nothing_produces_raises():
    ep = types.SimpleNamespace(missing={"Zetas": 0}, provides={"cai": 0}, parameter={})
    mech = types.SimpleNamespace(missing={"cai": 0}, provides={}, parameter={})
    with pytest.raises(ValueError, match="Zetas"):
        resolve(ep, mech)


# ---------------------------------------------------------------------------
# TransferPlan: the spaces and operators on top of resolve()
# ---------------------------------------------------------------------------


def _unit_cube(n: int) -> dolfinx.mesh.Mesh:
    return dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n)


def _backend(mech, mesh) -> GeneratedActivation:
    f0 = dolfinx.fem.Constant(mesh, np.array([1.0, 0.0, 0.0]))
    return GeneratedActivation(mech, mesh, f0, quadrature_degree=2)


def _ode_with(ep, mesh, parameters, missing_variables) -> beat.odesolver.DolfinODESolver:
    """A beat ODE solver on P1 with the given parameter and missing-variable arrays."""
    V = dolfinx.fem.functionspace(mesh, ("P", 1))
    return beat.odesolver.DolfinODESolver(
        v_ode=dolfinx.fem.Function(V),
        v_pde=dolfinx.fem.Function(V),
        init_states=ep.init_state_values(),
        parameters=parameters,
        fun=ep.generalized_rush_larsen,
        num_states=len(ep.init_state_values()),
        missing_variables=missing_variables,
        num_missing_variables=len(getattr(ep, "missing", {})),
    )


def test_unsupported_ep_ode_space_raises(split_modules, make_ep_solver):
    ep, mech = split_modules["zetasplit"]
    ode_on_dg1 = make_ep_solver(ep, _unit_cube(2), ("DG", 1)).ode
    with pytest.raises(NotImplementedError, match="DG1"):
        TransferPlan(resolve(ep, mech), ep, ode_on_dg1, _backend(mech, _unit_cube(1)))


def test_lmbda_going_back_needs_per_point_parameters(split_modules):
    """One parameter row for all points cannot hold a λ that varies between them."""
    ep, mech = split_modules["catrpnsplit"]
    mesh = _unit_cube(2)
    ode = _ode_with(ep, mesh, ep.init_parameter_values(), None)
    num_points = ode.num_points
    with pytest.raises(ValueError, match=rf"\({len(ep.init_parameter_values())}, {num_points}\)"):
        TransferPlan(resolve(ep, mech), ep, ode, _backend(mech, _unit_cube(1)))


def test_values_going_back_need_a_missing_variable_row_per_name(split_modules):
    ep, mech = split_modules["caisplit"]
    mesh = _unit_cube(2)
    num_points = dolfinx.fem.functionspace(mesh, ("P", 1)).dofmap.index_map.size_local
    ode = _ode_with(ep, mesh, ep.init_parameter_values(), np.zeros(num_points))
    with pytest.raises(ValueError, match=rf"\(1, {num_points}\)"):
        TransferPlan(resolve(ep, mech), ep, ode, _backend(mech, _unit_cube(1)))


def test_forward_interpolates_ep_values_into_the_backend_inputs(split_modules, make_ep_solver):
    """CaTrpn, linear in x on the EP mesh, arrives exactly at the backend's quadrature points.

    The CaTrpn split is the one whose EP side needs nothing back, so its generated
    ``missing_values`` takes no ``missing_variables`` argument.
    """
    ep, mech = split_modules["catrpnsplit"]
    ode = make_ep_solver(ep, _unit_cube(3)).ode
    assert ode.missing_variables is None
    V_ep = ode.v_ode.function_space
    x_ep = V_ep.tabulate_dof_coordinates()[:, 0]
    ode.values[ep.state_index("CaTrpn")] = 0.01 + 0.02 * x_ep

    mech_mesh = _unit_cube(1)
    backend = _backend(mech, mech_mesh)
    TransferPlan(resolve(ep, mech), ep, ode, backend).forward(0.0)

    x_q = dolfinx.fem.Function(backend.space)
    x_q.interpolate(
        dolfinx.fem.Expression(
            ufl.SpatialCoordinate(mech_mesh)[0],
            backend.space.element.interpolation_points,
        ),
    )
    assert np.allclose(backend.inputs["CaTrpn"].x.array, 0.01 + 0.02 * x_q.x.array)


@pytest.mark.parametrize("ode_element", [("P", 1), ("DG", 0)])
def test_backward_writes_lmbda_into_the_arrays_beat_holds(
    split_modules,
    make_ep_solver,
    ode_element,
):
    """λ lands in EP's own parameter array, in place: beat captured it at construction."""
    ep, mech = split_modules["catrpnsplit"]
    ode = make_ep_solver(ep, _unit_cube(3), ode_element).ode
    parameters = ode.parameters
    before = parameters.copy()
    backend = _backend(mech, _unit_cube(1))
    plan = TransferPlan(resolve(ep, mech), ep, ode, backend)

    backend.outputs["lmbda"].x.array[:] = 0.9
    plan.backward()

    assert ode.parameters is parameters
    row = ep.parameter["lmbda"]
    assert np.allclose(parameters[row], 0.9)
    others = np.arange(len(parameters)) != row
    assert np.array_equal(parameters[others], before[others])
