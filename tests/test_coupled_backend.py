"""The contract the controller drives an activation backend through.

X1 is here: ``CrossbridgeSegregated`` satisfies the contract, ``resolve`` pairs it with
the Ca_i split's EP module and refuses the zeta split's, and its ``Transfer`` records
agree with its ``missing``/``provides``.
"""

from mpi4py import MPI

import dolfinx
import pytest
from conftest import _f0

from simcardemsx.backends import CrossbridgeSegregated, GeneratedActivation
from simcardemsx.backends.base import CoupledBackend
from simcardemsx.controller import SimulationController
from simcardemsx.transfer_plan import resolve

SPLITS = ["zetasplit", "caisplit"]


def _generated(modules, mesh):
    return GeneratedActivation(modules.mechanics, mesh, _f0(mesh), quadrature_degree=2)


def _crossbridge() -> CrossbridgeSegregated:
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    return CrossbridgeSegregated(_f0(mesh), mesh, "Land2017", quadrature_degree=2)


@pytest.mark.parametrize("split", SPLITS)
def test_generated_activation_is_a_coupled_backend(split_modules, split):
    modules = split_modules[split]
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    backend = _generated(modules, mesh)

    assert backend.missing == modules.mechanics.missing
    assert backend.provides == modules.mechanics.provides
    assert isinstance(backend, CoupledBackend)

    backend.begin_step(3.0, 0.5)
    assert float(backend.t.value) == 3.0
    assert float(backend.dt.value) == 0.5


@pytest.mark.parametrize("split", SPLITS)
def test_resolve_against_the_backend_equals_against_its_module(split_modules, split):
    modules = split_modules[split]
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    backend = _generated(modules, mesh)
    assert resolve(modules.ep, backend) == resolve(modules.ep, backend.module)


def test_controller_calls_begin_step_then_post_solve(
    split_modules,
    make_ep_solver,
    make_mechanics,
):
    modules = split_modules["zetasplit"]
    ep_solver = make_ep_solver(
        modules.ep,
        dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 3, 3, 3),
    )
    problem, backend = make_mechanics(
        modules.mechanics,
        dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1),
        quadrature_degree=2,
    )
    controller = SimulationController(problem, ep_solver, backend, modules, 1.0, 0.05)

    calls: list = []
    begin, post = backend.begin_step, backend.post_solve

    def record_begin(t_n, dt):
        calls.append(("begin_step", float(t_n), float(dt)))
        begin(t_n, dt)

    def record_post():
        calls.append("post_solve")
        post()

    backend.begin_step = record_begin
    backend.post_solve = record_post
    controller.step()
    controller.step()

    assert calls == [
        ("begin_step", 0.0, 1.0),
        "post_solve",
        ("begin_step", 1.0, 1.0),
        "post_solve",
    ]


def test_crossbridge_is_a_coupled_backend():
    backend = _crossbridge()

    assert isinstance(backend, CoupledBackend)
    assert backend.missing == {"cai": 0}
    assert backend.provides == {"J_TRPN": 0}
    assert backend.inputs == {"cai": backend.cai}
    assert backend.outputs == {"J_TRPN": backend.J_TRPN}
    assert backend.quadrature_degree == 2
    for function in (*backend.inputs.values(), *backend.outputs.values()):
        assert function.function_space is backend.space


def test_crossbridge_resolves_against_the_calcium_split_only(split_modules):
    """X1. Ca_i in, J_TRPN back. The zeta split's EP module needs ``Zetas``/``Zetaw``,
    which crossbridge does not give, and gives ``XS``/``XW`` rather than ``cai``."""
    backend = _crossbridge()

    crossings = resolve(split_modules["caisplit"].ep, backend)
    assert crossings.forward == ("cai",)
    assert crossings.backward == ("J_TRPN",)
    assert not crossings.stretch_to_ep

    with pytest.raises(ValueError, match="Zetas") as error:
        resolve(split_modules["zetasplit"].ep, backend)
    assert "Zetaw" in str(error.value)


def test_crossbridge_transfer_records_agree_with_its_dicts():
    """X1. The ``Transfer`` records document the units of what ``missing``/``provides``
    name; they must name the same variables, in the same order."""
    backend = _crossbridge()

    assert [t.name for t in backend.wants_from_ep()] == sorted(
        backend.missing,
        key=backend.missing.__getitem__,
    )
    assert [t.name for t in backend.gives_to_ep()] == sorted(
        backend.provides,
        key=backend.provides.__getitem__,
    )
