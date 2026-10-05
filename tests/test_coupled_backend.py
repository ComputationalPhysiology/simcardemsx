"""The contract the controller drives an activation backend through.

X1 is here: ``CrossbridgeSegregated`` satisfies the contract, ``resolve`` pairs it with
the Ca_i split's EP module and refuses the zeta split's, and its ``Transfer`` records
agree with its ``missing``/``provides``.
"""

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
from conftest import _f0
from test_crossbridge_coupling import _crossbridge_factory

from simcardemsx.backends import CrossbridgeSegregated, GeneratedActivation
from simcardemsx.backends.base import CoupledBackend
from simcardemsx.controller import SimulationController
from simcardemsx.ode_model import load_ode_modules
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


@pytest.mark.parametrize("kind", ["generated", "crossbridge"])
def test_controller_calls_begin_step_advance_post_solve_in_order(
    split_modules,
    make_ep_solver,
    make_mechanics,
    kind,
):
    """Each step is ``begin_step(t_n, dt)``, then the mechanics driver's ``advance(t_n,
    dt)``, then ``post_solve``, for both backends. ``advance`` is recorded through a
    driver, so a controller that solved before preparing the step would fail here."""
    modules = split_modules["caisplit"]
    ep_solver = make_ep_solver(
        modules.ep,
        dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 3, 3, 3),
    )
    problem, backend = make_mechanics(
        modules.mechanics,
        dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1),
        quadrature_degree=2,
        backend_factory=(
            _crossbridge_factory("Land2017", modules.mechanics) if kind == "crossbridge" else None
        ),
    )

    calls: list = []

    class _RecordingDriver:
        def __init__(self, problem):
            self.problem = problem

        def advance(self, t_n: float, dt: float) -> bool:
            calls.append(("advance", t_n, dt))
            return self.problem.solve()

    controller = SimulationController(
        _RecordingDriver(problem),
        ep_solver,
        backend,
        modules,
        1.0,
        0.05,
    )

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
        ("advance", 0.0, 1.0),
        "post_solve",
        ("begin_step", 1.0, 1.0),
        ("advance", 1.0, 1.0),
        "post_solve",
    ]


def test_crossbridge_is_a_coupled_backend():
    backend = _crossbridge()

    assert isinstance(backend, CoupledBackend)
    assert backend.missing == {"cai": 0}
    assert backend.provides == {"J_TRPN": 0}
    assert backend.inputs == {"cai": backend.cai}
    assert backend.outputs == {"J_TRPN": backend.J_TRPN, "lmbda": backend.lmbda_prev}
    assert backend.outputs["lmbda"] is backend.lmbda_prev
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


def test_crossbridge_sends_the_stretch_to_an_ep_remainder_that_takes_it(
    tmp_path,
    make_ep_solver,
    make_mechanics,
):
    """An EP remainder that keeps ``lmbda`` as a parameter, with the Ca_i split's
    crossings otherwise: the controller constructs (``outputs["lmbda"]`` exists) and
    λ(u) of the accepted step lands in EP's ``parameters``."""
    ode_file = tmp_path / "cai_with_stretch.ode"
    ode_file.write_text(
        """
    parameters("ep", lmbda=1.0)
    states("ep", v=0.0, cai=1e-4)

    states("mechanics", x=0.0)
    expressions("mechanics")
    dx_dt = cai - x
    J_TRPN = dx_dt

    expressions("ep")
    dv_dt = -v * lmbda
    dcai_dt = -J_TRPN
    """,
    )
    modules = load_ode_modules(ode_file, tmp_path)
    assert "lmbda" in modules.ep.parameter
    assert resolve(modules.ep, _crossbridge()).stretch_to_ep

    problem, backend = make_mechanics(
        modules.mechanics,
        dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1),
        quadrature_degree=2,
        backend_factory=lambda mesh, f0, degree: CrossbridgeSegregated(
            f0,
            mesh,
            "Land2017",
            quadrature_degree=degree,
        ),
    )
    ep_solver = make_ep_solver(
        modules.ep,
        dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 3, 3, 3),
    )
    controller = SimulationController(problem, ep_solver, backend, modules, 1.0, 0.05)

    # The element is stretched by 10% along its fibres; one accepted step at dt == 0
    # (the identity) makes that the backend's stretch, then it goes back to EP.
    problem.u.interpolate(
        lambda x: np.vstack([0.1 * x[0], np.zeros_like(x[1]), np.zeros_like(x[2])]),
    )
    backend.begin_step(0.0, 0.0)
    backend.post_solve()
    controller.plan.backward()

    row = modules.ep.parameter["lmbda"]
    np.testing.assert_allclose(ep_solver.ode.parameters[row], 1.1, rtol=1e-12)
