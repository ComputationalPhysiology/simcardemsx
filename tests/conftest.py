import json
import types
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, NamedTuple

from mpi4py import MPI

import beat
import crossbridge
import dolfinx
import gotranx
import numpy as np
import pulse
import pytest
import ufl

from simcardemsx.backends import CrossbridgeSegregated, GeneratedActivation
from simcardemsx.checkpoint import Checkpointable
from simcardemsx.controller import SimulationController
from simcardemsx.mechanics import MechanicsDriver
from simcardemsx.ode_model import ODEModules, load_ode_modules

ODEFILES_DIR = Path(__file__).parent.parent / "numerical_experiments" / "odefiles"

SPLITS = ("caisplit", "zetasplit", "catrpnsplit")

#: Absolute Newton tolerance for the one-element problems, tightened from pulse's
#: default 1e-6. A pure relative tolerance cannot be met at rest: the first
#: residual there is already ~1e-9–1e-8, at round-off, so the line search reports
#: failure even though the state is converged. 1e-9 is chosen far below the λ
#: errors the coupling gates measure (as small as ~1.5e-4); see
#: ``tests/test_monolithic_coupling.py`` for the gates themselves.
SNES_ATOL = 1e-9


def calcium(t: float) -> float:
    """Prescribed Ca_i transient in mM: 1e-4 at rest, peaking at 1e-3 at t = 25 ms."""
    tau = max(t - 5.0, 0.0)
    return 1e-4 + 9e-4 * (tau / 20.0) * np.exp(1.0 - tau / 20.0)


def _f0(mesh: dolfinx.mesh.Mesh) -> dolfinx.fem.Constant:
    """The fibre direction of the one-element tests: x, as a Constant."""
    return dolfinx.fem.Constant(mesh, np.array([1.0, 0.0, 0.0]))


#: Differences of λ smaller than this are round-off, not a change of direction.
_FLAT = 1e-12


def _twitch(t: float) -> float:
    """Unit twitch shape: 0 until t = 5 ms, peaking at 1 at t = 25 ms."""
    tau = max(t - 5.0, 0.0)
    return (tau / 20.0) * np.exp(1.0 - tau / 20.0)


def _zetasplit_inputs(t: float) -> dict[str, float]:
    b = _twitch(t)
    return {"XS": 0.01 * b, "XW": 0.005 * b}


def _reversals(trace: np.ndarray) -> int:
    """Changes of the direction of λ, ignoring differences below round-off."""
    steps = np.diff(trace)
    steps = steps[np.abs(steps) >= _FLAT]
    return int(np.count_nonzero(np.diff(np.sign(steps))))


@cache
def _numpy_mech(split: str) -> types.ModuleType:
    """The ``mechanics`` component of ``split``, generated as numpy (GRL) and loaded.

    Generated exactly as :func:`simcardemsx.ode_model.generate_ode_code` generates
    the UFL module -- same component, same scheme, same ``missing_values`` -- but
    with ``gotran2py``.
    """
    ode = gotranx.load_ode(ODEFILES_DIR / f"ToRORd_dynCl_endo_{split}.ode")
    mechanics_comp = ode.get_component("mechanics")
    ep_ode = ode - mechanics_comp
    code = gotranx.cli.gotran2py.get_code(
        mechanics_comp.to_ode(),
        scheme=[gotranx.schemes.Scheme.generalized_rush_larsen],
        missing_values=ep_ode.missing_variables,
    )
    module = types.ModuleType(f"numpy_mechanics_{split}")
    exec(code, module.__dict__)
    return module


def land2017_from_ode(mech_module: ModuleType) -> dict[str, float]:
    """crossbridge ``Land2017`` parameters for the ``.ode``'s Land, from ``mech_module``'s values.

    The mapping of sub-project 5's X4 probe: rates from per ms to per s (x1000); ``Tref``
    and ``a`` from kPa to Pa (x1000), since crossbridge's tension formula is in Pa;
    ``eta_l``/``eta_s`` from ms to s (/1000); ``ca50_ref`` is ``cat50_ref`` (uM in both);
    ``SL0`` is crossbridge's own 1.8 um. crossbridge's defaults differ (e.g. ``kuw`` 26/s
    against 182/s, ``Tref`` 40.5 against 120 kPa, ``ca50_ref`` 2.5 against 0.805 uM).
    """
    values = mech_module.init_parameter_values()
    P = {name: float(values[index]) for name, index in mech_module.parameter.items()}
    return dict(
        SL0=1.8,
        a=P["p_a"] * 1000,
        b=P["p_b"],
        k=P["p_k"],
        eta_l=P["etal"] / 1000,
        eta_s=P["etas"] / 1000,
        k_trpn=P["ktrpn"] * 1000,
        ntrpn=P["ntrpn"],
        ca50_ref=P["cat50_ref"],
        ku=P["ku"] * 1000,
        nTm=P["ntm"],
        trpn50=P["Trpn50"],
        kuw=P["kuw"] * 1000,
        kws=P["kws"] * 1000,
        rw=P["rw"],
        rs=P["rs"],
        gs=P["gammas"] * 1000,
        gw=P["gammaw"] * 1000,
        phi=P["phi"],
        Aeff=P["Tot_A"],
        beta0=P["Beta0"],
        beta1=P["Beta1"],
        Tref=P["Tref"] * 1000,
    )


#: crossbridge ``Land2017``'s state attributes, by the ``.ode``'s state names.
_LAND2017_STATES = {
    "CaTRPN": "CaTrpn",
    "B": "TmB",
    "S": "XS",
    "W": "XW",
    "Zs": "Zetas",
    "Zw": "Zetaw",
    "Cd": "Cd",
}


def match_land2017_initial_states(model: crossbridge.Land2017, mech_module: ModuleType) -> None:
    """Set ``model``'s states, at every point, to ``mech_module``'s initial ones.

    crossbridge's own ``reset()`` starts at B = 0 with CaTRPN at its rest value; the
    ``.ode`` starts at TmB = 1, CaTrpn = 1e-8 and the rest 0.
    """
    init = mech_module.init_state_values()
    for attribute, name in _LAND2017_STATES.items():
        getattr(model, attribute)[:] = init[mech_module.state[name]]


#: The reference sarcomere length [um] of the models that define no ``SL0``.
SL_REF = {"RDQ18": 2.0}


def _crossbridge_factory(model: str, mech: ModuleType, *, stabilized: bool = True):
    """A ``backend_factory`` for :func:`_mechanics`/:func:`_dynamic_mechanics`.

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


@pytest.fixture(scope="session")
def split_modules(tmp_path_factory) -> dict[str, ODEModules]:
    """Generate and load the EP/mechanics module pair for each of the three
    ODE splits shipped in `numerical_experiments/odefiles`.

    Session-scoped: code generation goes through gotranx (parsing + black
    formatting), which is slow, and every test that only reads module state
    (parameters/monitor/missing/provides) can share the result. Each split
    gets its own directory from `tmp_path_factory` so the three
    `ep_model.py`/`mechanics_model.py` files don't collide on disk.
    """
    modules = {}
    for split in SPLITS:
        odefile = ODEFILES_DIR / f"ToRORd_dynCl_endo_{split}.ode"
        output_dir = tmp_path_factory.mktemp(split)
        modules[split] = load_ode_modules(odefile, output_dir)
    return modules


def _ep_solver(
    ep_module: ModuleType,
    mesh: dolfinx.mesh.Mesh,
    ode_element: tuple[str, int] = ("P", 1),
) -> beat.MonodomainSplittingSolver:
    """beat's monodomain splitting solver for a generated EP module, on ``mesh``.

    The module's default parameters are used, so ToR-ORd's own cellular stimulus
    fires at t = 0 at every point. Parameters and states are per point, shape
    ``(n, num_points)``, so that ``lmbda`` can differ between points; the missing
    variables start at zero (``None`` when the EP side needs nothing).
    """
    V = dolfinx.fem.functionspace(mesh, ode_element)
    time = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))
    pde = beat.MonodomainModel(time=time, mesh=mesh, M=0.01)
    v_ode = dolfinx.fem.Function(V)
    num_points = v_ode.x.array.size
    states = np.tile(ep_module.init_state_values()[:, None], (1, num_points))
    parameters = np.tile(ep_module.init_parameter_values()[:, None], (1, num_points))
    missing = getattr(ep_module, "missing", {})
    ode = beat.odesolver.DolfinODESolver(
        v_ode=v_ode,
        v_pde=pde.state,
        init_states=states,
        parameters=parameters,
        fun=ep_module.generalized_rush_larsen,
        num_states=states.shape[0],
        v_index=ep_module.state_index("v"),
        missing_variables=np.zeros((len(missing), num_points)) if missing else None,
        num_missing_variables=len(missing),
    )
    return beat.MonodomainSplittingSolver(pde=pde, ode=ode, theta=1)


@pytest.fixture
def make_ep_solver():
    """Factory for :func:`_ep_solver`: ``make_ep_solver(ep_module, mesh, ode_element)``."""
    return _ep_solver


#: ``backend_factory(mesh, f0, quadrature_degree)``: the activation backend a mechanics
#: builder below uses in place of its ``GeneratedActivation``.
_BackendFactory = Callable[
    [dolfinx.mesh.Mesh, dolfinx.fem.Constant, int],
    pulse.active_model.ActiveModel,
]


def _rollers(mesh: dolfinx.mesh.Mesh):
    """Roller conditions: ``u_i = 0`` on the face ``x_i = 0``, for i = 0, 1, 2."""

    def dirichlet_bc(V):
        V0, _ = V.sub(0).collapse()
        zero = dolfinx.fem.Function(V0)
        fdim = mesh.topology.dim - 1
        bcs = []
        for i in range(mesh.geometry.dim):
            facets = dolfinx.mesh.locate_entities_boundary(
                mesh,
                fdim,
                lambda x, i=i: np.isclose(x[i], 0.0),
            )
            dofs = dolfinx.fem.locate_dofs_topological((V.sub(i), V0), fdim, facets)
            bcs.append(dolfinx.fem.dirichletbc(zero, dofs, V.sub(i)))
        return bcs

    return dirichlet_bc


def _mechanics(
    mech_module: ModuleType,
    mesh: dolfinx.mesh.Mesh,
    *,
    quadrature_degree: int = 2,
    backend_quadrature_degree: int | None = None,
    scheme: Literal["monolithic", "segregated", "stabilized"] = "monolithic",
    backend_factory: _BackendFactory | None = None,
) -> tuple[pulse.StaticProblem, pulse.active_model.ActiveModel]:
    """The one-element setup of ``tests/test_monolithic_coupling.py``.

    Holzapfel-Ogden (transversely isotropic), incompressible, rollers on the three
    faces through the origin, ``snes_atol`` = :data:`SNES_ATOL`. The backend's
    quadrature degree defaults to the geometry's.

    The backend is a :class:`GeneratedActivation` of ``mech_module`` with ``scheme``,
    or, with ``backend_factory``, ``backend_factory(mesh, f0, backend_quadrature_degree)``
    (``mech_module`` and ``scheme`` are then not read).
    """
    geometry = pulse.Geometry(mesh=mesh, metadata={"quadrature_degree": quadrature_degree})
    f0 = dolfinx.fem.Constant(mesh, np.array([1.0, 0.0, 0.0]))
    s0 = dolfinx.fem.Constant(mesh, np.array([0.0, 1.0, 0.0]))
    if backend_quadrature_degree is None:
        backend_quadrature_degree = quadrature_degree
    backend: pulse.active_model.ActiveModel
    if backend_factory is not None:
        backend = backend_factory(mesh, f0, backend_quadrature_degree)
    else:
        backend = GeneratedActivation(
            mech_module,
            mesh,
            f0,
            quadrature_degree=backend_quadrature_degree,
            scheme=scheme,
        )
    material = pulse.HolzapfelOgden(
        f0=f0,
        s0=s0,
        **pulse.HolzapfelOgden.transversely_isotropic_parameters(),
    )
    model = pulse.CardiacModel(
        material=material,
        active=backend,
        compressibility=pulse.Incompressible(),
    )
    petsc_options = pulse.StaticProblem.default_parameters()["petsc_options"]
    petsc_options["snes_atol"] = SNES_ATOL
    problem = pulse.StaticProblem(
        model=model,
        geometry=geometry,
        bcs=pulse.BoundaryConditions(dirichlet=[_rollers(mesh)]),
        parameters={"base_bc": pulse.problem.BaseBC.free, "petsc_options": petsc_options},
    )
    return problem, backend


@pytest.fixture
def make_mechanics():
    """Factory for :func:`_mechanics`: ``make_mechanics(mech_module, mesh, ...)``."""
    return _mechanics


#: Spread of λ over the element's quadrature points above which a run is unstable.
#: The load and the boundary conditions are uniform, so the solution is uniform: the
#: monolithic runs stay below 1e-15. At the smaller time steps the naive scheme's
#: unstable mode is spatial, amplified from round-off, and the mean of λ hides it.
_UNSTABLE_SPREAD = 1e-3


def _caisplit_inputs(t: float) -> dict[str, float]:
    return {"cai": calcium(t)}


class _Run(NamedTuple):
    trace: np.ndarray  # mean(lmbda_prev) after each converged step, at dt, 2 dt, ...
    spread: np.ndarray  # max - min of lmbda_prev after each converged step
    dt: float
    t_fail: float  # end of the first step that failed, or t_end if none did
    t_end: float

    @property
    def t_unstable(self) -> float:
        """End of the first step after which λ is no longer uniform, or ``t_fail``.

        Instability is measured at its onset rather than by the Newton failure
        alone: when Newton gives up on the naive scheme depends on the solver
        settings (line search, tolerances), when the mode appears does not.
        """
        (unstable,) = np.nonzero(self.spread > _UNSTABLE_SPREAD)
        if unstable.size:
            return min(float((unstable[0] + 1) * self.dt), self.t_fail)
        return self.t_fail


def _run(
    split_modules,
    make_mechanics,
    split: str,
    scheme: Literal["monolithic", "segregated", "stabilized"],
    dt: float,
    t_end: float,
    inputs_of_t: Callable[[float], dict[str, float]],
    *,
    backend_factory: _BackendFactory | None = None,
) -> _Run:
    """Contract one element under the given inputs: λ after each step, and when it failed.

    The loop of the one-element ``StaticProblem`` gates (gate 1, S1, gate 4 and X5).
    Records the mean of λ over the element's quadrature points and its spread
    (max - min).

    Each step writes the inputs at ``t_{n+1}``, calls ``backend.begin_step(t_n, dt)``
    (after the inputs, since a crossbridge backend advances its model with them
    there), solves, and accepts the step with ``post_solve``. The run stops at the
    first step whose Newton solve fails or leaves a non-finite displacement.

    The backend is ``make_mechanics``'s: a ``GeneratedActivation`` of ``split``'s
    mechanics module with ``scheme``, or, with ``backend_factory``, whatever that
    builds (``scheme`` is then not read).
    """
    _, mech = split_modules[split]
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    problem, backend = make_mechanics(mech, mesh, scheme=scheme, backend_factory=backend_factory)

    trace: list[float] = []
    spread: list[float] = []
    for n in range(round(t_end / dt)):
        t_n, t_next = n * dt, (n + 1) * dt
        for name, value in inputs_of_t(t_next).items():
            backend.inputs[name].x.array[:] = value
        backend.begin_step(t_n, dt)

        ok = problem.solve(raise_on_failure=False)
        with np.errstate(over="ignore", invalid="ignore"):
            finite = bool(np.all(np.isfinite(problem.u.x.array)))
        if not ok or not finite:
            return _Run(np.array(trace), np.array(spread), dt, t_next, t_end)

        backend.post_solve()
        lmbda = backend.lmbda_prev.x.array
        trace.append(float(np.mean(lmbda)))
        spread.append(float(lmbda.max() - lmbda.min()))
    return _Run(np.array(trace), np.array(spread), dt, t_end, t_end)


def _at_whole_ms(trace: np.ndarray, dt: float) -> np.ndarray:
    """The entries of a trace recorded at t = 1, 2, ... ms."""
    per_ms = round(1.0 / dt)
    return trace[per_ms - 1 :: per_ms]


def _lmbda_error(run: _Run, reference: _Run) -> float:
    """The error of gate 1, S1 and X5: max |mean λ - the reference's| at t = 1, 2, ... ms."""
    return float(
        np.max(
            np.abs(_at_whole_ms(run.trace, run.dt) - _at_whole_ms(reference.trace, reference.dt)),
        ),
    )


def _observed_orders(errors: Mapping[float, float]) -> list[float]:
    """Observed orders of convergence between successive time steps, coarsest first:
    ``log(e_i / e_{i+1}) / log(dt_i / dt_{i+1})`` for ``errors = {dt: e}``."""
    dts = sorted(errors, reverse=True)
    return [
        float(np.log(errors[coarse] / errors[fine]) / np.log(coarse / fine))
        for coarse, fine in zip(dts, dts[1:])
    ]


@pytest.fixture(scope="session")
def caisplit_reference(split_modules) -> _Run:
    """The reference of gate 1, S1 and X5: the Ca_i split, monolithic, at dt 0.01 ms for
    40 ms.

    Built with :func:`_mechanics` itself, since the ``make_mechanics`` fixture is
    function-scoped and this one is shared by the session: 4000 static solves, run once.
    """
    return _run(split_modules, _mechanics, "caisplit", "monolithic", 0.01, 40.0, _caisplit_inputs)


@dataclass
class _TrueUActive(pulse.DynamicProblem):
    """A ``DynamicProblem`` that adds a hand-built end-of-step active stress.

    This is physcardems's ``cavity.py:56-63`` (``ControlledCavityDynamicProblem``),
    without ``dev`` -- ``GeneratedActivation.S`` takes no such argument -- built
    independently of :attr:`~pulse.active_model.ActiveModel.evaluate_at_end_of_step`
    so it can serve as the reference the flag is checked against: ``true_u_active``
    (a ``GeneratedActivation`` in D1, a ``CrossbridgeSegregated`` in X6) is evaluated
    at the true end-of-step displacement ``self.u``, exactly what the flag makes
    ``DynamicProblem`` itself do, but by a completely separate code path.
    """

    true_u_active: pulse.active_model.ActiveModel | None = None

    def _material_form(self, u, v, p):
        forms = super()._material_form(u, v, p)
        if self.true_u_active is not None:
            F = ufl.grad(self.u) + ufl.Identity(3)
            C = F.T * F
            var_C = ufl.grad(self.u_test).T * F + F.T * ufl.grad(self.u_test)
            forms[0] += ufl.inner(self.true_u_active.S(C), 0.5 * var_C) * self.geometry.dx
        return forms


def _dynamic_mechanics(
    mech_module: ModuleType,
    *,
    dt_ms: float,
    reference: bool = False,
    end_of_step: bool = True,
    quadrature_degree: int = 2,
    scheme: Literal["monolithic", "segregated", "stabilized"] = "monolithic",
    backend_factory: _BackendFactory | None = None,
) -> tuple[pulse.DynamicProblem, pulse.active_model.ActiveModel]:
    """The pinned D1/D2 element: a one-element ``pulse.DynamicProblem``.

    A unit cube scaled to L = 0.01 m (``mesh_unit`` "m", ``DynamicProblem``'s
    default), rollers (:func:`_rollers`), Holzapfel-Ogden (transversely
    isotropic), ``pulse.Compressible()``, ``pulse.Viscous()`` (its default eta,
    100 Pa s), rho = 1e3 (``DynamicProblem``'s default), ``dt =
    Variable(dt_ms * 1e-3, "s")``, ``snes_atol`` = :data:`SNES_ATOL`.

    With ``reference=True``, ``model.active`` is ``pulse.active_model.Passive()``
    and the problem is :class:`_TrueUActive`: the backend's ``S`` is added by hand
    at the true end-of-step displacement instead of through the flag, giving the
    reference D1 checks the flag against. Because the backend is not
    ``model.active`` in that case, ``pulse.StaticProblem`` never calls
    ``backend.register``, so it is called here instead.

    With ``end_of_step=False``, ``backend.evaluate_at_end_of_step`` is set to
    ``False`` on the instance *before* the problem is built: ``pulse.DynamicProblem``
    reads the flag when it compiles the form, i.e. in construction, not at solve
    time. This is the alpha_f variant.

    The backend is a :class:`GeneratedActivation` of ``mech_module`` with ``scheme``,
    its coupling scheme, or, with ``backend_factory``, ``backend_factory(mesh, f0,
    quadrature_degree)`` (``mech_module`` and ``scheme`` are then not read). The
    reference and the alpha_f variant treat either the same way.
    """
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    mesh.geometry.x[:] *= 0.01
    geometry = pulse.Geometry(mesh=mesh, metadata={"quadrature_degree": quadrature_degree})
    f0 = dolfinx.fem.Constant(mesh, np.array([1.0, 0.0, 0.0]))
    s0 = dolfinx.fem.Constant(mesh, np.array([0.0, 1.0, 0.0]))

    backend: pulse.active_model.ActiveModel
    if backend_factory is not None:
        backend = backend_factory(mesh, f0, quadrature_degree)
    else:
        backend = GeneratedActivation(
            mech_module,
            mesh,
            f0,
            quadrature_degree=quadrature_degree,
            scheme=scheme,
        )
    if not end_of_step:
        backend.evaluate_at_end_of_step = False

    material = pulse.HolzapfelOgden(
        f0=f0,
        s0=s0,
        **pulse.HolzapfelOgden.transversely_isotropic_parameters(),
    )
    model = pulse.CardiacModel(
        material=material,
        active=pulse.active_model.Passive() if reference else backend,
        compressibility=pulse.Compressible(),
        viscoelasticity=pulse.Viscous(),
    )

    petsc_options = pulse.DynamicProblem.default_parameters()["petsc_options"]
    petsc_options["snes_atol"] = SNES_ATOL
    parameters = {
        "dt": pulse.Variable(dt_ms * 1e-3, "s"),
        "petsc_options": petsc_options,
    }
    bcs = pulse.BoundaryConditions(dirichlet=[_rollers(mesh)])

    problem: pulse.DynamicProblem
    if reference:
        problem = _TrueUActive(
            model=model,
            geometry=geometry,
            bcs=bcs,
            parameters=parameters,
            true_u_active=backend,
        )
        backend.register(problem.u)
    else:
        problem = pulse.DynamicProblem(
            model=model,
            geometry=geometry,
            bcs=bcs,
            parameters=parameters,
        )
    return problem, backend


@pytest.fixture
def make_dynamic_mechanics():
    """Factory for :func:`_dynamic_mechanics`: ``make_dynamic_mechanics(mech_module, ...)``."""
    return _dynamic_mechanics


#: The time step of D1 and X6 on :func:`_dynamic_mechanics`'s element, in ms.
D1_DT_MS = 1.0
#: The end time of D1 and X6, in ms.
D1_T_END = 60.0


def _run_dynamic_until_failure(
    make_dynamic_mechanics,
    mech_module,
    *,
    reference: bool,
    end_of_step: bool,
    scheme: Literal["monolithic", "segregated", "stabilized"] = "monolithic",
    dt_ms: float = D1_DT_MS,
    t_end: float = D1_T_END,
    inputs_of_t: Callable[[float], dict[str, float]] = _caisplit_inputs,
    backend_factory: _BackendFactory | None = None,
) -> tuple[np.ndarray, float | None]:
    """Drive the pinned D1/D2 element exactly as :func:`_run` drives the one-element
    ``StaticProblem`` gates: write the inputs at ``t_{n+1}``, ``begin_step(t_n, dt)``,
    solve, ``post_solve``, record ``mean(lmbda_prev)``. The run stops at the first step
    that leaves a non-finite displacement or fails to converge.

    Parameters
    ----------
    reference, end_of_step:
        Passed on to ``make_dynamic_mechanics``: the ``_TrueUActive`` reference, and
        whether the flag is set (``False`` is the alpha_f evaluation). The backend is
        stepped the same way in all three.
    scheme, backend_factory:
        Passed on to ``make_dynamic_mechanics``: the ``GeneratedActivation``'s coupling
        scheme, or the factory of another backend.
    dt_ms:
        The time step in ms, of both the backend and the problem.
    t_end:
        The end time in ms: ``round(t_end / dt_ms)`` steps, from ``t = 0``.
    inputs_of_t:
        The backend's inputs at a time in ms, as ``{name: value}``; each value is
        written into all of ``backend.inputs[name]`` at ``t_{n+1}``. The default is the
        Ca_i split's prescribed ``calcium``.

    Returns
    -------
    trace : np.ndarray
        ``mean(lmbda_prev)`` after each converged step.
    t_fail : float | None
        The end (ms) of the step that failed, or ``None`` if none did.
    """
    problem, backend = make_dynamic_mechanics(
        mech_module,
        dt_ms=dt_ms,
        reference=reference,
        end_of_step=end_of_step,
        scheme=scheme,
        backend_factory=backend_factory,
    )
    trace: list[float] = []
    for n in range(round(t_end / dt_ms)):
        t_n, t_next = n * dt_ms, (n + 1) * dt_ms
        for name, value in inputs_of_t(t_next).items():
            backend.inputs[name].x.array[:] = value
        backend.begin_step(t_n, dt_ms)

        ok = problem.solve(raise_on_failure=False)
        with np.errstate(over="ignore", invalid="ignore"):
            finite = bool(np.all(np.isfinite(problem.u.x.array)))
        if not ok or not finite:
            return np.array(trace), t_next

        backend.post_solve()
        trace.append(float(np.mean(backend.lmbda_prev.x.array)))
    return np.array(trace), None


def _run_dynamic(make_dynamic_mechanics, mech_module, **kwargs) -> np.ndarray:
    """:func:`_run_dynamic_until_failure`, returning the trace only.

    Raises
    ------
    AssertionError
        If a step leaves a non-finite displacement, or fails to converge.
    """
    trace, t_fail = _run_dynamic_until_failure(make_dynamic_mechanics, mech_module, **kwargs)
    if t_fail is not None:
        raise AssertionError(
            f"step {trace.size} (t={t_fail} ms) failed to converge or went non-finite",
        )
    return trace


#: Mechanics time step of :func:`_ellipsoid_ep_mechanics`'s ``DynamicProblem``, in ms.
ELLIPSOID_DT_MS = 2.0

#: The quadrature degree of :func:`lv_ellipsoid`'s fibre field, and so the only one
#: :func:`_ellipsoid_ep_mechanics` accepts for its mechanics measure and backend states.
ELLIPSOID_QUADRATURE_DEGREE = 2

#: The beat length :func:`_ellipsoid_ep_mechanics` sets the circuit's heart rate from,
#: in ms. A ``CirculationClock`` driving that circuit must use the same ``period``:
#: the circuit reads ``beat_phase`` against its own ``RR`` and chamber offsets.
ELLIPSOID_PERIOD_MS = 1000.0

#: Newton's iteration budget for :func:`_ellipsoid_ep_mechanics` with ``cycle=True``,
#: raised from pulse's default of 50. On this ellipsoid, pulse's own cycle gate
#: measured one isovolumic-relaxation step needing 72 full-Newton iterations after
#: the switch into a volume constraint. This is a solver budget, not a tolerance.
CYCLE_SNES_MAX_IT = 150


@pytest.fixture(scope="module")
def lv_ellipsoid(tmp_path_factory):
    """A coarse LV ellipsoid, in metres, with analytic fibres (-60/60 degrees) on
    quadrature at :data:`ELLIPSOID_QUADRATURE_DEGREE`.

    The parameters of pulse's own coupling test (``tests/test_circulation_coupling.py``,
    the ``geo`` fixture), except the fibre space's degree. The 6 there would not compile
    next to the backend's states at the geometry's degree: FFCx takes an integral's
    points from its quadrature-element coefficients and requires them all to agree, so
    the form fails with a ``ValueError`` (24 degree-6 points against the states' 4).
    At the geometry's own degree the fibres are computed at exactly the points the
    backend and the material read them at, so they are unit length there. An
    interpolated field is not: P1 fibres, unit at the nodes, are 0.32 to 0.95 long
    at those points on this mesh, whose fibre angle turns through 120 degrees across a
    wall about one element thick. The backend's λ then scales with that length, and
    Holzapfel-Ogden's fibre invariant with its square. Module-scoped, so every problem a
    test module builds on it shares one geometry.
    """
    import cardiac_geometries

    return cardiac_geometries.mesh.lv_ellipsoid(
        outdir=tmp_path_factory.mktemp("lv_ellipsoid"),
        create_fibers=True,
        fiber_space=f"Quadrature_{ELLIPSOID_QUADRATURE_DEGREE}",
        r_short_endo=0.025,
        r_short_epi=0.035,
        r_long_endo=0.09,
        r_long_epi=0.097,
        psize_ref=0.05,
        mu_apex_endo=-np.pi,
        mu_base_endo=-np.arccos(5 / 17),
        mu_apex_epi=-np.pi,
        mu_base_epi=-np.arccos(5 / 20),
        comm=MPI.COMM_WORLD,
        fiber_angle_epi=-60,
        fiber_angle_endo=60,
    )


def _ellipsoid_ep_mechanics(
    ep_module: ModuleType,
    mech_module: ModuleType,
    lv_ellipsoid,
    *,
    circulation: bool,
    cycle: bool = False,
    tension_scale: dolfinx.fem.Function | ufl.core.expr.Expr | None = None,
    mech_parameters: Mapping[str, float] | None = None,
    quadrature_degree: int = ELLIPSOID_QUADRATURE_DEGREE,
) -> tuple[beat.MonodomainSplittingSolver, pulse.DynamicProblem, GeneratedActivation]:
    """EP and a ``pulse.DynamicProblem`` on the same LV ellipsoid (:func:`lv_ellipsoid`).

    EP is :func:`_ep_solver` on the ellipsoid's mesh, so ToR-ORd's own cellular
    stimulus fires everywhere at t = 0. Mechanics: Holzapfel-Ogden (transversely
    isotropic), ``pulse.Compressible()``, ``pulse.Viscous()``, the base fixed,
    ``dt`` = :data:`ELLIPSOID_DT_MS`, ``snes_atol`` = :data:`SNES_ATOL`, with the
    backend's states on quadrature at the geometry's ``quadrature_degree``, and its
    parameters overridden by ``mech_parameters`` (``GeneratedActivation``'s
    ``parameters``). ``quadrature_degree`` must be :data:`ELLIPSOID_QUADRATURE_DEGREE`,
    the degree of the fixture's fibre field: a quadrature coefficient only has values at
    its own element's points.

    With ``circulation=True`` the ENDO cavity is Regazzoni's LV, closed by the rest of
    his circuit (``drop_components=("timing", "LV")``), at the heart rate of
    :data:`ELLIPSOID_PERIOD_MS`; ``beat_phase`` is ``problem.circulation_missing
    ["beat_phase"]``, which the caller sets. The circuit's ``V_LV`` starts at the
    unloaded cavity volume, so the constraint holds at t = 0 without an inflation
    step, and the difference from the published ``V_LV`` is moved into ``V_LA``, so
    the total blood volume is the published one.

    With ``cycle=True`` the ENDO cavity is instead controlled
    (``pulse.problem.CavityControl``), with no circulation, for a
    ``pulse.cycle.CycleController`` to switch between constraints; the caller builds
    and initializes that controller. Newton's budget is then
    :data:`CYCLE_SNES_MAX_IT` iterations. With neither, there is no cavity at all.

    Raises
    ------
    ValueError
        If ``quadrature_degree`` is not :data:`ELLIPSOID_QUADRATURE_DEGREE`, or if both
        ``circulation`` and ``cycle`` are ``True``: the ENDO cavity is either a chamber
        of the circuit or controlled by the cycle, not both.
    """
    if quadrature_degree != ELLIPSOID_QUADRATURE_DEGREE:
        raise ValueError(
            f"quadrature_degree must be {ELLIPSOID_QUADRATURE_DEGREE}, the degree of the "
            f"lv_ellipsoid fixture's quadrature fibres, got {quadrature_degree}: the "
            "fibres have values only at their own element's points.",
        )
    if circulation and cycle:
        raise ValueError(
            "circulation and cycle are exclusive: the ENDO cavity is either a chamber "
            "of the closed-loop circuit or controlled by the five-phase cycle.",
        )
    mesh = lv_ellipsoid.mesh
    ep_solver = _ep_solver(ep_module, mesh)

    geometry = pulse.HeartGeometry.from_cardiac_geometries(
        lv_ellipsoid,
        metadata={"quadrature_degree": quadrature_degree},
    )
    backend = GeneratedActivation(
        mech_module,
        mesh,
        lv_ellipsoid.f0,
        quadrature_degree=quadrature_degree,
        parameters=mech_parameters,
        tension_scale=tension_scale,
    )
    material = pulse.HolzapfelOgden(
        f0=lv_ellipsoid.f0,
        s0=lv_ellipsoid.s0,
        **pulse.HolzapfelOgden.transversely_isotropic_parameters(),
    )
    model = pulse.CardiacModel(
        material=material,
        active=backend,
        compressibility=pulse.Compressible(),
        viscoelasticity=pulse.Viscous(),
    )
    petsc_options = pulse.DynamicProblem.default_parameters()["petsc_options"]
    petsc_options["snes_atol"] = SNES_ATOL
    if cycle:
        petsc_options["snes_max_it"] = CYCLE_SNES_MAX_IT
    parameters = {
        "base_bc": pulse.problem.BaseBC.fixed,
        "dt": pulse.Variable(ELLIPSOID_DT_MS * 1e-3, "s"),
        "petsc_options": petsc_options,
    }

    if cycle:
        # A controlled cavity needs the mesh in metres, pulse's default mesh_unit.
        control = pulse.problem.CavityControl(mesh)
        problem = pulse.DynamicProblem(
            model=model,
            geometry=geometry,
            parameters=parameters,
            cavities=[pulse.problem.Cavity(marker="ENDO", control=control)],
        )
        return ep_solver, problem, backend

    if not circulation:
        problem = pulse.DynamicProblem(model=model, geometry=geometry, parameters=parameters)
        return ep_solver, problem, backend

    from circulation import base, regazzoni2020
    from pulse.circulation import ChamberCoupling, GotranxCirculation, mL

    # The heart rate goes in before flattening: flat_ode_parameters derives RR and the
    # chambers' activation offsets from it, which overriding the flat HR would not.
    nested = base.remove_units(regazzoni2020.Regazzoni2020.default_parameters())
    circuit = GotranxCirculation(
        regazzoni2020.ODE_FILE,
        parameters=regazzoni2020.flat_ode_parameters(
            nested | {"HR": 1000.0 / ELLIPSOID_PERIOD_MS},
        ),
        drop_components=("timing", "LV"),
    )
    problem = pulse.DynamicProblem(
        model=model,
        geometry=geometry,
        parameters=parameters,
        # No volume: the chamber coupling below points it at the circuit's V_LV
        # during construction.
        cavities=[pulse.problem.Cavity(marker="ENDO", volume=None)],
        circulation=circuit,
        chambers=[ChamberCoupling("ENDO", "V_LV", "p_LV")],
        circulation_missing={
            "beat_phase": dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0)),
        },
    )

    unloaded = mesh.comm.allreduce(geometry.volume("ENDO"), op=MPI.SUM) / mL
    initial = np.asarray(circuit.initial_states, dtype=np.float64)
    i_LV, i_LA = circuit.state_index("V_LV"), circuit.state_index("V_LA")
    V_LA = initial[i_LA] + (initial[i_LV] - unloaded)
    for states in (
        problem.circulation_states,
        problem.circulation_states_old,
        problem.circulation_states_prev,
    ):
        states[i_LV].x.array[:] = unloaded
        states[i_LA].x.array[:] = V_LA
    return ep_solver, problem, backend


@dataclass
class _FailOnce:
    """A mechanics driver that fails one step after moving the mechanics.

    It delegates to ``driver``, except on its ``at_step``-th call to ``advance``
    (counting from 1): that call runs the inner ``advance``, so ``u`` has moved, and
    then returns ``False``. It fails once; the retry of that step delegates. Wrap a
    plain problem as ``_FailOnce(simcardemsx.mechanics.Solve(problem), at_step)``.
    """

    driver: MechanicsDriver
    at_step: int
    calls: int = 0

    @property
    def problem(self) -> pulse.StaticProblem:
        """The inner driver's problem."""
        return self.driver.problem

    def advance(self, t_n: float, dt: float) -> bool:
        self.calls += 1
        converged = self.driver.advance(t_n, dt)
        return converged and self.calls != self.at_step


#: What :func:`_coupled_state` returns: the arrays by ``"{namespace}/{name}"``, and the
#: metadata by namespace.
_CoupledState = tuple[dict[str, np.ndarray], dict[str, Any]]


def _coupled_state(
    controller: SimulationController,
    extra: Sequence[Checkpointable] = (),
) -> _CoupledState:
    """Copies of everything a restore must reproduce, of ``controller.components()``
    followed by ``extra``.

    The arrays are each component's restart Functions, as ``"{namespace}/{name}"``, and
    EP's whole ``parameters`` and ``missing_variables`` arrays (``"ep/parameters"``,
    ``"ep/missing_variables"``). The rows of those that come from the backend are the
    transfer plan's restart Functions (``"transfer/..."``); the whole arrays are added
    so that the other rows are checked too. The metadata is by namespace,
    round-tripped through JSON, as a checkpoint stores it.
    """
    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {}
    for component in [*controller.components(), *extra]:
        for name, f in component.restart_functions():
            key = f"{component.namespace}/{name}"
            assert key not in arrays, key
            arrays[key] = f.x.array.copy()
        metadata[component.namespace] = json.loads(json.dumps(component.restart_metadata()))
    ode = controller.ep_solver.ode
    assert isinstance(ode, beat.odesolver.DolfinODESolver)
    arrays["ep/parameters"] = ode.parameters.copy()
    if ode.missing_variables is not None:
        arrays["ep/missing_variables"] = ode.missing_variables.copy()
    return arrays, metadata


def _assert_same_state(a: _CoupledState, b: _CoupledState) -> None:
    """Two states from :func:`_coupled_state` are equal: the same keys, each array bit
    for bit, and the same metadata."""
    arrays_a, metadata_a = a
    arrays_b, metadata_b = b
    assert sorted(arrays_a) == sorted(arrays_b)
    for key, value in arrays_a.items():
        assert np.array_equal(value, arrays_b[key]), key
    assert metadata_a == metadata_b
