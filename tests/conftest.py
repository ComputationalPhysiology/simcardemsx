from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Literal

from mpi4py import MPI

import beat
import dolfinx
import numpy as np
import pulse
import pytest
import ufl

from simcardemsx.backends import GeneratedActivation
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
    scheme: Literal["monolithic", "segregated"] = "monolithic",
) -> tuple[pulse.StaticProblem, GeneratedActivation]:
    """The one-element setup of ``tests/test_monolithic_coupling.py``.

    Holzapfel-Ogden (transversely isotropic), incompressible, rollers on the three
    faces through the origin, ``snes_atol`` = :data:`SNES_ATOL`. The backend's
    quadrature degree defaults to the geometry's.
    """
    geometry = pulse.Geometry(mesh=mesh, metadata={"quadrature_degree": quadrature_degree})
    f0 = dolfinx.fem.Constant(mesh, np.array([1.0, 0.0, 0.0]))
    s0 = dolfinx.fem.Constant(mesh, np.array([0.0, 1.0, 0.0]))
    if backend_quadrature_degree is None:
        backend_quadrature_degree = quadrature_degree
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


@dataclass
class _TrueUActive(pulse.DynamicProblem):
    """A ``DynamicProblem`` that adds a hand-built end-of-step active stress.

    This is physcardems's ``cavity.py:56-63`` (``ControlledCavityDynamicProblem``),
    without ``dev`` -- ``GeneratedActivation.S`` takes no such argument -- built
    independently of :attr:`~pulse.active_model.ActiveModel.evaluate_at_end_of_step`
    so it can serve as the reference the flag is checked against: ``true_u_active``
    is evaluated at the true end-of-step displacement ``self.u``, exactly what the
    flag makes ``DynamicProblem`` itself do, but by a completely separate code path.
    """

    true_u_active: GeneratedActivation | None = None

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
) -> tuple[pulse.DynamicProblem, GeneratedActivation]:
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
    """
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    mesh.geometry.x[:] *= 0.01
    geometry = pulse.Geometry(mesh=mesh, metadata={"quadrature_degree": quadrature_degree})
    f0 = dolfinx.fem.Constant(mesh, np.array([1.0, 0.0, 0.0]))
    s0 = dolfinx.fem.Constant(mesh, np.array([0.0, 1.0, 0.0]))

    backend = GeneratedActivation(mech_module, mesh, f0, quadrature_degree=quadrature_degree)
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
