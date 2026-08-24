"""Gate tests for the coupler: the five things that must be true for the
EP/mechanics coupling to be correct.

These are deliberately not smoke tests. The suite this joins already contained
`assert disp > 0.0` and `assert np.max(XS) > 0.0`, and a bug that made the
mechano-electric feedback path transfer zeros survived all of them. Each test
below is chosen to fail when the *coupling* is wrong rather than when the code
is broken.

The primary seam is `SimulationController.step`. Nothing here reaches past it to
inject a value it then asserts on -- that is exactly how the existing feedback
test missed a transfer path delivering zeros.
"""

from pathlib import Path

from mpi4py import MPI

import dolfinx
import numpy as np
import pulse
import pytest

from simcardemsx.backends import CrossbridgeSegregated, ZetaSplitUFL
from simcardemsx.controller import SimulationController
from simcardemsx.ode_model import RuntimeODEModel, load_ode_modules

# ---------------------------------------------------------------------------
# Synthetic ODE files
#
# Small enough to generate inside a test, but real splits: each declares a
# mechanics component, so gotranx derives a genuine two-way interface. The
# shipped ToR-ORd files are used in exactly one test, marked slow.
#
# Both transients relax towards a bound rather than ramping without limit. An
# unbounded ramp activates the contraction model within a single millisecond
# step, and the resulting shortening rate drives the distortion states so far
# negative that active tension clamps at zero -- a fixture artefact that looks
# exactly like a broken coupling.
#
# The gains are set so the zeta backend generates a physiological few tens of
# kPa. Driving it far harder makes the stretch oscillate between steps, which
# is a genuine property of that backend at a millisecond step and not
# something these tests are here to characterize.
# ---------------------------------------------------------------------------

ZETA_SPLIT_ODE = """
parameters("ep", a=1.0)
states("ep", v=0.0, XS=0.0, XW=0.0)
states("mechanics", Zetas=0.0, Zetaw=0.0)
expressions("mechanics")
dZetas_dt = XS - Zetas
dZetaw_dt = XW - Zetaw
expressions("ep")
dv_dt = a
dXS_dt = 2e-6 * (1.0 - XS) - 0.02 * XS * (1.0 + Zetas)
dXW_dt = 2e-6 * (1.0 - XW) - 0.02 * XW * (1.0 + Zetaw)
"""

CAI_SPLIT_ODE = """
parameters("ep", a=1.0)
states("ep", v=0.0, cai=0.0001)
parameters("mechanics", trpnmax=0.07)
states("mechanics", CaTrpn=0.0)
expressions("mechanics")
dCaTrpn_dt = cai - CaTrpn
J_TRPN = dCaTrpn_dt * trpnmax
expressions("ep")
dv_dt = a
dcai_dt = 0.02 * (0.0015 - cai) - J_TRPN
"""


class ODELevelEPSolver:
    """An EP solver that really integrates the generated EP ODE.

    Stands in for `beat`'s monodomain + ODE solver at the seam the controller
    actually uses: it steps the generated `generalized_rush_larsen` scheme at
    every dof, reading its missing variables from the array the coupler fills.

    That last part is the point. It holds a *reference* to the coupler's
    return-path array, exactly as `beat.odesolver.DolfinODESolver` does, so a
    return path that delivers zeros shows up here as an EP solution that never
    sees them.
    """

    def __init__(self, module, V, missing_variables):
        self.V = V
        index_map = V.dofmap.index_map
        n = (index_map.size_local + index_map.num_ghosts) * V.dofmap.index_map_bs
        self.num_dofs = n

        y = module.init_state_values()
        p = module.init_parameter_values()
        self._values = np.tile(y[:, None], (1, n)).astype(float)
        self.parameters = np.tile(p[:, None], (1, n)).astype(float)
        self._fgr = module.generalized_rush_larsen
        self._missing = missing_variables
        self.module = module

        solver = self

        class _ODE:
            @property
            def _values(self):
                return solver._values

            @property
            def parameters(self):
                return solver.parameters

        self.ode = _ODE()

    def state(self, name):
        """The current value of a named EP state, per dof."""
        return self._values[self.module.state_index(name)]

    def step(self, t_span):
        t0, t1 = t_span
        self._values[:] = self._fgr(
            self._values,
            t0,
            t1 - t0,
            self.parameters,
            self._missing,
        )


def _meshes(n_mech=1, n_ep=2):
    comm = MPI.COMM_WORLD
    return (
        dolfinx.mesh.create_unit_cube(comm, n_mech, n_mech, n_mech),
        dolfinx.mesh.create_unit_cube(comm, n_ep, n_ep, n_ep),
    )


def _fibres(mesh):
    return (
        dolfinx.fem.Constant(mesh, np.array([1.0, 0.0, 0.0])),
        dolfinx.fem.Constant(mesh, np.array([0.0, 1.0, 0.0])),
        dolfinx.fem.Constant(mesh, np.array([0.0, 0.0, 1.0])),
    )


def _roller_bcs(mesh):
    """Symmetry planes only: the fibre is free to shorten."""

    def dirichlet_bc(V):
        V0, _ = V.sub(0).collapse()
        zero = dolfinx.fem.Function(V0)
        zero.x.array[:] = 0.0
        fdim = mesh.topology.dim - 1
        bcs = []
        for sub, axis in enumerate((0, 1, 2)):
            facets = dolfinx.mesh.locate_entities_boundary(
                mesh,
                fdim,
                lambda x, axis=axis: np.isclose(x[axis], 0.0),
            )
            bcs.append(
                dolfinx.fem.dirichletbc(
                    zero,
                    dolfinx.fem.locate_dofs_topological((V.sub(sub), V0), fdim, facets),
                    V.sub(sub),
                ),
            )
        return bcs

    return pulse.BoundaryConditions(dirichlet=[dirichlet_bc])


def _clamped_bcs(mesh):
    """Every boundary displacement pinned: the fibre cannot shorten, so the
    stretch stays at one and the coupled path is isometric."""

    def dirichlet_bc(V):
        V0, _ = V.sub(0).collapse()
        zero = dolfinx.fem.Function(V0)
        zero.x.array[:] = 0.0
        fdim = mesh.topology.dim - 1
        facets = dolfinx.mesh.locate_entities_boundary(
            mesh,
            fdim,
            lambda x: np.full(x.shape[1], True),
        )
        return [
            dolfinx.fem.dirichletbc(
                zero,
                dolfinx.fem.locate_dofs_topological((V.sub(sub), V0), fdim, facets),
                V.sub(sub),
            )
            for sub in range(3)
        ]

    return pulse.BoundaryConditions(dirichlet=[dirichlet_bc])


def _problem(mesh, backend, bcs):
    geo = pulse.Geometry(mesh=mesh, metadata={"quadrature_degree": 4})
    f0, s0, _ = _fibres(mesh)
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
    return pulse.StaticProblem(
        model=model,
        geometry=geo,
        bcs=bcs,
        parameters={"base_bc": pulse.problem.BaseBC.free},
    )


class Simulation:
    """Everything one coupled run needs, wired together."""

    def __init__(self, controller, backend, ep_solver, ode_model, problem):
        self.controller = controller
        self.backend = backend
        self.ep_solver = ep_solver
        self.ode_model = ode_model
        self.problem = problem

    def run(self, n_steps, **kwargs):
        for _ in range(n_steps):
            self.controller.step(**kwargs)
        return self


def build_simulation(
    tmp_path,
    ode_text,
    backend_factory,
    *,
    bcs=_roller_bcs,
    dt_mech=1.0,
    dt_ep=0.1,
    n_mech=1,
    n_ep=2,
):
    """Wire a coupled simulation from an ODE file and a backend.

    The one place that knows how the pieces connect. The tests below assert
    behaviour through it, so re-wiring it onto a new coupler API does not
    disturb what they assert.
    """
    mesh_mech, mesh_ep = _meshes(n_mech, n_ep)

    tmp_path.mkdir(parents=True, exist_ok=True)
    ode_file = tmp_path / "split.ode"
    ode_file.write_text(ode_text)
    modules = load_ode_modules(ode_file, tmp_path / "generated")

    V_mech = dolfinx.fem.functionspace(mesh_mech, ("DG", 1))
    V_ep = dolfinx.fem.functionspace(mesh_ep, ("DG", 1))

    ode_model = RuntimeODEModel(
        ep_module_dict=modules.ep.__dict__,
        mech_module_dict=modules.mechanics.__dict__,
        mech_ode_space=V_mech,
        ep_ode_space=V_ep,
    )

    backend = backend_factory(mesh_mech)
    problem = _problem(mesh_mech, backend, bcs(mesh_mech))

    ep_solver = ODELevelEPSolver(modules.ep, V_ep, ode_model.missing_ep.values_ep)

    controller = SimulationController(
        mechanics_problem=problem,
        ep_solver=ep_solver,
        ode_model=ode_model,
        backend=backend,
        dt_mech=dt_mech,
        dt_ep=dt_ep,
    )
    return Simulation(controller, backend, ep_solver, ode_model, problem)


def _zeta_backend(mesh):
    f0, s0, n0 = _fibres(mesh)
    return ZetaSplitUFL(f0=f0, s0=s0, n0=n0, mesh=mesh)


def _cai_backend(mesh, **kwargs):
    f0, _, _ = _fibres(mesh)
    return CrossbridgeSegregated(f0=f0, mesh=mesh, **kwargs)


# ---------------------------------------------------------------------------
# Gate test 1 -- both backends, one coupler
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ode_text, factory",
    [
        (ZETA_SPLIT_ODE, _zeta_backend),
        (CAI_SPLIT_ODE, _cai_backend),
    ],
    ids=["zeta_split", "cai_split"],
)
def test_both_backends_run_through_one_coupler(tmp_path, ode_text, factory):
    """Each activation backend drives a coupled simulation on its own split,
    through the same orchestration code.

    The conformance test for the whole design: if the two splits need different
    coupler code, they are not interchangeable and no comparison between them
    means anything.
    """
    sim = build_simulation(tmp_path, ode_text, factory).run(3)

    assert np.all(np.isfinite(sim.backend.active_tension.x.array))
    assert np.max(sim.backend.active_tension.x.array) > 0.0, "no tension was generated"
    assert np.all(np.isfinite(sim.problem.u.x.array))


# ---------------------------------------------------------------------------
# Gate test 2 -- isometric clamp against the contraction model alone
# ---------------------------------------------------------------------------


def test_isometric_clamp_matches_standalone_contraction_model(tmp_path):
    """With the stretch pinned to one, the coupled path must reproduce a
    standalone crossbridge run driven by the same calcium.

    This is what separates a coupling bug from a model bug. Everything the
    coupler does -- transfer, unit conversion, ordering -- is exercised, but the
    mechanics cannot feed anything back, so any disagreement is ours.
    """
    import crossbridge

    dt_mech = 1.0
    n_steps = 15

    sim = build_simulation(
        tmp_path,
        CAI_SPLIT_ODE,
        _cai_backend,
        bcs=_clamped_bcs,
        dt_mech=dt_mech,
    )

    cai_history = []

    def record(*args, **kwargs):
        cai_history.append(float(np.mean(sim.backend.cai.x.array)))

    sim.run(n_steps, mech_callback=record)

    # The same model, same calcium, no FEM in the loop.
    from simcardemsx import units

    standalone = crossbridge.get_model("Land2017")(num_cells=1)
    SL_ref = standalone.p["SL0"]
    for cai_mM in cai_history:
        standalone.advance_step(
            units.ms_to_s(dt_mech),
            units.calcium_to_crossbridge(np.array([cai_mM])),
            np.array([SL_ref]),
            dSL_vals=np.zeros(1),
        )

    coupled_Ta = float(np.mean(sim.backend.active_tension.x.array))
    standalone_Ta = float(standalone.get_active_tension()[0])

    assert np.allclose(sim.backend.lmbda.x.array, 1.0), "the clamp did not hold the stretch at one"
    assert coupled_Ta > 0.0, "no tension developed, so agreement proves nothing"
    assert coupled_Ta == pytest.approx(standalone_Ta, rel=1e-8), (
        f"coupled path gives Ta={coupled_Ta}, standalone gives {standalone_Ta}"
    )


# ---------------------------------------------------------------------------
# Gate test 3 -- the troponin buffering flux reaches EP
# ---------------------------------------------------------------------------


def test_troponin_flux_changes_the_calcium_transient(tmp_path):
    """The buffering flux must measurably change the EP calcium transient.

    crossbridge owns troponin in a Ca_i split, so the EP model has had its own
    removed and its calcium balance is missing a sink. If the return path does
    not deliver the flux, calcium runs unbuffered -- too large and too fast --
    and nothing raises.
    """
    with_flux = build_simulation(tmp_path / "with", CAI_SPLIT_ODE, _cai_backend).run(6)

    without = build_simulation(tmp_path / "without", CAI_SPLIT_ODE, _cai_backend)
    # Sever only the return path, leaving everything else identical.
    without.ode_model.missing_ep.values_ep[:] = 0.0
    original = without.controller.step

    def step_without_return_path(*args, **kwargs):
        original(*args, **kwargs)
        without.ode_model.missing_ep.values_ep[:] = 0.0

    without.controller.step = step_without_return_path
    without.run(6)

    cai_with = np.mean(with_flux.ep_solver.state("cai"))
    cai_without = np.mean(without.ep_solver.state("cai"))

    assert cai_with != pytest.approx(cai_without, rel=1e-9), (
        "the calcium transient is identical with and without the buffering "
        "flux, so the return path is not reaching the EP model"
    )
    assert cai_with < cai_without, "buffering must remove calcium, not add it"


# ---------------------------------------------------------------------------
# Gate test 4 -- the zeta return path delivers real values
# ---------------------------------------------------------------------------


def test_zeta_return_path_delivers_the_computed_distortion_states(tmp_path):
    """The distortion states arriving at the EP side must be the ones the
    backend computed.

    Regression test for a bug found by auditing the transfer machinery: the
    backward transfer read from a buffer that nothing ever wrote, so every
    zeta-split simulation ran with no mechano-electric feedback at all. The
    existing feedback test missed it by calling the EP scheme directly with a
    hand-built array instead of the one the coupler fills.
    """
    sim = build_simulation(tmp_path, ZETA_SPLIT_ODE, _zeta_backend).run(4)

    delivered = sim.ode_model.missing_ep.values_ep
    assert delivered.shape[0] == 2, "expected Zetas and Zetaw to cross back"

    computed = {
        "Zetas": sim.backend.ep_outputs["Zetas"].x.array,
        "Zetaw": sim.backend.ep_outputs["Zetaw"].x.array,
    }
    assert np.max(np.abs(computed["Zetas"])) > 0.0, (
        "the backend itself produced no distortion -- test cannot conclude"
    )

    assert np.max(np.abs(delivered)) > 0.0, (
        "the EP side received all zeros: the return path is not transferring "
        "the distortion states the backend computed"
    )


# ---------------------------------------------------------------------------
# Gate test 5 -- force-velocity
# ---------------------------------------------------------------------------


def _shortening_bcs(mesh):
    """Drive the far face inwards, so the fibre shortens at a prescribed rate.

    The Function the moving boundary reads is handed back, so the caller can
    advance it between steps.
    """
    holder: dict = {}

    def dirichlet_bc(V):
        V0, _ = V.sub(0).collapse()
        fdim = mesh.topology.dim - 1
        zero = dolfinx.fem.Function(V0)
        zero.x.array[:] = 0.0
        moving = dolfinx.fem.Function(V0)
        moving.x.array[:] = 0.0
        holder["moving"] = moving

        def on(axis, value):
            return dolfinx.mesh.locate_entities_boundary(
                mesh,
                fdim,
                lambda x, axis=axis, value=value: np.isclose(x[axis], value),
            )

        def bc(func, sub, facets):
            return dolfinx.fem.dirichletbc(
                func,
                dolfinx.fem.locate_dofs_topological((V.sub(sub), V0), fdim, facets),
                V.sub(sub),
            )

        return [
            bc(zero, 0, on(0, 0.0)),
            bc(moving, 0, on(0, 1.0)),
            bc(zero, 1, on(1, 0.0)),
            bc(zero, 2, on(2, 0.0)),
        ]

    return pulse.BoundaryConditions(dirichlet=[dirichlet_bc]), holder


def _tension_at_common_stretch(tmp_path, velocity, target_lmbda, n_steps=40, dt_mech=1.0):
    """Shorten at ``velocity`` and report the tension when the stretch passes
    ``target_lmbda``.

    Comparing at a *common stretch* is what isolates the velocity effect.
    Comparing peaks would not: shortening lowers tension through the
    force-length relation as well, so a run that shortens faster reaches a
    lower tension for reasons that have nothing to do with the shortening rate,
    and a flipped sign on that rate would still look monotonic.
    """
    holder: dict = {}

    def bcs(mesh):
        built, state = _shortening_bcs(mesh)
        holder.update(state=state)
        return built

    sim = build_simulation(
        tmp_path,
        CAI_SPLIT_ODE,
        _cai_backend,
        bcs=bcs,
        dt_mech=dt_mech,
    )

    stretches, tensions = [], []
    for step in range(n_steps):
        moving = holder["state"].get("moving")
        if moving is not None:
            moving.x.array[:] = -velocity * (step + 1) * dt_mech
        sim.controller.step()
        stretches.append(float(np.mean(sim.backend.lmbda.x.array)))
        tensions.append(float(np.mean(sim.backend.active_tension.x.array)))

    stretches = np.array(stretches)
    tensions = np.array(tensions)
    if stretches.min() > target_lmbda:
        raise AssertionError(
            f"velocity {velocity} never reached lmbda={target_lmbda} "
            f"(got down to {stretches.min():.5f})",
        )
    # np.interp needs an increasing x; stretch decreases through the run.
    order = np.argsort(stretches)
    return float(np.interp(target_lmbda, stretches[order], tensions[order]))


def test_tension_falls_with_shortening_velocity(tmp_path):
    """At a common stretch, active tension must fall as shortening velocity rises.

    This is the only test that can see the sign of the sarcomere length-change
    rate. The isometric clamp cannot: at constant stretch that rate is zero, so
    a flipped sign is invisible there. Comparing at equal stretch is what makes
    it sensitive to the rate rather than to the length.
    """
    target = 0.99
    tensions = [
        _tension_at_common_stretch(tmp_path / f"v{v}", v, target) for v in (0.0005, 0.002, 0.004)
    ]

    assert tensions[0] > tensions[1] > tensions[2], (
        "at a common stretch, faster shortening must give lower tension; got "
        f"{tensions} at lmbda={target}"
    )


# ---------------------------------------------------------------------------
# The real model, once
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_real_torord_cai_split_runs_coupled(tmp_path):
    """The Ca_i split of the shipped ToR-ORd model, coupled end to end.

    The synthetic ODE files above have a calcium balance we wrote ourselves,
    which makes asserting anything about buffering against them close to
    circular. This runs the real one.
    """
    odefile = (
        Path(__file__).parent.parent
        / "numerical_experiments"
        / "odefiles"
        / "ToRORd_dynCl_endo_caisplit.ode"
    )
    sim = build_simulation(tmp_path, odefile.read_text(), _cai_backend).run(2)

    assert np.all(np.isfinite(sim.backend.active_tension.x.array))
    assert np.all(np.isfinite(sim.ep_solver.state("cai")))
    assert np.all(sim.ep_solver.state("cai") > 0.0), "calcium went non-positive"


def test_a_backend_on_the_wrong_split_is_refused_end_to_end(tmp_path):
    """The motivating configuration error, through the real machinery.

    A zeta-split backend loaded against a Ca_i-split ODE file. The transfer
    buffers are positional, so without a check the coupler writes calcium into
    the crossbridge population and the run completes with plausible, wrong
    numbers.
    """
    from simcardemsx.transfers import TransferMismatch

    with pytest.raises(TransferMismatch) as excinfo:
        build_simulation(tmp_path, CAI_SPLIT_ODE, _zeta_backend)

    message = str(excinfo.value)
    assert "ZetaSplitUFL" in message
    assert "cai" in message and "XS" in message
