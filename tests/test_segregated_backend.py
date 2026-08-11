"""
Tests for the Ca_i-split crossbridge backend and the unit conversions it needs.

This backend is where three libraries' conventions meet, and where a segregated
coupling has to be stabilized to be convergent at all. Both are places where
mistakes are silent rather than loud, so the tests concentrate there:

- `test_stiffness_rescaling_is_applied` -- the one chain rule in the unit
  layer. Omitting it leaves the scheme stable but the Newton tangent wrong.
- `test_stabilization_is_in_the_tangent` -- the stabilization must be visible
  to Newton, or it is inert and the instability it exists to remove remains.
- `test_troponin_flux_conserves_calcium` -- integrating the returned J_TRPN
  must account for exactly the calcium the model absorbed, or the coupled
  calcium balance leaks.
"""

from mpi4py import MPI

import crossbridge
import dolfinx
import numpy as np
import pulse
import pytest
import ufl

from simcardemsx import units
from simcardemsx.backends import CrossbridgeSegregated, Transfer

MODELS = sorted(crossbridge.MODEL_REGISTRY)


@pytest.fixture
def mesh():
    return dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)


@pytest.fixture
def f0(mesh):
    return dolfinx.fem.Constant(mesh, np.array([1.0, 0.0, 0.0]))


#: RDQ18 defines no SL0, so a reference sarcomere length must be supplied for it.
SL_REF_FOR = {"RDQ18": 2.0}


def _make(f0, mesh, model="Land2017", **kwargs):
    kwargs.setdefault("SL_ref", SL_REF_FOR.get(model))
    return CrossbridgeSegregated(f0=f0, mesh=mesh, model=model, **kwargs)


def _drive(backend, cai_mM=1e-3, n=40, dt=0.5):
    """Advance the backend at a fixed calcium, in the ms units it expects."""
    backend.cai.x.array[:] = cai_mM
    for i in range(n):
        backend.step(t=i * dt, dt=dt)
    return backend


def _integrate(expr, mesh):
    return mesh.comm.allreduce(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(expr * ufl.dx)),
        op=MPI.SUM,
    )


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------


def test_calcium_conversion_anchor():
    """A physiological ToR-ORd calcium must land in crossbridge's range.

    1e-4 mM is 0.1 uM, a typical diastolic level; getting the factor wrong by
    1000 puts it somewhere the model will integrate quite happily into a flat
    line.
    """
    np.testing.assert_allclose(units.calcium_to_crossbridge(np.array([1e-4])), 0.1)
    np.testing.assert_allclose(units.calcium_to_crossbridge(np.array([1.6e-3])), 1.6)


def test_conversions_round_trip():
    x = np.array([0.3, 1.0, 2.5])
    np.testing.assert_allclose(units.calcium_from_crossbridge(units.calcium_to_crossbridge(x)), x)
    np.testing.assert_allclose(units.s_to_ms(units.ms_to_s(0.5)), 0.5)


def test_stiffness_rescaling_is_the_chain_rule():
    """Ka is reported per unit of the model's Lambda = SL/SL0, but consumed
    against the solver's lambda, where SL = lambda * SL_ref."""
    Ka = np.array([100.0, 200.0])

    # The usual case: no rescaling, which is why omitting it goes unnoticed
    np.testing.assert_allclose(
        units.active_stiffness_to_mechanics(Ka, SL_ref=1.8, SL0=1.8),
        Ka,
    )
    # And the general one
    np.testing.assert_allclose(
        units.active_stiffness_to_mechanics(Ka, SL_ref=2.0, SL0=1.8),
        Ka * (2.0 / 1.8),
    )


def test_troponin_flux_matches_torord_units():
    """J_TRPN = rate * trpnmax, converted from per-second to per-millisecond."""
    rate = np.array([2.0])  # occupied fraction per second
    np.testing.assert_allclose(units.troponin_flux(rate), 2.0 * 0.07 / 1000.0)
    np.testing.assert_allclose(units.troponin_flux(rate, trpnmax_mM=0.1), 2.0 * 0.1 / 1000.0)


# ---------------------------------------------------------------------------
# The unit layer is actually wired in
# ---------------------------------------------------------------------------


def test_stiffness_rescaling_is_applied(mesh, f0):
    """The backend must apply the chain rule, not just define it.

    Two backends with different SL_ref, driven identically, must report Ka
    differing by exactly SL_ref/SL0 -- while Ta, which is not a derivative with
    respect to stretch, is unaffected by the rescaling itself.
    """
    same = _drive(CrossbridgeSegregated(f0=f0, mesh=mesh, SL_ref=1.8))
    scaled = _drive(CrossbridgeSegregated(f0=f0, mesh=mesh, SL_ref=1.8 * 1.25))

    # Both sit at lambda = 1 (never solved), so the models see SL = SL_ref and
    # differ only through it; compare the ratio against the pure factor by
    # reconstructing what crossbridge itself reported.
    raw_same = same.model.get_active_stiffness()
    raw_scaled = scaled.model.get_active_stiffness()
    np.testing.assert_allclose(same.active_stiffness.x.array, raw_same, rtol=1e-12)
    np.testing.assert_allclose(scaled.active_stiffness.x.array, raw_scaled * 1.25, rtol=1e-12)


def test_step_requires_an_explicit_dt(mesh, f0):
    """Silently defaulting dt would put the ODE on a different clock from the
    coupling, which is exactly the kind of thing that produces plausible but
    wrong transients."""
    backend = CrossbridgeSegregated(f0=f0, mesh=mesh)
    with pytest.raises(ValueError, match="dt"):
        backend.step(t=0.0)


def test_time_unit_is_milliseconds(mesh, f0):
    """The backend takes ms and crossbridge takes s; a 1000x error here would
    look like an implausibly fast or slow twitch, not a crash."""
    ms = _drive(CrossbridgeSegregated(f0=f0, mesh=mesh), n=20, dt=1.0)

    reference = crossbridge.Land2017(num_cells=ms.num_cells)
    for _ in range(20):
        reference.advance_step(1e-3, 1.0, 1.8, dSL_vals=0.0)  # 1 ms in seconds

    np.testing.assert_allclose(
        ms.active_tension.x.array,
        reference.get_active_tension(),
        rtol=1e-10,
    )


# ---------------------------------------------------------------------------
# The stabilization
# ---------------------------------------------------------------------------


def test_stabilization_vanishes_at_its_own_fixed_point(mesh, f0):
    """With lambda == lambda_prev the stress must be exactly the unstabilized
    one, so the term biases nothing."""
    backend = _drive(CrossbridgeSegregated(f0=f0, mesh=mesh))
    naive = _drive(CrossbridgeSegregated(f0=f0, mesh=mesh, stabilized=False))

    # Undeformed: lambda = 1 = lambda_prev
    C = ufl.variable(ufl.Identity(3))
    resid = backend.S(C) - naive.S(C)
    assert _integrate(ufl.inner(resid, resid), mesh) < 1e-20


def test_stabilization_is_in_the_tangent(mesh, f0):
    """Newton must see the active stiffness.

    A stabilization term the Jacobian does not contain is inert: the solve
    converges to the same unstabilized answer and the oscillatory instability
    survives. So the check is that the assembled tangent *changes* when the
    stabilization is switched on.
    """
    V = dolfinx.fem.functionspace(mesh, ("P", 1, (3,)))
    rng = np.random.default_rng(0)

    # Drawn once, outside tangent(): the two tangents must be evaluated at the
    # *same* displacement, or they differ for that reason alone and the
    # comparison below says nothing about Ka.
    u = dolfinx.fem.Function(V)
    u.x.array[:] = 0.03 * rng.standard_normal(u.x.array.size)
    direction = rng.standard_normal(V.dofmap.index_map.size_local * V.dofmap.index_map_bs)

    def tangent(stabilized):
        backend = _drive(CrossbridgeSegregated(f0=f0, mesh=mesh, stabilized=stabilized))
        # lambda_prev != lambda(u), so the stabilization term is actually live
        backend.lmbda_prev.x.array[:] = 1.04
        v = ufl.TestFunction(V)
        F = ufl.Identity(3) + ufl.grad(u)
        C = F.T * F
        R = ufl.inner(backend.S(C), 0.5 * ufl.derivative(C, u, v)) * ufl.dx
        J = ufl.derivative(R, u, ufl.TrialFunction(V))
        mat = dolfinx.fem.assemble_matrix(dolfinx.fem.form(J))
        mat.scatter_reverse()
        return mat.to_dense() @ direction

    with_Ka = tangent(True)
    without_Ka = tangent(False)
    rel = np.linalg.norm(with_Ka - without_Ka) / max(np.linalg.norm(without_Ka), 1e-30)
    assert rel > 1e-3, (
        f"the stabilization does not affect the Jacobian, so Newton cannot see it (rel {rel})"
    )


def test_stress_matches_the_energy(mesh, f0):
    """S must equal 2 dPsi/dC. Guards the delegation to pulse against drift."""
    backend = _drive(CrossbridgeSegregated(f0=f0, mesh=mesh))
    backend.lmbda_prev.x.array[:] = 1.05  # so the stabilization term is live

    u = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("P", 2, (3,))))
    u.interpolate(lambda x: np.vstack([0.12 * x[0], np.zeros_like(x[1]), np.zeros_like(x[2])]))
    F = ufl.Identity(3) + ufl.grad(u)
    C = ufl.variable(F.T * F)

    resid = backend.S(C) - 2.0 * ufl.diff(backend.strain_energy(C), C)
    assert _integrate(ufl.inner(resid, resid), mesh) < 1e-16


# ---------------------------------------------------------------------------
# The calcium return path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", MODELS)
def test_troponin_flux_conserves_calcium(mesh, f0, model):
    """Integrating the returned J_TRPN must equal trpnmax times the change in
    occupancy -- the property the EP calcium balance depends on."""
    backend = _make(f0, mesh, model)
    dt = units.s_to_ms(backend.model.dt)  # step at the model's own dt

    start = backend.model.bound_calcium_fraction().copy()
    backend.cai.x.array[:] = 2e-3  # 2 uM

    integrated = np.zeros(backend.num_cells)
    for i in range(200):
        backend.step(t=i * dt, dt=dt)
        integrated += backend.J_TRPN.x.array * dt

    absorbed = (backend.model.bound_calcium_fraction() - start) * backend.trpnmax
    assert np.max(np.abs(absorbed)) > 1e-6, f"{model}: no calcium was absorbed"
    np.testing.assert_allclose(integrated, absorbed, rtol=1e-9, atol=1e-15)


def test_missing_reference_length_is_an_error(mesh, f0):
    """RDQ18 has no SL0. Guessing one would put the model at an arbitrary point
    on its force-length curve, which scales the force it generates."""
    with pytest.raises(ValueError, match="SL_ref"):
        CrossbridgeSegregated(f0=f0, mesh=mesh, model="RDQ18")
    CrossbridgeSegregated(f0=f0, mesh=mesh, model="RDQ18", SL_ref=2.0)  # explicit is fine


def test_declares_the_calcium_split(mesh, f0):
    """cai in, J_TRPN back -- matching a caisplit ODE file's `missing` dicts."""
    backend = CrossbridgeSegregated(f0=f0, mesh=mesh)
    assert [t.name for t in backend.wants_from_ep()] == ["cai"]
    assert [t.name for t in backend.gives_to_ep()] == ["J_TRPN"]
    assert backend.wants_from_ep()[0].unit == "mM"
    assert "cai" in backend.ep_inputs
    assert "J_TRPN" in backend.ep_outputs


# ---------------------------------------------------------------------------
# Interface and models
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", MODELS)
def test_every_crossbridge_model_can_drive_the_backend(mesh, f0, model):
    """The registry promises interchangeability; this is where that is used."""
    backend = _make(f0, mesh, model)
    dt = units.s_to_ms(backend.model.dt)
    _drive(backend, cai_mM=2e-3, n=50, dt=dt)

    assert np.all(np.isfinite(backend.active_tension.x.array))
    assert np.all(backend.active_stiffness.x.array >= 0.0)
    assert np.max(backend.active_tension.x.array) > 0.0, f"{model} generated no tension"


def test_conforms_to_the_backend_interface(mesh, f0):
    backend = CrossbridgeSegregated(f0=f0, mesh=mesh)
    for name in ("S", "P", "Fe", "register", "wants_from_ep", "gives_to_ep", "step", "post_solve"):
        assert callable(getattr(backend, name)), f"missing {name}"
    assert isinstance(backend.active_tension, dolfinx.fem.Function)
    for t in backend.wants_from_ep() + backend.gives_to_ep():
        assert isinstance(t, Transfer) and t.name


def test_arrays_cover_every_local_dof(mesh, f0):
    """num_cells must match the Function array length exactly, ghosts included.

    A mismatch here is the classic way this breaks in parallel, and it would
    show up as a shape error only if the arrays happened to differ in length.
    """
    backend = CrossbridgeSegregated(f0=f0, mesh=mesh)
    assert backend.num_cells == backend.Ta_current.x.array.size
    assert backend.model.num_cells == backend.Ta_current.x.array.size


def test_post_solve_is_deliberately_a_noop(mesh, f0):
    """The stretch is captured in step(), so that the value the ODE used and the
    value the stabilization measures against cannot diverge."""
    backend = _drive(CrossbridgeSegregated(f0=f0, mesh=mesh))
    before = (
        backend.Ta_current.x.array.copy(),
        backend.Ka_current.x.array.copy(),
        backend.lmbda_prev.x.array.copy(),
    )
    backend.post_solve()
    for a, b in zip(before, (backend.Ta_current, backend.Ka_current, backend.lmbda_prev)):
        np.testing.assert_array_equal(a, b.x.array)


def test_lambda_prev_is_the_stretch_the_ode_used(mesh, f0):
    """The invariant the whole scheme rests on."""
    backend = CrossbridgeSegregated(f0=f0, mesh=mesh)
    u = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("P", 2, (3,))))
    u.interpolate(lambda x: np.vstack([-0.08 * x[0], np.zeros_like(x[1]), np.zeros_like(x[2])]))
    backend.register(u)

    backend.cai.x.array[:] = 1e-3
    backend.step(t=0.0, dt=0.5)

    np.testing.assert_allclose(backend.lmbda_prev.x.array, 0.92, rtol=1e-9)
    np.testing.assert_array_equal(backend.lmbda_prev.x.array, backend.lmbda.x.array)


def test_drives_an_unmodified_static_problem(mesh, f0):
    """End to end: stock pulse.StaticProblem, contracting cube."""
    s0 = dolfinx.fem.Constant(mesh, np.array([0.0, 1.0, 0.0]))
    backend = CrossbridgeSegregated(f0=f0, mesh=mesh)

    model = pulse.CardiacModel(
        material=pulse.HolzapfelOgden(
            f0=f0,
            s0=s0,
            **pulse.HolzapfelOgden.transversely_isotropic_parameters(),
        ),
        active=backend,
        compressibility=pulse.Incompressible(),
    )

    def dirichlet_bc(V):
        V0, _ = V.sub(0).collapse()
        zero = dolfinx.fem.Function(V0)
        fdim = mesh.topology.dim - 1
        out = []
        for i in range(3):
            facets = dolfinx.mesh.locate_entities_boundary(
                mesh,
                fdim,
                lambda x, i=i: np.isclose(x[i], 0.0),
            )
            dofs = dolfinx.fem.locate_dofs_topological((V.sub(i), V0), fdim, facets)
            out.append(dolfinx.fem.dirichletbc(zero, dofs, V.sub(i)))
        return out

    problem = pulse.StaticProblem(
        model=model,
        geometry=pulse.Geometry(mesh=mesh, metadata={"quadrature_degree": 4}),
        bcs=pulse.BoundaryConditions(dirichlet=(dirichlet_bc,)),
        parameters={"base_bc": pulse.problem.BaseBC.free},
    )
    assert backend.u is problem.u

    backend.cai.x.array[:] = 2e-3
    for i in range(20):
        backend.step(t=i * 0.5, dt=0.5)
        problem.solve()
        backend.post_solve()

    assert np.max(backend.active_tension.x.array) > 0.0, "no tension generated"
    assert np.mean(backend.lmbda.x.array) < 1.0, "fibre did not shorten"
