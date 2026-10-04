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

A step is a trial until ``post_solve`` accepts it (``begin_step`` advances a copy of
the model), so the tests that drive the backend call ``post_solve`` after each
``begin_step``. Review Focus 1-5 and X3 pin that: nothing is accepted before
``post_solve``, or at all when the controller's solve fails.
"""

from dataclasses import dataclass

from mpi4py import MPI

import beat
import crossbridge
import dolfinx
import numpy as np
import pulse
import pytest
import ufl
from conftest import land2017_from_ode, match_land2017_initial_states

from simcardemsx import units
from simcardemsx.averaging import family_name
from simcardemsx.backends import CrossbridgeSegregated, Transfer
from simcardemsx.controller import SimulationController

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
    """Advance the backend at a fixed calcium, in the ms units it expects, accepting
    each step."""
    backend.cai.x.array[:] = cai_mM
    for i in range(n):
        backend.begin_step(i * dt, dt)
        backend.post_solve()
    return backend


def _model_states(model) -> dict[str, np.ndarray]:
    """Copies of every array of a crossbridge model: its states and step bookkeeping."""
    return {
        name: value.copy() for name, value in vars(model).items() if isinstance(value, np.ndarray)
    }


def _assert_same_states(actual: dict[str, np.ndarray], expected: dict[str, np.ndarray]) -> None:
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        np.testing.assert_array_equal(actual[name], value, err_msg=name)


def _set_stretch(u: dolfinx.fem.Function, stretch: float) -> None:
    """Set ``u`` in place so that its fibre (x) stretch is exactly ``stretch`` everywhere."""
    u.interpolate(
        lambda x: np.vstack([(stretch - 1.0) * x[0], np.zeros_like(x[1]), np.zeros_like(x[2])]),
    )


def _stretched(mesh, stretch: float) -> dolfinx.fem.Function:
    """A P2 displacement whose fibre (x) stretch is exactly ``stretch`` everywhere."""
    u = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("P", 2, (3,))))
    _set_stretch(u, stretch)
    return u


@pytest.fixture
def advanced_with(monkeypatch) -> list[tuple[np.ndarray, np.ndarray]]:
    """``(SL, dSL)`` of every ``crossbridge.Land2017.advance_step`` call, in order.

    Patched on the class, so it also sees the deep-copied trials ``begin_step`` makes.
    """
    calls: list[tuple[np.ndarray, np.ndarray]] = []
    advance_step = crossbridge.Land2017.advance_step

    def record(model, dt, Ca_val, SL_vals, dSL_vals=None):
        calls.append((np.array(SL_vals, copy=True), np.array(dSL_vals, copy=True)))
        advance_step(model, dt, Ca_val, SL_vals, dSL_vals=dSL_vals)

    monkeypatch.setattr(crossbridge.Land2017, "advance_step", record)
    return calls


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
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="dt"):
        backend.step(t=0.0)


def test_step_is_a_deprecated_alias_of_begin_step(mesh, f0):
    """Direct users of ``step`` must now call ``post_solve`` to accept the step, so
    the warning names both."""
    stepped = CrossbridgeSegregated(f0=f0, mesh=mesh)
    begun = CrossbridgeSegregated(f0=f0, mesh=mesh)
    stepped.cai.x.array[:] = begun.cai.x.array[:] = 1e-3

    with pytest.warns(DeprecationWarning, match="begin_step.*post_solve"):
        stepped.step(t=0.0, dt=0.5)
    begun.begin_step(0.0, 0.5)
    stepped.post_solve()
    begun.post_solve()

    _assert_same_states(_model_states(stepped.model), _model_states(begun.model))


def test_time_unit_is_milliseconds(mesh, f0):
    """The backend takes ms and crossbridge takes s; a 1000x error here would
    look like an implausibly fast or slow twitch, not a crash."""
    ms = _drive(CrossbridgeSegregated(f0=f0, mesh=mesh), n=20, dt=1.0)

    reference = crossbridge.Land2017(num_cells=ms.num_cells)
    for _ in range(20):
        reference.advance_step(1e-3, 1.0, 1.8, dSL_vals=0.0)  # 1 ms in seconds

    np.testing.assert_allclose(
        ms.tension_kPa.x.array,
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
        backend.begin_step(i * dt, dt)
        backend.post_solve()
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
    for name in (
        "S",
        "P",
        "Fe",
        "register",
        "wants_from_ep",
        "gives_to_ep",
        "begin_step",
        "step",
        "post_solve",
    ):
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


def test_states_live_on_quadrature_when_asked(mesh, f0):
    """With ``quadrature_degree`` everything lives on a scalar quadrature space at that
    degree, as for ``GeneratedActivation``; with neither option on DG1, as before.
    ``active_tension`` is the P1 average either way."""
    backend = CrossbridgeSegregated(f0=f0, mesh=mesh, quadrature_degree=2)
    element = backend.space.ufl_element()
    assert (element.family_name, element.degree) == ("quadrature", 2)
    assert backend.quadrature_degree == 2
    assert backend.mesh is mesh
    for function in (backend.cai, backend.J_TRPN, backend.lmbda_prev, backend.tension_kPa):
        assert function.function_space is backend.space
    assert backend.num_cells == backend.tension_kPa.x.array.size

    default = CrossbridgeSegregated(f0=f0, mesh=mesh)
    element = default.space.ufl_element()
    assert (family_name(element), element.degree) == ("DG", 1)
    assert default.quadrature_degree is None

    for b in (backend, default):
        p1 = b.active_tension.function_space.ufl_element()
        assert (family_name(p1), p1.degree) == ("P", 1)

    with pytest.raises(ValueError, match="element.*quadrature_degree"):
        CrossbridgeSegregated(f0=f0, mesh=mesh, element=("DG", 1), quadrature_degree=2)


def test_is_evaluated_at_the_end_of_the_step(mesh, f0):
    """Under ``pulse.DynamicProblem`` the increment ``λ(u) - λ_n`` must be measured at
    ``u_{n+1}``, not at the alpha_f point. pulse's ``ActiveModel`` defaults the flag to
    False, so deleting the line would otherwise go unnoticed."""
    assert CrossbridgeSegregated.evaluate_at_end_of_step is True
    assert CrossbridgeSegregated(f0=f0, mesh=mesh).evaluate_at_end_of_step is True


def test_post_solve_accepts_the_step(mesh, f0):
    """``begin_step`` advances a trial and accepts nothing: the model, the stretch, the
    reported tension and stiffness, ``active_tension`` and ``J_TRPN`` are those of the
    last accepted step until ``post_solve`` commits it.

    ``SL_ref`` is not Land2017's ``SL0`` (1.8 um), so that the stiffness's rescaling is
    not the identity."""
    backend = _drive(CrossbridgeSegregated(f0=f0, mesh=mesh, SL_ref=2.0), n=4)
    assert backend.SL0 == 1.8
    states = _model_states(backend.model)
    reported = {
        name: getattr(backend, name).x.array.copy()
        for name in ("tension_kPa", "stiffness_kPa", "active_tension", "J_TRPN", "lmbda_prev")
    }

    backend.begin_step(2.0, 0.5)

    _assert_same_states(_model_states(backend.model), states)
    for name, value in reported.items():
        np.testing.assert_array_equal(getattr(backend, name).x.array, value, err_msg=name)
    trial = _model_states(backend._trial)
    assert not np.array_equal(trial["CaTRPN"], states["CaTRPN"]), "the trial did not advance"

    backend.post_solve()

    _assert_same_states(_model_states(backend.model), trial)
    np.testing.assert_array_equal(backend.tension_kPa.x.array, backend.model.get_active_tension())
    np.testing.assert_array_equal(
        backend.stiffness_kPa.x.array,
        units.active_stiffness_to_mechanics(
            backend.model.get_active_stiffness(),
            SL_ref=2.0,
            SL0=1.8,
        ),
    )
    assert not np.array_equal(backend.stiffness_kPa.x.array, reported["stiffness_kPa"])
    np.testing.assert_array_equal(
        backend.J_TRPN.x.array,
        units.troponin_flux(backend.model.get_calcium_binding_rate()),
    )
    assert not np.array_equal(backend.active_tension.x.array, reported["active_tension"])


def test_lambda_prev_is_the_stretch_the_ode_used(mesh, f0, advanced_with):
    """The invariant the whole scheme rests on (ADR 0002): the stretch the trial is
    advanced with is the stabilization's ``lmbda_prev``, the accepted λ_n its increment
    ``λ(u) - λ_n`` is measured from during the solve, and not λ of wherever ``u`` is.
    The shortening velocity is (λ_n - λ_{n-1}) SL_ref / dt, in um/s, and accepting a
    step shifts λ_n to λ_{n-1}.

    The velocities are compared bit for bit, computed in the backend's order of
    operations, from the accepted stretches."""
    backend = CrossbridgeSegregated(f0=f0, mesh=mesh)
    u = _stretched(mesh, 0.92)
    backend.register(u)
    backend.cai.x.array[:] = 1e-3
    dt_s = 0.5e-3
    backend.begin_step(0.0, 0.5)
    backend.post_solve()  # accepts λ(u) = 0.92 as λ_n, with λ_{n-1} = 1
    lmbda_n = backend.lmbda_prev.x.array.copy()
    np.testing.assert_allclose(lmbda_n, 0.92, rtol=1e-9)

    # u moves before the next step, as a failed solve leaves it.
    _set_stretch(u, 0.97)
    advanced_with.clear()
    backend.begin_step(0.5, 0.5)

    ((SL, dSL),) = advanced_with
    np.testing.assert_array_equal(backend._active.lmbda_prev.x.array, lmbda_n)
    np.testing.assert_array_equal(SL, units.stretch_to_sarcomere_length(lmbda_n, backend.SL_ref))
    np.testing.assert_array_equal(dSL, (lmbda_n - 1.0) * backend.SL_ref / dt_s)

    # The step is accepted at u: λ = 0.97 is the new λ_n, and 0.92 the new λ_{n-1}.
    backend.post_solve()
    lmbda_new = backend.lmbda_prev.x.array.copy()
    np.testing.assert_allclose(lmbda_new, 0.97, rtol=1e-9)
    advanced_with.clear()
    backend.begin_step(1.0, 0.5)

    ((SL, dSL),) = advanced_with
    np.testing.assert_array_equal(SL, units.stretch_to_sarcomere_length(lmbda_new, backend.SL_ref))
    np.testing.assert_array_equal(dSL, (lmbda_new - lmbda_n) * backend.SL_ref / dt_s)


def test_reset_stretch_seeds_a_deformed_start(mesh, f0, advanced_with):
    """A problem that starts deformed: ``reset_stretch`` takes λ(u) as the stretch at
    rest, so the first step is advanced at it with no shortening velocity. The model and
    everything reported are untouched. Without a displacement it cannot."""
    with pytest.raises(RuntimeError, match="register"):
        CrossbridgeSegregated(f0=f0, mesh=mesh).reset_stretch()

    backend = CrossbridgeSegregated(f0=f0, mesh=mesh)
    backend.register(_stretched(mesh, 0.9))
    states = _model_states(backend.model)
    reported = {
        name: getattr(backend, name).x.array.copy()
        for name in ("tension_kPa", "stiffness_kPa", "active_tension", "J_TRPN")
    }

    backend.reset_stretch()

    _assert_same_states(_model_states(backend.model), states)
    for name, value in reported.items():
        np.testing.assert_array_equal(getattr(backend, name).x.array, value, err_msg=name)

    backend.cai.x.array[:] = 1e-3
    backend.begin_step(0.0, 0.5)
    ((SL, dSL),) = advanced_with
    np.testing.assert_allclose(SL, 0.9 * backend.SL_ref, rtol=1e-12)
    np.testing.assert_array_equal(dSL, 0.0)


# ---------------------------------------------------------------------------
# Review Focus: inputs the gates do not exercise
# ---------------------------------------------------------------------------


def test_begin_step_twice_starts_from_the_committed_state(mesh, f0):
    """Review Focus 1: a caller retrying a step calls ``begin_step`` again without a
    ``post_solve``. The retry must advance the committed model, not the first trial."""
    retried = _drive(CrossbridgeSegregated(f0=f0, mesh=mesh), n=4)
    once = _drive(CrossbridgeSegregated(f0=f0, mesh=mesh), n=4)

    # The first attempt differs in dt and calcium, so a second call that returned early
    # on a pending trial, or advanced the first trial, would not match the fresh backend.
    retried.cai.x.array[:] = 5e-3
    retried.begin_step(2.0, 0.25)
    retried.cai.x.array[:] = 1e-3
    retried.begin_step(2.0, 0.5)
    retried.post_solve()
    once.begin_step(2.0, 0.5)
    once.post_solve()

    _assert_same_states(_model_states(retried.model), _model_states(once.model))
    np.testing.assert_array_equal(retried.tension_kPa.x.array, once.tension_kPa.x.array)


def test_begin_step_at_rest_is_the_identity(mesh, f0, advanced_with):
    """Review Focus 2: a rest step at ``dt == 0`` (rodero_05's unloaded solve) leaves
    the model unchanged, and never divides by ``dt`` for the shortening velocity, even
    with a stretch that changed over the step before.

    It accepts ``u`` as the rest state: ``u`` moves during that solve, and the next
    step is advanced at the new λ with no shortening velocity."""
    backend = _drive(CrossbridgeSegregated(f0=f0, mesh=mesh), n=4)
    u = _stretched(mesh, 0.95)
    backend.register(u)
    backend.begin_step(2.0, 0.5)
    backend.post_solve()  # λ_n = 0.95, λ_{n-1} = 1
    states = _model_states(backend.model)
    reported = {
        name: getattr(backend, name).x.array.copy()
        for name in ("tension_kPa", "stiffness_kPa", "J_TRPN")
    }

    with np.errstate(divide="raise", invalid="raise"):
        backend.begin_step(2.5, 0.0)
        _set_stretch(u, 0.9)  # the solve moves u
        backend.post_solve()

    _assert_same_states(_model_states(backend.model), states)
    for name, value in reported.items():
        np.testing.assert_array_equal(getattr(backend, name).x.array, value, err_msg=name)
    np.testing.assert_allclose(backend.lmbda_prev.x.array, 0.9, rtol=1e-9)

    advanced_with.clear()
    backend.begin_step(2.5, 0.5)
    ((SL, dSL),) = advanced_with
    np.testing.assert_array_equal(
        SL,
        units.stretch_to_sarcomere_length(backend.lmbda_prev.x.array, backend.SL_ref),
    )
    np.testing.assert_array_equal(dSL, 0.0)


def test_post_solve_without_register_keeps_the_stretch(mesh, f0):
    """Review Focus 3: used directly, without a problem, there is no displacement to
    take λ from. The step is still committed, and λ stays at its last accepted value."""
    backend = CrossbridgeSegregated(f0=f0, mesh=mesh)
    backend.cai.x.array[:] = 1e-3
    backend.begin_step(0.0, 0.5)
    trial = backend._trial

    backend.post_solve()

    assert backend.model is trial
    np.testing.assert_array_equal(backend.lmbda_prev.x.array, 1.0)


def test_post_solve_without_begin_step_raises(mesh, f0):
    """Review Focus 4: ``post_solve`` with no ``begin_step`` since the last commit must
    not commit a stale trial (or nothing) a second time."""
    backend = CrossbridgeSegregated(f0=f0, mesh=mesh)
    with pytest.raises(RuntimeError, match="begin_step"):
        backend.post_solve()

    backend.begin_step(0.0, 0.5)
    backend.post_solve()
    with pytest.raises(RuntimeError, match="begin_step"):
        backend.post_solve()


def _land2017(mech_module):
    """``backend_factory`` for ``make_mechanics``: crossbridge Land2017 with the ``.ode``'s
    parameters and initial states, on quadrature at the degree it is given."""

    def factory(mesh, f0, quadrature_degree):
        backend = CrossbridgeSegregated(
            f0,
            mesh,
            "Land2017",
            quadrature_degree=quadrature_degree,
            params=land2017_from_ode(mech_module),
        )
        assert isinstance(backend.model, crossbridge.Land2017)
        match_land2017_initial_states(backend.model, mech_module)
        return backend

    return factory


def _unit_cube(n: int) -> dolfinx.mesh.Mesh:
    return dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n)


def test_controller_refuses_mismatched_crossbridge_quadrature(
    split_modules,
    make_ep_solver,
    make_mechanics,
):
    """Review Focus 5: as for ``GeneratedActivation``, states at a quadrature degree
    other than the form's would silently change the quadrature of the whole momentum
    integral."""
    modules = split_modules["caisplit"]
    ep_solver = make_ep_solver(modules.ep, _unit_cube(1))
    problem, backend = make_mechanics(
        modules.mechanics,
        _unit_cube(1),
        quadrature_degree=2,
        backend_quadrature_degree=3,
        backend_factory=_land2017(modules.mechanics),
    )
    assert backend.quadrature_degree == 3
    with pytest.raises(ValueError, match="degree 3.*degree 2"):
        SimulationController(problem, ep_solver, backend, modules, 1.0, 0.1)


@dataclass
class _FailingSolve:
    """A mechanics driver that solves for real, and reports failure once ``fail`` is set."""

    problem: pulse.StaticProblem
    fail: bool = False

    def advance(self, t_n: float, dt: float) -> bool:
        converged = self.problem.solve()
        return converged and not self.fail


def _accepted(backend: CrossbridgeSegregated, ode) -> dict[str, np.ndarray]:
    """Copies of what accepting a step changes, in the backend and then in EP."""
    return {
        **{f"model.{name}": value for name, value in _model_states(backend.model).items()},
        "lmbda_prev": backend.lmbda_prev.x.array.copy(),
        "_lmbda_old": backend._lmbda_old.copy(),
        "tension_kPa": backend.tension_kPa.x.array.copy(),
        "stiffness_kPa": backend.stiffness_kPa.x.array.copy(),
        "active_tension": backend.active_tension.x.array.copy(),
        "outputs['J_TRPN']": backend.outputs["J_TRPN"].x.array.copy(),
        "ode.missing_variables": ode.missing_variables.copy(),
    }


@pytest.mark.slow
def test_failed_step_accepts_nothing(split_modules, make_ep_solver, make_mechanics):
    """X3: a step whose solve fails makes the controller raise ``RuntimeError``, and
    nothing of it has been accepted.

    Gate 5's set-up (beat EP on the 3x3x3 cube, one element), with crossbridge Land2017
    on quadrature. Two converged steps come first, so that the model and both stretches
    have left their initial values. The next step's solve runs for real and its driver
    reports failure, so ``begin_step`` has advanced a trial and the solve has moved
    ``u``; the controller must raise without calling ``post_solve`` or moving anything
    back to EP. The failed step's EP micro-steps are not rolled back, for any driver.
    """
    modules = split_modules["caisplit"]
    ep_solver = make_ep_solver(modules.ep, _unit_cube(3))
    problem, backend = make_mechanics(
        modules.mechanics,
        _unit_cube(1),
        quadrature_degree=2,
        backend_factory=_land2017(modules.mechanics),
    )
    assert isinstance(backend, CrossbridgeSegregated)
    driver = _FailingSolve(problem)
    controller = SimulationController(driver, ep_solver, backend, modules, 1.0, 0.05)
    ode = ep_solver.ode
    assert isinstance(ode, beat.odesolver.DolfinODESolver)
    assert ode.missing_variables is not None

    controller.step()
    controller.step()
    model = backend.model
    before = _accepted(backend, ode)
    assert np.any(before["outputs['J_TRPN']"] != 0.0)
    assert np.all(before["lmbda_prev"] != 1.0)
    assert np.all(before["_lmbda_old"] != 1.0)
    assert not np.array_equal(before["_lmbda_old"], before["lmbda_prev"])

    driver.fail = True
    with pytest.raises(RuntimeError, match="did not converge"):
        controller.step()

    assert backend.model is model
    after = _accepted(backend, ode)
    for name, value in before.items():
        assert np.array_equal(after[name], value), name

    # The failed step did have something to accept: its trial, pending still.
    backend.post_solve()
    assert not np.array_equal(backend.model.CaTRPN, before["model.CaTRPN"])


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
        backend.begin_step(i * 0.5, 0.5)
        problem.solve()
        backend.post_solve()

    assert np.max(backend.active_tension.x.array) > 0.0, "no tension generated"
    assert np.mean(backend.lmbda_prev.x.array) < 1.0, "fibre did not shorten"


def test_reset_stretch_discards_a_pending_step(mesh, f0):
    """The pending trial was advanced with the old λ_n, while the stress would measure
    from the new one, so ``reset_stretch`` drops it: ``post_solve`` needs a fresh
    ``begin_step``."""
    backend = CrossbridgeSegregated(f0=f0, mesh=mesh)
    backend.register(_stretched(mesh, 0.9))
    backend.cai.x.array[:] = 1e-3
    backend.begin_step(0.0, 0.5)
    backend.reset_stretch()

    with pytest.raises(RuntimeError, match="begin_step"):
        backend.post_solve()
