"""
Tests for the activation-backend interface and the zeta-split backend.

Two things are being established here.

First, that porting the Land model onto the backend interface did not change
the physics. `test_matches_the_removed_mechanics_problem` writes out the stress
form that `MechanicsProblem._material_form` used to build, by hand, and requires
the backend to reproduce it exactly. That form is now deleted from the package,
so the test is the only remaining record of it -- which is the point.

Second, that the interface is actually usable by `pulse.StaticProblem` without
a custom problem subclass, which was the reason for supplying `S` directly
rather than a strain-energy potential.
"""

from mpi4py import MPI

import dolfinx
import numpy as np
import pulse
import pytest
import ufl

from simcardemsx.backends import Transfer, ZetaSplitUFL


@pytest.fixture
def mesh():
    return dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)


@pytest.fixture
def dirs(mesh):
    f0 = dolfinx.fem.Constant(mesh, np.array([1.0, 0.0, 0.0]))
    s0 = dolfinx.fem.Constant(mesh, np.array([0.0, 1.0, 0.0]))
    n0 = dolfinx.fem.Constant(mesh, np.array([0.0, 0.0, 1.0]))
    return f0, s0, n0


def _backend(mesh, dirs, activated=True, **kwargs):
    f0, s0, n0 = dirs
    b = ZetaSplitUFL(f0=f0, s0=s0, n0=n0, mesh=mesh, **kwargs)
    if activated:
        b.XS.x.array[:] = 0.02
        b.XW.x.array[:] = 0.05
    return b


def _displacement(mesh, stretch=1.1):
    V = dolfinx.fem.functionspace(mesh, ("P", 2, (3,)))
    u = dolfinx.fem.Function(V)
    u.interpolate(
        lambda x: np.vstack(
            [(stretch - 1.0) * x[0], np.zeros_like(x[1]), np.zeros_like(x[2])],
        ),
    )
    return u


def _integrate(expr, mesh):
    return mesh.comm.allreduce(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(expr * ufl.dx)),
        op=MPI.SUM,
    )


# ---------------------------------------------------------------------------
# The port must not have changed the physics
# ---------------------------------------------------------------------------


def test_matches_the_removed_mechanics_problem(mesh, dirs):
    """The backend's S must equal what MechanicsProblem._material_form built.

    That method has been deleted; its stress construction is reproduced here
    verbatim so the equivalence stays checkable. It read:

        f2 = ufl.inner(C * f0, f0)
        lmbda = ufl.sqrt(f2)
        Sa = self.model.active.Ta(lmbda) * ufl.outer(f0, f0)

    Requires `invariant` explicitly, because that is the convention the deleted
    code used. The default is now `stretch`, which differs by a factor of the
    fibre stretch -- a deliberate correction, not a porting error, and the next
    test pins the factor.
    """
    f0, _, _ = dirs
    backend = _backend(mesh, dirs, formulation=pulse.ActiveStressFormulation.invariant)
    backend.step(t=1.0)

    u = _displacement(mesh)
    F = ufl.Identity(3) + ufl.grad(u)
    C = F.T * F

    # The old construction, written out
    f2 = ufl.inner(C * f0, f0)
    lmbda = ufl.sqrt(f2)
    Sa_reference = backend.Ta(lmbda) * ufl.outer(f0, f0)

    resid = backend.S(C) - Sa_reference
    assert _integrate(ufl.inner(resid, resid), mesh) < 1e-20

    # ...and it is not vacuous: the active stress is actually non-zero here
    assert _integrate(ufl.inner(Sa_reference, Sa_reference), mesh) > 1.0


def test_stretch_is_the_default(mesh, dirs):
    """The R&Q normalization is the default.

    Under it `Ta` is the tension per unit deformed fibre cross-section, which
    is what the Land model's calibration means by it and what the crossbridge
    backends' active stiffness is defined against -- so all backends agree on
    what the number means. The historical `invariant` form remains available
    only for reproducing results generated before this changed.
    """
    backend = _backend(mesh, dirs)
    assert backend.formulation == pulse.ActiveStressFormulation.stretch


def test_stretch_formulation_differs_by_the_stretch(mesh, dirs):
    """S_invariant = lmbda * S_stretch, exactly.

    Pins the factor a user opts into when switching to the Regazzoni &
    Quarteroni convention, which the crossbridge backends require.
    """
    f0, _, _ = dirs
    u = _displacement(mesh, stretch=1.15)
    F = ufl.Identity(3) + ufl.grad(u)
    C = F.T * F

    invariant = _backend(mesh, dirs, formulation=pulse.ActiveStressFormulation.invariant)
    stretch = _backend(mesh, dirs)
    for b in (invariant, stretch):
        b.step(t=1.0)

    lmbda = ufl.sqrt(ufl.inner(C * f0, f0))
    resid = invariant.S(C) - lmbda * stretch.S(C)
    assert _integrate(ufl.inner(resid, resid), mesh) < 1e-20


def test_first_piola_is_normalized_under_stretch_formulation(mesh, dirs):
    """Under the stretch convention |P f0| is exactly Ta, independent of
    stretch -- the property the R&Q stabilization is derived against."""
    f0, _, _ = dirs
    backend = _backend(mesh, dirs)
    backend.step(t=1.0)

    for stretch in (0.95, 1.1, 1.25):
        u = _displacement(mesh, stretch=stretch)
        F = ufl.Identity(3) + ufl.grad(u)
        C = F.T * F
        P = backend.P(F)
        lmbda = ufl.sqrt(ufl.inner(C * f0, f0))

        traction = _integrate(ufl.sqrt(ufl.inner(P * f0, P * f0)), mesh)
        expected = _integrate(backend.Ta(lmbda), mesh)
        np.testing.assert_allclose(traction, expected, rtol=1e-9)


# ---------------------------------------------------------------------------
# Interface conformance
# ---------------------------------------------------------------------------


BACKENDS = [ZetaSplitUFL]


@pytest.mark.parametrize("Backend", BACKENDS)
def test_conforms_to_the_protocol(mesh, dirs, Backend):
    """Every backend must present the whole interface, so the coupler can drive
    any of them without special-casing."""
    f0, s0, n0 = dirs
    backend = Backend(f0=f0, s0=s0, n0=n0, mesh=mesh)

    # Checked member by member rather than with isinstance: ActivationBackend is
    # a plain Protocol, and a runtime_checkable one would only verify that names
    # exist, not that they are callable or of the right type -- which is all the
    # coupler actually depends on.
    for name in ("S", "P", "Fe", "register", "wants_from_ep", "gives_to_ep", "step", "post_solve"):
        assert callable(getattr(backend, name)), f"{Backend.__name__} is missing {name}"

    assert isinstance(backend.active_tension, dolfinx.fem.Function)
    assert isinstance(backend.ep_inputs, dict)


@pytest.mark.parametrize("Backend", BACKENDS)
def test_declared_transfers_are_well_formed(mesh, dirs, Backend):
    """Names must be non-empty and inputs must have a Function to land in --
    the coupler relies on both."""
    f0, s0, n0 = dirs
    backend = Backend(f0=f0, s0=s0, n0=n0, mesh=mesh)

    wants = backend.wants_from_ep()
    gives = backend.gives_to_ep()
    assert all(isinstance(t, Transfer) and t.name for t in wants + gives)

    for t in wants:
        assert t.name in backend.ep_inputs, f"no input Function declared for {t.name}"
        assert isinstance(backend.ep_inputs[t.name], dolfinx.fem.Function)


def test_zeta_split_declares_the_zeta_split(mesh, dirs):
    """The declared interface must match what gotranx derives from a
    zeta-split ODE file: XS/XW in, Zetas/Zetaw back."""
    backend = _backend(mesh, dirs)
    assert {t.name for t in backend.wants_from_ep()} == {"XS", "XW"}
    assert {t.name for t in backend.gives_to_ep()} == {"Zetas", "Zetaw"}


def test_strain_energy_refuses_rather_than_lying(mesh, dirs):
    """There is no closed-form potential for this backend. Raising is better
    than returning something plausible that Newton would then differentiate."""
    backend = _backend(mesh, dirs)
    u = _displacement(mesh)
    F = ufl.Identity(3) + ufl.grad(u)
    with pytest.raises(NotImplementedError, match="strain-energy"):
        backend.strain_energy(F.T * F)


# ---------------------------------------------------------------------------
# It works with an unmodified pulse.StaticProblem
# ---------------------------------------------------------------------------


def test_drives_an_unmodified_static_problem(mesh, dirs):
    """The reason for supplying S directly: no MechanicsProblem subclass.

    Solves a contracting cube with roller boundaries and checks it actually
    shortens along the fibre -- the deleted subclass's job, done by stock pulse.
    """
    f0, s0, n0 = dirs
    backend = _backend(mesh, dirs, activated=False)

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
    geo = pulse.Geometry(mesh=mesh, metadata={"quadrature_degree": 4})

    def dirichlet_bc(V):
        V0, _ = V.sub(0).collapse()
        zero = dolfinx.fem.Function(V0)
        fdim = mesh.topology.dim - 1
        bcs = []
        for i in range(3):
            facets = dolfinx.mesh.locate_entities_boundary(
                mesh,
                fdim,
                lambda x, i=i: np.isclose(x[i], 0.0),
            )
            dofs = dolfinx.fem.locate_dofs_topological((V.sub(i), V0), fdim, facets)
            bcs.append(dolfinx.fem.dirichletbc(zero, dofs, V.sub(i)))
        return bcs

    problem = pulse.StaticProblem(
        model=model,
        geometry=geo,
        bcs=pulse.BoundaryConditions(dirichlet=(dirichlet_bc,)),
        parameters={"base_bc": pulse.problem.BaseBC.free},
    )

    # register() is called by StaticProblem, so the backend now has u
    assert backend.u is problem.u

    # Unactivated: no load, no active stress, so the cube must not move.
    backend.step(t=0.0)
    problem.solve()
    backend.post_solve()
    rest = problem.u.x.array.copy()
    assert np.linalg.norm(rest) < 1e-10, "unactivated cube moved"

    # Now activate and step, rather than jumping. Comparison is against the
    # *unactivated* solve: re-solving with unchanged XS/XW would correctly
    # reproduce the same displacement, since the first solve already reached
    # equilibrium and the stretch rate is then zero.
    #
    # The stepping matters. Going from lambda = 1 to equilibrium in a single
    # step is an enormous shortening velocity, and the distortion states then
    # drive `active_fraction` negative, where the model clamps it to zero --
    # correct force-velocity behaviour, but it would make this test assert
    # against Ta = 0 for reasons that have nothing to do with the port.
    backend.XS.x.array[:] = 0.02
    backend.XW.x.array[:] = 0.05
    for step in range(1, 11):
        backend.step(t=0.2 * step)
        problem.solve()
        backend.post_solve()

    assert np.linalg.norm(problem.u.x.array - rest) > 0.0, "activation produced no motion"
    assert np.all(backend.active_tension.x.array >= 0.0), "active tension went negative"
    assert np.max(backend.active_tension.x.array) > 0.0, "no active tension was generated"
    # Contraction along the fibre: lambda < 1
    assert np.mean(backend.lmbda.x.array) < 1.0, "fibre did not shorten"


def test_post_solve_advances_the_zeta_states(mesh, dirs):
    """post_solve carries the state forward; without it every step would
    restart from the same previous values."""
    backend = _backend(mesh, dirs)
    backend.register(_displacement(mesh, stretch=1.05))

    backend.step(t=1.0)
    backend.post_solve()
    first = backend.Zetas_prev.x.array.copy()

    backend.register(_displacement(mesh, stretch=1.20))
    backend.step(t=2.0)
    backend.post_solve()

    assert not np.allclose(first, backend.Zetas_prev.x.array), "zeta states did not advance"
    np.testing.assert_allclose(float(backend._t_prev.value), 2.0)


def test_land_alias_still_works_but_warns(mesh, dirs):
    """Existing code importing LandModel keeps working, loudly."""
    from simcardemsx.land import LandModel

    f0, s0, n0 = dirs
    with pytest.warns(DeprecationWarning, match="ZetaSplitUFL"):
        model = LandModel(f0=f0, s0=s0, n0=n0, mesh=mesh)
    assert isinstance(model, ZetaSplitUFL)


@pytest.mark.parametrize("space", [("DG", 0), ("DG", 1), ("Lagrange", 1)])
def test_activation_space_is_configurable(space):
    """A researcher can choose where the activation state lives.

    Trading interpolation error against cost is a real choice: a quadrature
    element matching the form removes the error between the activation state
    and what the assembler integrates, but costs more dofs.
    """
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    f0 = dolfinx.fem.Constant(mesh, np.array([1.0, 0.0, 0.0]))
    s0 = dolfinx.fem.Constant(mesh, np.array([0.0, 1.0, 0.0]))
    n0 = dolfinx.fem.Constant(mesh, np.array([0.0, 0.0, 1.0]))

    backend = ZetaSplitUFL(f0=f0, s0=s0, n0=n0, mesh=mesh, element=space)
    expected = dolfinx.fem.functionspace(mesh, space)

    assert backend.function_space.element.signature == expected.element.signature
    # Everything the activation owns must land on that one space, or the
    # stretch and the state it is measured against stop being comparable.
    for name in ("XS", "XW"):
        assert backend.ep_inputs[name].function_space is backend.function_space


def test_both_backends_specify_the_space_the_same_way():
    """One rule, not a per-backend convention."""
    from simcardemsx.backends import CrossbridgeSegregated

    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    f0 = dolfinx.fem.Constant(mesh, np.array([1.0, 0.0, 0.0]))
    s0 = dolfinx.fem.Constant(mesh, np.array([0.0, 1.0, 0.0]))
    n0 = dolfinx.fem.Constant(mesh, np.array([0.0, 0.0, 1.0]))

    space = ("DG", 0)
    zeta = ZetaSplitUFL(f0=f0, s0=s0, n0=n0, mesh=mesh, element=space)
    cai = CrossbridgeSegregated(f0=f0, mesh=mesh, element=space)

    assert zeta.function_space.element.signature == cai.function_space.element.signature


def test_the_default_space_is_unchanged():
    """The default must reproduce what the package did before the space became
    configurable, or making it configurable would have moved everyone's
    results."""
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    f0 = dolfinx.fem.Constant(mesh, np.array([1.0, 0.0, 0.0]))
    s0 = dolfinx.fem.Constant(mesh, np.array([0.0, 1.0, 0.0]))
    n0 = dolfinx.fem.Constant(mesh, np.array([0.0, 0.0, 1.0]))

    default = ZetaSplitUFL(f0=f0, s0=s0, n0=n0, mesh=mesh)
    explicit = ZetaSplitUFL(f0=f0, s0=s0, n0=n0, mesh=mesh, element=("DG", 1))

    assert default.function_space.element.signature == explicit.function_space.element.signature

    for backend in (default, explicit):
        backend.XS.x.array[:] = 0.05
        backend.XW.x.array[:] = 0.02

    points = default.function_space.element.interpolation_points
    a = dolfinx.fem.Function(default.function_space)
    a.interpolate(dolfinx.fem.Expression(default.Ta(1.0), points))
    b = dolfinx.fem.Function(explicit.function_space)
    b.interpolate(dolfinx.fem.Expression(explicit.Ta(1.0), points))

    assert np.allclose(a.x.array, b.x.array)
    assert np.max(a.x.array) > 0.0, "no tension, so agreement proves nothing"
