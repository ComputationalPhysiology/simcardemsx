# %% [markdown]
# # A thick-walled cylinder against its exact solution
#
# The [previous tutorial](05_zero_mech_reference.ipynb) checked one element, which
# deforms homogeneously. A heart wall does not: the stretch varies through the wall, and
# the mesh only approximates its curved surfaces. This tutorial checks both against an
# exact solution of the full three-dimensional problem, and refines the mesh to see the
# error fall.
#
# The geometry is a thick-walled tube, inner radius $R_i$ = 2 cm and outer radius
# $R_o$ = 3 cm, that cannot lengthen along its axis (plane strain). Its fibres run around
# the tube and its sheets point radially. The wall is incompressible, so if every
# cross-section stays circular, a point at radius $R$ moves to $r(R)$ with
#
# $$
# r(R)^2 = r_i^2 + R^2 - R_i^2, \qquad \lambda = \frac{r}{R},
# $$
#
# where $r_i$ is the deformed inner radius and $\lambda$ the circumferential stretch, which
# is the fibre stretch here. Radial equilibrium, with a pressure $p$ on the inner surface
# and none on the outer, then gives
#
# $$
# p = \int_{R_i}^{R_o} \lambda \frac{d\hat\psi}{d\lambda} \frac{R}{r^2} \, dR,
# $$
#
# with $\hat\psi(\lambda)$ the strain energy at that deformation. The class `ThickCylinder`
# of [zero-mech](https://github.com/ComputationalPhysiology/zero-mech) evaluates this
# integral by Gauss-Legendre quadrature, with the strain energy differentiated by sympy.
# So for each load we get the exact $r_i$ to compare the finite element solution with.
#
# By symmetry, a quarter of the tube is enough.

# %%
from mpi4py import MPI

import dolfinx
import matplotlib.pyplot as plt
import numpy as np
import pulse
import scipy.optimize
import sympy as sp
import ufl
import zero_mech

RI, RO, H = 0.02, 0.03, 0.002  # m; H is the slice's length along the axis
ENDO = 1  # the facet tag of the inner surface

# %% [markdown]
# ## The mesh
#
# We mesh a box in $(R, \theta, z)$ and map it to $(R\cos\theta, R\sin\theta, z)$. The
# vertices then lie exactly on the circles. The faces between them are flat, so the
# surfaces are approximated to second order in the mesh size. The slice is one element
# long, since nothing varies along the axis.


# %%
def quarter_annulus(n_r, n_theta):
    """A quarter of the tube, with ``n_r`` cells through the wall and ``n_theta`` around."""
    mesh = dolfinx.mesh.create_box(
        MPI.COMM_WORLD,
        [[RI, 0.0, 0.0], [RO, np.pi / 2, H]],
        [n_r, n_theta, 1],
        dolfinx.mesh.CellType.tetrahedron,
    )
    x = mesh.geometry.x
    R, theta = x[:, 0].copy(), x[:, 1].copy()
    x[:, 0], x[:, 1] = R * np.cos(theta), R * np.sin(theta)
    fdim = mesh.topology.dim - 1
    inner = dolfinx.mesh.locate_entities_boundary(
        mesh,
        fdim,
        lambda x: np.isclose(np.hypot(x[0], x[1]), RI),
    )
    tags = dolfinx.mesh.meshtags(mesh, fdim, inner, np.full_like(inner, ENDO))
    return mesh, tags


mesh, _ = quarter_annulus(4, 8)
fig, ax = plt.subplots(figsize=(4, 4))
for cell in dolfinx.mesh.entities_to_geometry(
    mesh,
    3,
    np.arange(mesh.topology.index_map(3).size_local),
):
    xy = mesh.geometry.x[cell][:, :2]
    for i in range(4):
        for j in range(i + 1, 4):
            ax.plot(*xy[[i, j]].T * 100, color="#16191C", lw=0.4)
ax.set(xlabel="x [cm]", ylabel="y [cm]", aspect="equal", title="Cross-section, 4 x 8")
plt.show()

# %% [markdown]
# ## The finite element problem
#
# The material is pulse's transversely isotropic Holzapfel-Ogden, incompressible, as in
# the earlier tutorials. The fibre and sheet directions are written as UFL expressions of
# the reference coordinates, so they are exactly unit length wherever the form reads
# them. A field interpolated at the nodes would not be between them.
#
# The tube is held by rollers on the two symmetry planes ($x = 0$ and $y = 0$) and on
# both ends of the slice, which is the plane strain of the exact solution. The pressure
# acts on the inner surface as a follower load: pulse's `NeumannBC` applies $-p$ times the
# deformed normal.
#
# The active tension comes from pulse's `ActiveStress` in the `stretch` formulation, the
# same normalization the simcardemsx backends use:
# $\mathbf{P}_a = T_a \mathbf{F} f_0 \otimes f_0 / |\mathbf{F} f_0|$. We prescribe it,
# uniform through the wall, rather than compute it with a cell model. The Land model's
# tension depends on the local stretch, so it would vary through the wall, and the
# reference would need a cell model at every point of its integral.

# %%
QUADRATURE_DEGREE = 4


def rollers(V):
    """No normal displacement on the planes x = 0 and y = 0 and on both ends."""
    mesh = V.mesh
    V0, _ = V.sub(0).collapse()
    zero = dolfinx.fem.Function(V0)
    planes = [
        (0, lambda x: np.isclose(x[0], 0.0)),
        (1, lambda x: np.isclose(x[1], 0.0)),
        (2, lambda x: np.isclose(x[2], 0.0) | np.isclose(x[2], H)),
    ]
    bcs = []
    for i, where in planes:
        facets = dolfinx.mesh.locate_entities_boundary(mesh, mesh.topology.dim - 1, where)
        dofs = dolfinx.fem.locate_dofs_topological((V.sub(i), V0), mesh.topology.dim - 1, facets)
        bcs.append(dolfinx.fem.dirichletbc(zero, dofs, V.sub(i)))
    return bcs


def build(n_r, n_theta):
    """The problem on a mesh of ``n_r`` x ``n_theta`` cells, and its pressure and tension."""
    mesh, tags = quarter_annulus(n_r, n_theta)
    X = ufl.SpatialCoordinate(mesh)
    R = ufl.sqrt(X[0] ** 2 + X[1] ** 2)
    f0 = ufl.as_vector([-X[1] / R, X[0] / R, 0.0])  # around the tube
    s0 = ufl.as_vector([X[0] / R, X[1] / R, 0.0])  # through the wall
    p = dolfinx.fem.Constant(mesh, 0.0)  # Pa
    Ta = dolfinx.fem.Constant(mesh, 0.0)  # Pa
    model = pulse.CardiacModel(
        material=pulse.HolzapfelOgden(
            f0=f0,
            s0=s0,
            **pulse.HolzapfelOgden.transversely_isotropic_parameters(),
        ),
        active=pulse.ActiveStress(
            f0,
            activation=pulse.Variable(Ta, "Pa"),
            formulation=pulse.ActiveStressFormulation.stretch,
        ),
        compressibility=pulse.Incompressible(),
    )
    petsc_options = pulse.StaticProblem.default_parameters()["petsc_options"]
    petsc_options["snes_atol"] = 1e-9
    problem = pulse.StaticProblem(
        model=model,
        geometry=pulse.Geometry(
            mesh=mesh,
            facet_tags=tags,
            markers={"ENDO": (ENDO, 2)},
            metadata={"quadrature_degree": QUADRATURE_DEGREE},
        ),
        bcs=pulse.BoundaryConditions(
            dirichlet=[rollers],
            neumann=[pulse.NeumannBC(traction=pulse.Variable(p, "Pa"), marker=ENDO)],
        ),
        parameters={
            "base_bc": pulse.problem.BaseBC.free,
            "mesh_unit": "m",
            "petsc_options": petsc_options,
        },
    )
    return problem, p, Ta


# %% [markdown]
# We read the deformed inner radius at the inner surface's vertices. They lie on the
# circle $R = R_i$ exactly. The midpoints of quadratic elements lie on the chords between
# them.


# %%
def inner_radius(problem):
    """The mean deformed radius of the inner surface's vertices."""
    mesh = problem.u.function_space.mesh
    vertices = dolfinx.mesh.locate_entities_boundary(
        mesh,
        0,
        lambda x: np.isclose(np.hypot(x[0], x[1]), RI),
    )
    points = mesh.geometry.x[dolfinx.mesh.entities_to_geometry(mesh, 0, vertices).ravel()]
    tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)
    colliding = dolfinx.geometry.compute_colliding_cells(
        mesh,
        dolfinx.geometry.compute_collisions_points(tree, points),
        points,
    )
    cells = [colliding.links(i)[0] for i in range(len(points))]
    u = problem.u.eval(points, cells)
    return float(np.mean(np.hypot(points[:, 0] + u[:, 0], points[:, 1] + u[:, 1])))


# %% [markdown]
# ## The loading
#
# The tube is inflated to 2 kPa (15 mmHg) in five steps. Then, at that pressure, the
# tension rises to 3 kPa in five more, and the tube contracts.
#
# The tension is kept small enough that the wall stays stretched along the fibres,
# $\lambda > 1$ everywhere. Holzapfel-Ogden's fibre term only acts for $\lambda > 1$. If the
# stretch crossed 1 inside the wall, the integrand of the exact solution would have a
# kink there, and its quadrature would converge more slowly. At 2 kPa that happens at a
# tension of about 4 kPa.

# %%
P_END, TA_END = 2000.0, 3000.0  # Pa
steps = np.linspace(0.2, 1.0, 5)
loads = [(float(P_END * s), 0.0) for s in steps] + [(P_END, float(TA_END * s)) for s in steps]


def solve(n_r, n_theta):
    """The deformed inner radius after each load, on an ``n_r`` x ``n_theta`` mesh."""
    problem, p, Ta = build(n_r, n_theta)
    radii = []
    for p_value, Ta_value in loads:
        p.value, Ta.value = p_value, Ta_value
        problem.solve()
        radii.append(inner_radius(problem))
    return np.array(radii)


# %% [markdown]
# ## The exact solution
#
# zero-mech's model is the same material, Holzapfel-Ogden with pulse's parameter values,
# and `StretchActiveStress`, the `stretch` formulation, with its tension `Ta` as a
# parameter. By default, `ThickCylinder` takes axis 0 as circumferential and axis 1 as
# radial, so the fibres are along axis 0 and the sheets along axis 1. For each load we
# find the cavity volume (per unit length, $\pi r_i^2$) at which the tube's pressure is
# the applied one, and read off $r_i$.


# %%
def material_parameters():
    """pulse's Holzapfel-Ogden parameters as plain numbers, in base units (moduli in Pa)."""
    return {
        name: float(value.to_base_units())
        for name, value in pulse.HolzapfelOgden.transversely_isotropic_parameters().items()
    }


tube = zero_mech.thick_wall.ThickCylinder(
    zero_mech.Model(
        material=zero_mech.material.HolzapfelOgden(
            use_heaviside=True,
            f0=sp.Matrix([1, 0, 0]),
            s0=sp.Matrix([0, 1, 0]),
        ),
        compressibility=zero_mech.compressibility.Incompressible(),
        active=zero_mech.StretchActiveStress(f0=sp.Matrix([1, 0, 0])),
    ),
    Ri=RI,
    Ro=RO,
    parameters=material_parameters(),
)


def exact_inner_radius(p_value, Ta_value):
    V = scipy.optimize.brentq(
        lambda V: float(tube.pressure(V, {"Ta": Ta_value})) - p_value,
        0.5 * np.pi * RI**2,
        2.0 * np.pi * RI**2,
        xtol=1e-16,
        rtol=1e-14,
    )
    return float(tube.inner_radius(V)), float(np.min(tube.stretch(V)))


exact, min_stretch = np.array([exact_inner_radius(*load) for load in loads]).T
print(f"smallest fibre stretch in the wall: {min_stretch.min():.3f}")

# %% [markdown]
# ## Refining the mesh

# %%
meshes = [(2, 4), (4, 8), (8, 16), (16, 32)]
radii = {mesh_size: solve(*mesh_size) for mesh_size in meshes}
errors = np.array([np.abs(radii[mesh_size] - exact).max() for mesh_size in meshes])
for (n_r, n_theta), error in zip(meshes, errors):
    print(f"{n_r:3d} x {n_theta:3d} cells: max |r_i - exact| = {error * 1e3:.2e} mm")
orders = np.log(errors[:-1] / errors[1:]) / np.log(2)
print("observed orders:", np.round(orders, 2))

# %% [markdown]
# The exact solution is cheap, so we draw it as a curve and the finest mesh's solution as
# points on it.

# %%
p_curve = np.linspace(0.0, P_END, 41)
Ta_curve = np.linspace(0.0, TA_END, 41)
exact_inflation = [exact_inner_radius(p_value, 0.0)[0] for p_value in p_curve]
exact_contraction = [exact_inner_radius(P_END, Ta_value)[0] for Ta_value in Ta_curve]
pressure = np.array([p_value for p_value, _ in loads])
tension = np.array([Ta_value for _, Ta_value in loads])
inflation, contraction = slice(0, 5), slice(5, 10)
finest = radii[meshes[-1]]
cells = np.array([n_r for n_r, _ in meshes])

fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
axes[0].plot(np.array(exact_inflation) * 1e3, p_curve / 1e3, color="#16191C", lw=2.4, label="exact")
axes[0].plot(
    finest[inflation] * 1e3,
    pressure[inflation] / 1e3,
    "o",
    color="#12706B",
    label="16 x 32 cells",
)
axes[0].set(xlabel="$r_i$ [mm]", ylabel="$p$ [kPa]", title="Inflation")
axes[0].legend(frameon=False)
axes[1].plot(Ta_curve / 1e3, np.array(exact_contraction) * 1e3, color="#16191C", lw=2.4)
axes[1].plot(tension[contraction] / 1e3, finest[contraction] * 1e3, "o", color="#12706B")
axes[1].set(xlabel="$T_a$ [kPa]", ylabel="$r_i$ [mm]", title=f"Contraction at {P_END / 1e3:g} kPa")
axes[2].loglog(cells, errors * 1e3, "o-", color="#12706B", label="max error")
axes[2].loglog(
    cells,
    errors[0] * 1e3 * (cells / cells[0]) ** -2.0,
    "k:",
    lw=1,
    label="second order",
)
axes[2].set_xticks(cells, [str(n) for n in cells])
axes[2].minorticks_off()
axes[2].set(xlabel="cells through the wall", ylabel="max $|r_i - r_i^{exact}|$ [mm]")
axes[2].set_title("Convergence")
axes[2].legend(frameon=False)
fig.tight_layout()
plt.show()

# %% [markdown]
# The error falls at second order as the mesh is refined, which is what quadratic
# elements on a mesh whose curved surfaces are flat between the vertices should give.
# That holds through the active contraction too, so the `stretch` active stress is
# evaluated consistently on a deforming, curved wall.
#
# Two things are left out. The tension here is prescribed and uniform. A cell model would
# make it depend on the local stretch, and the exact solution would then have to step a
# cell model at every point of its integral. zero-mech accepts a parameter that varies
# through the wall, so that is possible, but it is not done here. And a ventricle has no
# exact solution: zero-mech's `ThickSphere` is exact for a sphere, but only for a passive
# wall, since no single fibre direction is tangent to a sphere everywhere.
#
# The [next tutorial](07_reaction_eikonal_slab.ipynb) returns to electrophysiology. It
# replaces the monodomain model by the reaction-eikonal model on the Niederer benchmark
# slab, and couples it to mechanics.
