# %% [markdown]
# # Checking against an independent 0D solution
#
# Tutorial 3 compared the coupling schemes with each other, and with a run at a smaller
# time step. That shows the schemes converge, but not that they converge to the solution
# of the problem we meant to pose. This tutorial checks that against an independent
# reference: [zero-mech](https://github.com/ComputationalPhysiology/zero-mech), a library
# of 0D mechanics whose stresses are written as sympy strain energies.
#
# The element of tutorial 3 is held by rollers on the three faces through the origin and
# is otherwise free, and everything acting on it is uniform. So it deforms homogeneously,
# on any mesh:
#
# $$
# \mathbf{F} = \operatorname{diag}(\lambda_1, \lambda_2, \lambda_3), \qquad
# \lambda_3 = \frac{1}{\lambda_1 \lambda_2},
# $$
#
# with $\lambda_1$ the fibre stretch $\lambda$. Its finite element solution must therefore
# also solve a 0D problem: the three normal stresses vanish,
# $P_{11} = P_{22} = P_{33} = 0$. zero-mech supplies that stress, by differentiating its
# own strain energies, independently of pulse's UFL forms.

# %%
import types
from pathlib import Path

from mpi4py import MPI

import dolfinx
import gotranx
import matplotlib.pyplot as plt
import numpy as np
import pulse
import scipy.optimize
import sympy as sp
import zero_mech

from simcardemsx.backends import GeneratedActivation
from simcardemsx.ode_model import load_ode_modules

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
odefile = HERE.parent / "odefiles" / "ToRORd_dynCl_endo_caisplit.ode"
modules = load_ode_modules(odefile, HERE / "generated_odes" / odefile.stem)

# %% [markdown]
# ## The active stress in zero-mech
#
# simcardemsx adds the active tension along the fibres with the normalization of
# Regazzoni & Quarteroni, $\mathbf{P}_a = T_a \mathbf{F} f_0 \otimes f_0 / |\mathbf{F} f_0|$
# (tutorial 3). In zero-mech, that is `StretchActiveStress`, the strain energy
# $T_a(\lambda_f - 1)$ with $\lambda_f = |\mathbf{F} f_0|$. zero-mech's `ActiveStress` is
# the older form, $\frac{1}{2} T_a (I_{4f} - 1)$, whose stress $T_a \mathbf{F} f_0
# \otimes f_0$ is larger by a factor of the stretch. On our diagonal $\mathbf{F}$:

# %%
f0 = sp.Matrix([1, 0, 0])
F = sp.Matrix(3, 3, sp.symbols("F_11 F_12 F_13 F_21 F_22 F_23 F_31 F_32 F_33"))
l1, l2, l3 = sp.symbols("lambda_1 lambda_2 lambda_3", positive=True)
diagonal = {F[i, j]: sp.diag(l1, l2, l3)[i, j] for i in range(3) for j in range(3)}
for active in (zero_mech.StretchActiveStress(f0=f0), zero_mech.ActiveStress(f0=f0)):
    psi = active.strain_energy(F)
    P11 = sp.simplify(sp.diff(psi, F[0, 0]).subs(diagonal))
    print(f"{type(active).__name__:20s} psi = {psi},  P_11 = {P11}")

# %% [markdown]
# With `StretchActiveStress`, the fibres pull with exactly $T_a$, whatever their stretch.
#
# The passive material is pulse's transversely isotropic Holzapfel-Ogden, incompressible,
# with fibres along $x$ and sheets along $y$, as in tutorial 3. We write it in zero-mech
# with pulse's parameter values, which are in pascals, and turn the stress into a numpy
# function of $(\lambda_1, \lambda_2, p, T_a)$.

# %%
PA_PER_KPA = 1000.0  # the cell model's tension is in kPa, pulse's stresses in Pa


def homogeneous_stress(active):
    """``(λ1, λ2, p, Ta [Pa]) -> [P11, P22, P33]`` in Pa, with zero-mech's ``active``."""
    model = zero_mech.Model(
        material=zero_mech.material.HolzapfelOgden(
            use_heaviside=True,
            f0=sp.Matrix([1, 0, 0]),
            s0=sp.Matrix([0, 1, 0]),
        ),
        compressibility=zero_mech.compressibility.Incompressible(),
        active=active,
    )
    parameters = {
        model[name]: float(value.to_base_units())
        for name, value in pulse.HolzapfelOgden.transversely_isotropic_parameters().items()
    }
    experiment = zero_mech.experiments.MechanicalExperiment(F11=l1, F22=l2, F33=1 / (l1 * l2))
    P = model.first_piola_kirchhoff(experiment.F).xreplace(parameters)
    return sp.lambdify((l1, l2, model["p"], model["Ta"]), [P[i, i] for i in range(3)], "numpy")


stretch_stress = homogeneous_stress(zero_mech.StretchActiveStress(f0=sp.Matrix([1, 0, 0])))

# %% [markdown]
# ## The contraction model in 0D
#
# The tension comes from the Land model in the `mechanics` component of the Ca$_i$
# split. In the finite element run, `GeneratedActivation` compiles that component to UFL.
# For the 0D side we compile the same component to numpy, with the same
# generalized Rush-Larsen step, using gotranx's own generator.

# %%
ode = gotranx.load_ode(odefile)
mechanics_component = ode.get_component("mechanics")
code = gotranx.cli.gotran2py.get_code(
    mechanics_component.to_ode(),
    scheme=[gotranx.schemes.Scheme.generalized_rush_larsen],
    missing_values=(ode - mechanics_component).missing_variables,
)
land = types.ModuleType("land_numpy")
exec(code, land.__dict__)
print("states:", list(land.state), "\ninputs:", land.missing)

# %% [markdown]
# Under `scheme="monolithic"`, the backend takes one step of the Land model *inside*
# Newton, at the stretch $\lambda$ Newton is iterating on and the stretch rate
# $(\lambda - \lambda_n)/\Delta t$ (tutorial 3). The 0D twin does the same inside a root
# find. Each step looks for $(\lambda_1, \lambda_2, p)$ at which the stress vanishes,
# where the tension is the Land model advanced from the last accepted states at the trial
# $\lambda_1$. It then accepts the states of the solution.


# %%
class Homogeneous0D:
    """The Land model and zero-mech's stress on a homogeneous element, coupled monolithically."""

    def __init__(self, stress):
        self.stress = stress
        self.states = land.init_state_values()
        self.parameters = land.init_parameter_values()
        self.inputs = np.zeros(len(land.missing))
        self.x = np.array([1.0, 1.0, 0.0])  # λ1, λ2, p
        self.lmbda = 1.0

    def advance(self, lmbda, t_n, dt):
        """The Land states and tension [kPa] one step on, at the stretch ``lmbda``."""
        parameters = self.parameters.copy()
        parameters[land.parameter_index("lmbda")] = lmbda
        parameters[land.parameter_index("dLambda")] = (lmbda - self.lmbda) / dt
        states = land.generalized_rush_larsen(self.states.copy(), t_n, dt, parameters, self.inputs)
        Ta = land.monitor_values(t_n, states, parameters, self.inputs)[land.monitor_index("Ta")]
        return states, Ta

    def residual(self, x, t_n, dt):
        _, Ta = self.advance(x[0], t_n, dt)
        return np.asarray(self.stress(x[0], x[1], x[2], PA_PER_KPA * Ta), dtype=float)

    def step(self, t_n, dt, cai):
        self.inputs[land.missing["cai"]] = cai
        solution = scipy.optimize.root(
            self.residual,
            self.x,
            args=(t_n, dt),
            method="hybr",
            options={"xtol": 1e-13},
        )
        assert np.abs(self.residual(solution.x, t_n, dt)).max() < 1e-7  # Pa
        self.x = solution.x
        self.states, Ta = self.advance(self.x[0], t_n, dt)
        self.lmbda = float(self.x[0])
        return self.lmbda, Ta


# %% [markdown]
# ## The finite element run
#
# This is tutorial 3's element, under the monolithic scheme, with the same calcium
# transient.

# %%
QUADRATURE_DEGREE = 2


def rollers(V):
    """u_i = 0 on the face x_i = 0, for i = 0, 1, 2."""
    mesh = V.mesh
    V0, _ = V.sub(0).collapse()
    zero = dolfinx.fem.Function(V0)
    bcs = []
    for i in range(3):
        facets = dolfinx.mesh.locate_entities_boundary(
            mesh,
            mesh.topology.dim - 1,
            lambda x, i=i: np.isclose(x[i], 0.0),
        )
        dofs = dolfinx.fem.locate_dofs_topological((V.sub(i), V0), mesh.topology.dim - 1, facets)
        bcs.append(dolfinx.fem.dirichletbc(zero, dofs, V.sub(i)))
    return bcs


def build():
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    f0 = dolfinx.fem.Constant(mesh, np.array([1.0, 0.0, 0.0]))
    s0 = dolfinx.fem.Constant(mesh, np.array([0.0, 1.0, 0.0]))
    backend = GeneratedActivation(
        modules.mechanics,
        mesh,
        f0,
        quadrature_degree=QUADRATURE_DEGREE,
        scheme="monolithic",
    )
    model = pulse.CardiacModel(
        material=pulse.HolzapfelOgden(
            f0=f0,
            s0=s0,
            **pulse.HolzapfelOgden.transversely_isotropic_parameters(),
        ),
        active=backend,
        compressibility=pulse.Incompressible(),
    )
    petsc_options = pulse.StaticProblem.default_parameters()["petsc_options"]
    petsc_options["snes_atol"] = 1e-9
    problem = pulse.StaticProblem(
        model=model,
        geometry=pulse.Geometry(mesh=mesh, metadata={"quadrature_degree": QUADRATURE_DEGREE}),
        bcs=pulse.BoundaryConditions(dirichlet=[rollers]),
        parameters={"base_bc": pulse.problem.BaseBC.free, "petsc_options": petsc_options},
    )
    return problem, backend


def calcium(t):
    """Ca_i in mM: 0.1 µM at rest, peaking at 1 µM at t = 25 ms."""
    tau = max(t - 5.0, 0.0)
    return 1e-4 + 9e-4 * (tau / 20.0) * np.exp(1.0 - tau / 20.0)


DT, T_END = 1.0, 60.0
times = DT * np.arange(1, round(T_END / DT) + 1)

problem, backend = build()
fem: dict[str, list[float]] = {"lmbda": [], "Ta": []}
for t_next in times:
    backend.inputs["cai"].x.array[:] = calcium(t_next)
    backend.begin_step(t_next - DT, DT)
    problem.solve()
    backend.post_solve()
    fem["lmbda"].append(backend.lmbda_prev.x.array.mean())
    fem["Ta"].append(backend.tension_kPa.x.array.mean())
print(f"uniform: λ varies by {np.ptp(backend.lmbda_prev.x.array):.1e} over the element")

# %% [markdown]
# ## Both solutions


# %%
def run_0d(stress):
    zero_d = Homogeneous0D(stress)
    lmbda, Ta = zip(*(zero_d.step(t_next - DT, DT, calcium(t_next)) for t_next in times))
    return np.array(lmbda), np.array(Ta)


lmbda_0d, Ta_0d = run_0d(stretch_stress)
difference = np.abs(np.array(fem["lmbda"]) - lmbda_0d)
print(f"max |λ_FEM - λ_0D| = {difference.max():.1e}")

fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
axes[0].plot(times, fem["Ta"], color="#16191C", lw=2.4, label="finite elements")
axes[0].plot(times, Ta_0d, color="#12706B", lw=1.4, ls="--", label="zero-mech")
axes[0].set(ylabel="$T_a$ [kPa]", title="Active tension")
axes[1].plot(times, fem["lmbda"], color="#16191C", lw=2.4, label="finite elements")
axes[1].plot(times, lmbda_0d, color="#12706B", lw=1.4, ls="--", label="zero-mech")
axes[1].set(ylabel="$\\lambda$ [-]", title="Fibre stretch")
axes[1].legend(frameon=False)
axes[2].semilogy(times, np.maximum(difference, 1e-17), color="#12706B")
axes[2].set(ylabel="$|\\lambda_{FEM} - \\lambda_{0D}|$", title="Difference", ylim=(1e-17, 1e-2))
for ax in axes:
    ax.set_xlabel("time [ms]")
fig.tight_layout()
plt.show()

# %% [markdown]
# The two agree to round-off. That covers more than the passive material:
# the active stress's normalization, the unit conversion from kPa to Pa, and the
# monolithic coupling, in which Newton differentiates through the Land model's step.
#
# ## Would the check notice a mistake?
#
# An agreement to round-off says little unless a mistake would break it. Here is one:
# the 0D side with the older `invariant` active stress, which is off by a factor
# $\lambda$ along the fibres.

# %%
lmbda_invariant, _ = run_0d(homogeneous_stress(zero_mech.ActiveStress(f0=sp.Matrix([1, 0, 0]))))
miss = np.abs(np.array(fem["lmbda"]) - lmbda_invariant)
print(f"max |λ_FEM - λ_0D| with the invariant form: {miss.max():.1e}")

fig, ax = plt.subplots(figsize=(5.5, 3.6))
ax.semilogy(times, np.maximum(difference, 1e-17), color="#12706B", label="StretchActiveStress")
ax.semilogy(times, np.maximum(miss, 1e-17), color="#B33F0C", label="ActiveStress (invariant)")
ax.set(xlabel="time [ms]", ylabel="$|\\lambda_{FEM} - \\lambda_{0D}|$", ylim=(1e-17, 1e-2))
ax.legend(frameon=False)
fig.tight_layout()
plt.show()

# %% [markdown]
# The wrong stress misses by about ten orders of magnitude more than round-off, as soon as the
# tension rises.
#
# ## What this does and does not check
#
# Both sides take the same time step, the same generalized Rush-Larsen step of the same
# Land model, by design. So the comparison checks the *spatial* problem and the coupling
# inside it, against the discrete problem in time. Whether the time stepping converges
# is the question tutorial 3 answered, with its run at a smaller time step.
#
# The test suite runs this comparison (`tests/test_zero_mech_reference.py`) for both the
# Ca$_i$ and the zeta split. It also runs a second one, on the coupled loop of tutorial 4
# with an EP solver: EP on a uniform mesh, stimulated everywhere at once, stays uniform
# too. So a 0D twin of the whole controller step must match $\lambda$, $v$, Ca$_i$ and
# $T_a$, which it does to about $10^{-12}$ of their largest values.
#
# The [next tutorial](06_thick_cylinder.ipynb) drops the homogeneity: a thick-walled tube,
# whose stretch varies through the wall, against its exact solution.
