# %% [markdown]
# # Reaction-eikonal electromechanics on the Niederer slab
#
# The monodomain model of tutorial 4 resolves the upstroke of the action potential in
# space, and that needs a fine mesh. On a coarse one the wave travels too slowly, or not at
# all. A whole heart at 0.2 mm has millions of nodes, so many electromechanics studies
# replace the monodomain model by the **reaction-eikonal** (R-E) model of Neic et al.
# (*J Comput Phys* 2017).
#
# R-E separates *when* a point activates from *how* its cells respond. An eikonal equation
#
# $$
# \sqrt{\nabla t_a^T\,\mathbf{V}\,\nabla t_a} = 1, \qquad
# \mathbf{V} = v_l^2\, f_0 f_0^T + v_t^2\,(s_0 s_0^T + n_0 n_0^T),
# $$
#
# gives the activation time $t_a$ of every point from the conduction velocities along
# ($v_l$) and across ($v_t$) the fibres. It is solved once, before the run, and its
# solution does not slow down on a coarse mesh, since the velocities are its input. The
# cell model is then driven by a **foot current**: from $t_a$ on, the membrane
# potential gets an exponentially rising push,
#
# $$
# \frac{dv}{dt} = \frac{A}{\tau_F}\, e^{(t - t_a)/\tau_F}, \qquad
# 0 \le t - t_a \le T_{foot}, \quad v < V_{th},
# $$
#
# with $A$ = 0.91 mV and $\tau_F$ = 0.25 ms, until the cell's own sodium current takes
# over. There are two variants:
#
# - **R-E+** keeps the monodomain PDE, without its stimulus. Diffusion still shapes the
#   upstroke. Neic et al. designed the foot as a latest time of activation, and most
#   points do activate earlier than the foot alone would make them, pulled up by their
#   neighbours, though a point beside resting tissue can be slightly later.
# - **R-E−** drops the PDE. Every node is an isolated cell, fired by its own foot current.
#
# For electromechanics, what matters is that every point gets its calcium transient at
# the right time, and R-E gives that at a fraction of the cost. Strocchi et al. (*PLoS
# Comput Biol* 2023) drive ToR-ORd and Land with R-E− in a four-chamber heart, with
# activation times that do not depend on the mechanics, as here. Petras et al. (*Comput
# Biol Med* 2023) drive a ten Tusscher-Panfilov model and Land with R-E+.
#
# This tutorial uses the slab of the Niederer benchmark (Niederer et al., *Phil Trans R Soc
# A* 2011), a 20 × 7 × 3 mm block of tissue with fibres along $x$, stimulated in one corner.
# It
#
# 1. composes the benchmark's cell model with Land's contraction model, both fetched from
#    a model registry;
# 2. calibrates the eikonal on a fine monodomain run;
# 3. compares R-E+ and R-E− with the monodomain model on coarse meshes, against a fine
#    monodomain reference;
# 4. couples each of the three to mechanics.

# %%
import time
from pathlib import Path

from mpi4py import MPI

import beat
import dolfinx
import gotranx
import matplotlib.pyplot as plt
import numpy as np
import odezoo
import pulse
from beat.activation import ActivationTracker
from beat.conduction_velocity import measure_conduction_velocity
from beat.eikonal import Source, solve_eikonal
from beat.reaction_eikonal import ExponentialFoot, FootCurrentODESolver

from simcardemsx.averaging import make_averager
from simcardemsx.backends import GeneratedActivation
from simcardemsx.controller import SimulationController, dolfin_ode_solver
from simcardemsx.ode_model import load_ode_modules
from simcardemsx.transfer_plan import resolve

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
ODEFILES = HERE.parent / "odefiles"

tt06_file = odezoo.get("tentusscher_2006_epi", version="pmr-5dc4239")
land_file = odezoo.get("land_2017", version="github-6e5a5aa")
print("TT06:", tt06_file.name)
print("Land:", land_file.name)

# %% [markdown]
# ## The cell model
#
# The benchmark's cell model is the ten Tusscher-Panfilov 2006 epicardial model (TT06;
# ten Tusscher & Panfilov, *Am J Physiol Heart Circ Physiol* 2006). Here it drives the Land
# contraction model of tutorial 1. Both come from
# [odezoo](https://github.com/ComputationalPhysiology/odezoo), a registry of cell models:
# `odezoo.get` downloads a model once, by a pinned version, checks its sha256, and returns
# the path of the cached file. TT06 is the CellML file of the Physiome Model Repository,
# and Land the `.ode` file of the
# [crossbridge](https://github.com/ComputationalPhysiology/crossbridge) package.
#
# `gotranx.compose` joins them following a *recipe*. Each part is a file with its own time
# unit (TT06 runs in ms, Land in s), and the composed model runs in ms. The part names
# matter: the Land part is called `mechanics`, which is the component simcardemsx puts on
# the mechanics side, so the composed model is already split. Two of Land's names are
# renamed, since TT06 has an `h` (a sodium-channel gate) and simcardemsx sets a
# `dLambda` of its own.

# %%
glue_file = ODEFILES / "tt06_land.glue.ode"
recipe = {
    "time_unit": "ms",
    "glue": glue_file.name,
    "parts": [
        {"name": "ep", "file": str(tt06_file), "time_unit": "ms"},
        {
            "name": "mechanics",
            "file": str(land_file),
            "time_unit": "s",
            "rename": {"h": "h_land", "dLambda": "dLambda_land"},
        },
    ],
}
print(glue_file.read_text())

# %% [markdown]
# The *glue* file says how the parts meet. It gives the mechanics component the stretch
# `lmbda` and its rate `dLambda`, which simcardemsx sets from the displacement, and it
# replaces Land's three inputs: its calcium (µM) by TT06's `Ca_i` (mM), and the sarcomere
# length and its rate by the stretch. Nothing goes back to TT06, which keeps its own
# lumped calcium buffering: the coupling is one-way, as in Petras et al., whose
# ten Tusscher-Panfilov model drives Land without troponin feeding back.
#
# The composed model is saved as an `.ode` file and generated into the two modules, as
# for the files in `examples/odefiles`. gotranx may warn that it drops a guard on a
# concentration while composing, which is harmless here.

# %%
ode = gotranx.compose(recipe, base_dir=ODEFILES)
odefile = HERE / "generated_odes" / "tt06_land" / "tt06_land.ode"
odefile.parent.mkdir(parents=True, exist_ok=True)
ode.save(odefile)
modules = load_ode_modules(odefile, odefile.parent)
ep_module, mech_module = modules.ep, modules.mechanics
print("mechanics states:", list(mech_module.state))

crossings = resolve(ep_module, mech_module)
print("EP → mechanics:", crossings.forward)
print("mechanics → EP:", crossings.backward)
print("λ sent to EP:  ", crossings.stretch_to_ep)

# %% [markdown]
# ## Calibrating the eikonal
#
# The eikonal needs three numbers: the conduction velocity along the fibres, $CV_L$, the
# velocity across them, $CV_T$, and the time at which its sources start. All three are
# measured with the same cell model and tissue as the reference, so that R-E propagates as
# the reference does.
#
# ### Conduction velocities
#
# `beat.conduction_velocity.measure_conduction_velocity` runs a plane wave along or across
# the fibres of a thin structured slab, at a given mesh size and time step, and fits the
# velocity to the activation times, as openCARP's tuneCV does. It is run at the mesh size
# of the fine reference below, 0.2 mm, with the benchmark's conductivities, $\chi$ and
# $C_m$, and the activation time is the 0 mV crossing, the benchmark's definition. TT06's
# own stimulus is switched off here and everywhere below: the stimulus is applied to the
# tissue.

# %%
DT_EP = 0.05  # ms
DX_REF = 0.2  # mm
ureg = beat.units.ureg
conductivities = beat.conductivities.default_conductivities("Niederer")
C_m = ureg.Quantity(1.0, "uF/cm**2")
STIM_AMPLITUDE = ureg.Quantity(50_000.0, "uA/cm**3")

iV = ep_module.state_index("V")
ep_parameters = ep_module.init_parameter_values(stim_amplitude=0.0)

velocity = {}
for direction in ("longitudinal", "transverse"):
    velocity[direction] = measure_conduction_velocity(
        fun=ep_module.generalized_rush_larsen,
        init_states=ep_module.init_state_values(),
        parameters=ep_parameters,
        v_index=iV,
        conductivities=conductivities,
        dx=DX_REF,
        dt=DT_EP,
        direction=direction,
        C_m=C_m,
        # A current per unit area of membrane: the benchmark's, per unit volume, over chi
        amplitude=(STIM_AMPLITUDE / conductivities["chi"]).to("uA/cm**2"),
        threshold=0.0,
    )
CV_L, CV_T = velocity["longitudinal"], velocity["transverse"]
print(f"CV_L = {CV_L:.4f} mm/ms, CV_T = {CV_T:.4f} mm/ms")

# %% [markdown]
# ### The slab
#
# The benchmark's slab, at any mesh size, with the stimulated cells tagged: those with
# every vertex in the 1.5 mm corner cube. At 0.5 mm that is the whole cube, at 0.2 mm a
# 1.4 mm cube, and at 1.0 mm a 1 mm one. The nine points P1–P9 are the benchmark's: the
# eight corners, from the stimulated one (P1) to the far one (P8), and the centre (P9).

# %%
LX, LY, LZ = 20.0, 7.0, 3.0  # mm
STIM_SIZE = 1.5  # mm
POINTS = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, LY, 0.0],
        [LX, 0.0, 0.0],
        [LX, LY, 0.0],
        [0.0, 0.0, LZ],
        [0.0, LY, LZ],
        [LX, 0.0, LZ],
        [LX, LY, LZ],
        [LX / 2, LY / 2, LZ / 2],
    ],
)
P1, P3, P8 = 0, 2, 7  # rows of POINTS


def slab(dx):
    """The benchmark slab at mesh size dx, with the stimulated cells tagged 1."""
    geo = beat.geometry.get_3D_slab_geometry(MPI.COMM_WORLD, Lx=LX, Ly=LY, Lz=LZ, dx=dx)
    corner = dolfinx.mesh.locate_entities(
        geo.mesh,
        3,
        lambda x: (
            (x[0] <= STIM_SIZE + 1e-10) & (x[1] <= STIM_SIZE + 1e-10) & (x[2] <= STIM_SIZE + 1e-10)
        ),
    )
    tags = dolfinx.mesh.meshtags(geo.mesh, 3, corner, np.full(corner.size, 1, dtype=np.int32))
    return geo, tags


# %% [markdown]
# `ep_solver` builds the three kinds of EP on a slab. The cell model is stepped at the
# nodes of a P1 space, as in tutorial 4. For monodomain, the stimulus is the benchmark's,
# 50 000 µA/cm³ for 2 ms in the tagged cells. For R-E, beat's `FootCurrentODESolver` wraps
# the ODE solver and adds the foot current at each node's $t_a$, and the PDE is kept
# without a stimulus (R-E+) or dropped (R-E−).


# %%
def ep_solver(mode, geo, tags, t_a=None, solver_type="direct"):
    """EP on the slab: "monodomain", or R-E with ("re+") or without ("re-") the PDE."""
    mesh = geo.mesh
    time_ep = dolfinx.fem.Constant(mesh, 0.0)
    I_s = None
    if mode == "monodomain":
        I_s = beat.stimulation.define_stimulus(
            mesh=mesh,
            chi=conductivities["chi"],
            time=time_ep,
            subdomain_data=tags,
            marker=1,
            mesh_unit="mm",
            amplitude=STIM_AMPLITUDE,
            start=0.0,
            duration=2.0,
        )
    pde = beat.MonodomainModel(
        time=time_ep,
        mesh=mesh,
        M=beat.conductivities.define_conductivity_tensor(f0=geo.f0, **conductivities),
        I_s=I_s,
        C_m=C_m.to("uF/mm**2").magnitude,
        params=beat.MonodomainModel.default_parameters(solver_type),
    )
    v_ode = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("P", 1)))
    num_points = v_ode.x.array.size
    ode = beat.odesolver.DolfinODESolver(
        v_ode=v_ode,
        v_pde=pde.state,
        fun=ep_module.generalized_rush_larsen,
        init_states=np.tile(ep_module.init_state_values()[:, None], (1, num_points)),
        parameters=np.tile(ep_parameters[:, None], (1, num_points)),
        num_states=len(ep_module.state),
        v_index=iV,
    )
    if mode == "monodomain":
        return beat.SplittingSolver(pde=pde, ode=ode, theta=1.0)
    foot = FootCurrentODESolver(ode, t_a, ExponentialFoot())
    return beat.SplittingSolver(pde=pde if mode == "re+" else None, ode=foot, theta=1.0)


# %% [markdown]
# `run_ep` runs EP alone and records when each node first rises through 0 mV
# (`beat.activation.ActivationTracker`, interpolated linearly in time). The activation
# times at P1–P9 are interpolated in space from the node values, since P9 is not a node
# of every mesh.


# %%
def run_ep(mode, dx, t_end, solver_type="direct"):
    geo, tags = slab(dx)
    t_a = None if mode == "monodomain" else eikonal(geo, tags)
    solver = ep_solver(mode, geo, tags, t_a=t_a, solver_type=solver_type)
    tracker = ActivationTracker(solver.v, method="threshold", threshold=0.0)
    start = time.perf_counter()
    for k in range(round(t_end / DT_EP)):
        t = k * DT_EP
        solver.step((t, t + DT_EP))
        tracker.update((k + 1) * DT_EP)
    activation = tracker.t_a
    return {
        "geo": geo,
        "tags": tags,
        "activation": activation,
        "eikonal": t_a,
        "at_points": beat.utils.PointEvaluator(geo.mesh, POINTS)(activation),
        "seconds": time.perf_counter() - start,
    }


# %% [markdown]
# ### The fine reference and the stimulus delay
#
# The reference is the monodomain model at 0.2 mm, which takes about a minute, with an
# iterative solver (conjugate gradients with algebraic multigrid) for the PDE. It runs to
# 45 ms, past the last activation. Its stimulated nodes reach 0 mV some time after the
# stimulus starts, and the mean of those times, `STIM_DELAY`, is when the eikonal's
# sources must reach 0 mV too.

# %%
reference = run_ep("monodomain", DX_REF, 45.0, solver_type="iterative")
activation_ref = reference["activation"]
stimulated = np.unique(
    activation_ref.function_space.dofmap.list[reference["tags"].find(1)].ravel(),
)
STIM_DELAY = float(np.mean(activation_ref.x.array[stimulated]))
print(f"reference: {reference['seconds']:.0f} s")
print(f"STIM_DELAY = {STIM_DELAY:.3f} ms, the mean over {stimulated.size} stimulated nodes")

# %% [markdown]
# ### The foot's latency
#
# A node fired at $t_a$ does not reach 0 mV at $t_a$: the foot takes about a millisecond
# to lift it to the sodium current's threshold, and the upstroke takes a little more. So
# a source started at `STIM_DELAY` would reach 0 mV one latency late, and every R-E time
# would lag the reference by that much. One TT06 cell with the foot at $t_a$ = 1 ms (on a
# time step, and with the foot added after each ODE step, as `FootCurrentODESolver` does)
# measures that latency, and the sources start at `STIM_DELAY - FOOT_LATENCY`. Every method
# is then compared by the same quantity, the 0 mV time, against the reference.


# %%
def first_crossing(t, v, threshold=0.0):
    """The first time v rises through threshold, linear in time (nan if it never does)."""
    i = np.flatnonzero((v[:-1] < threshold) & (v[1:] >= threshold))
    if i.size == 0:
        return np.nan
    i = i[0]
    return t[i] + (threshold - v[i]) / (v[i + 1] - v[i]) * (t[i + 1] - t[i])


T_A = 1.0  # ms
foot = ExponentialFoot()
states = ep_module.init_state_values()
t_cell, v_cell = [0.0], [states[iV]]
for k in range(round(10.0 / DT_EP)):
    states = ep_module.generalized_rush_larsen(states, k * DT_EP, DT_EP, ep_parameters)
    states[iV] += DT_EP * foot.rate((k + 1) * DT_EP - T_A, states[iV])
    t_cell.append((k + 1) * DT_EP)
    v_cell.append(states[iV])
FOOT_LATENCY = first_crossing(np.array(t_cell), np.array(v_cell)) - T_A
SOURCE_TIME = STIM_DELAY - FOOT_LATENCY
print(f"FOOT_LATENCY = {FOOT_LATENCY:.3f} ms")
print(f"the eikonal's sources start at {SOURCE_TIME:.3f} ms")


# %% [markdown]
# The eikonal's sources are the stimulated cells' vertices, all at `SOURCE_TIME`, and
# `beat.eikonal.solve_eikonal` solves it with the two velocities, $CV_T$ both across the
# fibres and normal to the sheets.


# %%
def eikonal(geo, tags):
    return solve_eikonal(
        geo.mesh,
        (CV_L, CV_T, CV_T),
        (geo.f0, geo.s0, geo.n0),
        [Source((tags, 1), SOURCE_TIME)],
    )


# %% [markdown]
# ## Activation, EP only
#
# Monodomain, R-E+ and R-E− at 0.5 mm and 1.0 mm. Monodomain runs to 65 ms, since it
# is slower on these meshes, and R-E to 60 ms, well past its last activation. These take
# seconds each.

# %%
MODES = {"monodomain": "monodomain", "re+": "R-E+", "re-": "R-E−"}
runs = {}
for dx in (0.5, 1.0):
    for mode in MODES:
        runs[mode, dx] = run_ep(mode, dx, 65.0 if mode == "monodomain" else 60.0)
        print(f"{MODES[mode]:>10}, {dx} mm: {runs[mode, dx]['seconds']:.1f} s")

# %% [markdown]
# The activation times at P1–P9, and each method's error against the reference:

# %%
reference_at = reference["at_points"]


def row(label, values, signed=False):
    """One line of a table; "—" where a node of the point's cell never activated."""
    cells = ["—" if np.isnan(v) else f"{v:+.2f}" if signed else f"{v:.2f}" for v in values]
    print(f"{label:<24}" + "".join(f"{cell:>8}" for cell in cells))


print(f"{'activation (ms)':<24}" + "".join(f"{f'P{i}':>8}" for i in range(1, 10)))
row(f"reference, {DX_REF} mm", reference_at)
for (mode, dx), run in runs.items():
    row(f"{MODES[mode]}, {dx} mm", run["at_points"])
print()
print(f"{'error (ms)':<24}" + "".join(f"{f'P{i}':>8}" for i in range(1, 10)))
for (mode, dx), run in runs.items():
    row(f"{MODES[mode]}, {dx} mm", run["at_points"] - reference_at, signed=True)
print()
for (mode, dx), run in runs.items():
    never = int(np.count_nonzero(np.isnan(run["activation"].x.array)))
    if never:
        size = run["activation"].x.array.size
        print(f"{MODES[mode]}, {dx} mm: {never} of {size} nodes never reach 0 mV")

# Where the monodomain model at 1.0 mm does activate
run = runs["monodomain", 1.0]
space = run["activation"].function_space
activated = ~np.isnan(run["activation"].x.array)
on_edge = np.all(np.isclose(space.tabulate_dof_coordinates()[:, 1:], 0.0), axis=1)
corner_nodes = np.unique(space.dofmap.list[run["tags"].find(1)].ravel())
print(
    f"monodomain, 1.0 mm: {np.count_nonzero(activated & on_edge)} of the "
    f"{np.count_nonzero(activated)} nodes that activate are on the edge y = z = 0, which has "
    f"{np.count_nonzero(on_edge)}; {np.count_nonzero(activated[corner_nodes])} of the "
    f"{corner_nodes.size} stimulated nodes activate",
)

# %% [markdown]
# This is the main finding of Neic et al. On a coarse mesh the monodomain model is late,
# the more so the further the wave has to travel across the fibres: at 0.5 mm it reaches
# P2 and P6 about 20 ms after the reference. At 1.0 mm the wave does not enter the slab.
# Six of the eight stimulated nodes never reach 0 mV, and the nodes that do activate are
# one row, the whole edge $y = z = 0$ from P1 to P3. A wave runs along that edge while
# the tissue beside it stays at rest, which is an artefact of this coarse mesh, not
# propagation through the tissue. The tables show "—" for it even at P1 and P3, which do
# activate, because a point's value is interpolated from the nodes of its cell, and some
# of those never activate. R-E is within a few milliseconds at 0.5 mm and within 10 ms at
# 1.0 mm, since the eikonal's speed is its input and does not depend on the mesh.
#
# These are the numbers this tutorial checks. The tolerances are twice the largest errors
# measured when the tutorial was written.

# %%
TOLERANCE = {("re+", 0.5): 4.5, ("re-", 0.5): 11.0, ("re+", 1.0): 11.0, ("re-", 1.0): 18.0}
for (mode, dx), tolerance in TOLERANCE.items():
    error = runs[mode, dx]["at_points"] - reference_at
    assert np.all(np.abs(error) < tolerance), (mode, dx, error)

# Coarse monodomain is later at P8 than R-E- is
error_P8 = {mode: runs[mode, 0.5]["at_points"][P8] - reference_at[P8] for mode in MODES}
assert abs(error_P8["monodomain"]) > abs(error_P8["re-"]), error_P8

# At 1.0 mm the monodomain wave does not enter the slab: most nodes never reach 0 mV, and
# those that do are on the edge y = z = 0
activation_mono_1 = runs["monodomain", 1.0]["activation"].x.array
assert np.count_nonzero(np.isnan(activation_mono_1)) > activation_mono_1.size / 2
assert np.all(on_edge[activated]), np.count_nonzero(activated & ~on_edge)

# %% [markdown]
# R-E+ and R-E− differ in how a node's activation relates to its $t_a$. Under R-E− every
# node is fired by its foot alone, so it reaches 0 mV one foot latency after $t_a$. Under
# R-E+ diffusion mostly pulls nodes up, and some activate milliseconds before their
# $t_a$. A node beside tissue still at rest can instead lose current to it, and activate
# slightly later than an isolated cell would. Under R-E+, $t_a$ is not the activation
# time.

# %%
for mode in ("re+", "re-"):
    run = runs[mode, 0.5]
    lag = run["activation"].x.array - run["eikonal"].x.array
    print(
        f"{MODES[mode]}, 0.5 mm: activation - t_a from {lag.min():.2f} to {lag.max():.2f} ms, "
        f"median {np.median(lag):.2f} ms",
    )

# %% [markdown]
# ### Against the published numbers
#
# Niederer et al. report the last activation, at P8, converging to about 42.5–43 ms
# for three codes at 0.05 mm and 0.001 ms. beat's own Niederer demo
# (`demos/niederer_benchmark.py` in fenicsx-beat) tabulates its monodomain results at
# P1–P9, at a time step of 0.05 ms. Its convention differs slightly from the one here,
# in time and in space. In time, it takes the start of the first time step after which
# $v$ is above 0 mV, where this tutorial interpolates linearly in time, a difference of
# less than a time step. In space, it evaluates $v$ at the point, where this tutorial
# finds the activation time of every node and interpolates those to the point. At a node
# the two are the same, but at a point that is not a node they can differ by more.

# %%
NIEDERER_P8 = (42.5, 43.0)  # ms, Niederer et al. 2011
BEAT_TABLE = {  # ms, fenicsx-beat's demos/niederer_benchmark.py, dt 0.05 ms
    0.5: [1.25, 51.1, 34.9, 58.9, 14.1, 49.5, 34.0, 56.65, 26.05],
    0.2: [1.25, 29.7, 32.9, 40.2, 9.55, 30.0, 32.95, 39.9, 18.9],
    0.1: [1.25, 26.85, 33.3, 40.35, 8.4, 27.5, 33.85, 40.55, 18.95],
}
print(f"{'activation (ms)':<24}" + "".join(f"{f'P{i}':>8}" for i in range(1, 10)))
for dx, values in BEAT_TABLE.items():
    row(f"beat's table, {dx} mm", values)
row(f"monodomain here, {DX_REF} mm", reference_at)
row("monodomain here, 0.5 mm", runs["monodomain", 0.5]["at_points"])
for dx, ours in ((0.2, reference_at), (0.5, runs["monodomain", 0.5]["at_points"])):
    difference = np.abs(ours - np.array(BEAT_TABLE[dx]))
    print(
        f"largest difference from beat's table at {dx} mm: {difference.max():.2f} ms, "
        f"at P{difference.argmax() + 1}",
    )
print(f"Niederer et al., converged, at P8: {NIEDERER_P8[0]}–{NIEDERER_P8[1]} ms")

# %% [markdown]
# beat's table agrees with the monodomain runs here to within the difference of the two
# conventions. The largest difference at 0.2 mm is at P9, the centre, which is not a node
# of that mesh, so it is the convention in space. beat's 0.5 mm row also predates beat
# 0.13.0, which activates six of the nine points one time step later (measured when this
# tutorial was written). Its 0.1 mm row shows that the 0.2 mm reference is not
# converged: across the fibres (P2, P5, P6) the wave arrives up to about 3 ms earlier at
# 0.1 mm, since $CV_T$ is still too low at 0.2 mm. A 0.1 mm reference would take too long
# for this tutorial.
#
# R-E−'s errors, and R-E+'s at 1.0 mm, are largest at the far corners, P4 and P8. Two
# parts of that error can be seen at P8. A continuous eikonal front, an ellipsoid growing from the
# stimulated cube with the two velocities, can be computed by hand. The discrete eikonal
# is later than it, more so on the coarser mesh (which also has a smaller stimulated
# cube). And the reference reaches P8 earlier than that front.

# %%
print(f"{'at P8 (ms)':<36}{'0.5 mm':>8}{'1.0 mm':>8}")
continuum, discrete = [], []
for dx in (0.5, 1.0):
    run = runs["re-", dx]
    cells = run["tags"].find(1)
    vertices = dolfinx.mesh.entities_to_geometry(run["geo"].mesh, 3, cells).ravel()
    box = run["geo"].mesh.geometry.x[vertices].max(axis=0)
    gap = np.maximum(POINTS[P8] - box, 0.0)
    distance = np.sqrt((gap[0] / CV_L) ** 2 + (gap[1] / CV_T) ** 2 + (gap[2] / CV_T) ** 2)
    continuum.append(STIM_DELAY + distance)
    t_a_P8 = beat.utils.PointEvaluator(run["geo"].mesh, POINTS[[P8]])(run["eikonal"])[0]
    discrete.append(t_a_P8 + FOOT_LATENCY)
rows = {
    "continuous front, at 0 mV": continuum,
    "discrete eikonal t_a + FOOT_LATENCY": discrete,
    "R-E−": [runs["re-", dx]["at_points"][P8] for dx in (0.5, 1.0)],
}
for label, values in rows.items():
    print(f"{label:<36}" + "".join(f"{v:8.2f}" for v in values))
print(f"{'reference, 0.2 mm':<36}{reference_at[P8]:8.2f}")

# %% [markdown]
# So R-E−'s error at P8 is the eikonal solver's own error, which grows with the mesh
# size, plus the difference between the reference and an ellipsoidal front with the
# velocities measured along the axes: on these tetrahedra the 0.2 mm reference travels
# faster along the diagonal than that front. (The coarse monodomain runs are the other
# way: at 0.5 mm they reach P2, P4, P6 and P8 17 to 21 ms late, in the table above.)
# This is an observation on this set-up, not an established mechanism. $CV_T$ is not
# converged at 0.2 mm, the reference's stimulated cube is 1.4 mm rather than 1.5, and
# `STIM_DELAY` is one mean over its nodes, and each of these moves the comparison. The
# published 42.5–43 ms is later than both fine monodomain runs (here, and beat's 0.1 mm
# row), and close to the continuous front from the full 1.5 mm cube. R-E+ activates
# earlier after its $t_a$ than R-E− does (the median lags above), so it runs earlier than
# R-E−. That offsets the far-corner error, and adds an error along the axes (P2, P3, P5),
# where R-E+ is further from the reference than R-E−.
#
# The activation maps on the face $z = 0$. Grey marks nodes that never reach 0 mV.

# %%
levels = np.linspace(0.0, 60.0, 13)
cmap = plt.get_cmap("viridis").with_extremes(over="0.85")
maps = [("reference, 0.2 mm", reference["activation"])]
maps += [(f"eikonal $t_a$, {dx} mm", runs["re-", dx]["eikonal"]) for dx in (0.5, 1.0)]
maps += [(f"{MODES[mode]}, {dx} mm", runs[mode, dx]["activation"]) for mode, dx in runs]

fig, axes = plt.subplots(3, 3, figsize=(13, 5.6), sharex=True, sharey=True)
for ax, (title, function) in zip(axes.flat, maps):
    xyz = function.function_space.tabulate_dof_coordinates()
    bottom = np.isclose(xyz[:, 2], 0.0)
    values = np.nan_to_num(function.x.array[bottom], nan=1e3)
    contours = ax.tricontourf(
        xyz[bottom, 0],
        xyz[bottom, 1],
        values,
        levels=levels,
        cmap=cmap,
        extend="max",
    )
    ax.set(aspect="equal", title=title)
for ax in axes[-1]:
    ax.set_xlabel("x [mm]")
for ax in axes[:, 0]:
    ax.set_ylabel("y [mm]")
fig.colorbar(contours, ax=axes, label="activation time [ms]", shrink=0.8)
plt.show()

# %% [markdown]
# ## Coupled runs
#
# Now each of the three drives mechanics, through `SimulationController` as in tutorial 4.
# EP runs at 0.5 mm. The mechanics mesh is coarser, about 2 mm (10 × 4 × 2 boxes of
# tetrahedra), with P2 displacement and the contraction states at the points of a degree 2
# quadrature rule. A 1 mm mechanics mesh changes $T_a$ at P8 by about 5 %, and would take
# about 6 minutes more (both measured when this tutorial was written). The two meshes need
# not match: the controller interpolates between them.
#
# The material is that of tutorial 4, an incompressible Holzapfel-Ogden material with
# fibres along $x$, held by rollers on the faces $x = 0$, $y = 0$ and $z = 0$. These hold
# the face $x = 0$ in $x$ and let the slab shorten freely along its fibres. Rollers on
# $x = 0$ alone would leave the slab free to slide and rotate, a singular problem, and a
# fixed face would clamp the corner P1.
#
# Land is all on the mechanics side, in `GeneratedActivation` under the `monolithic`
# scheme, so the whole contraction model is stepped inside Newton. Two things are changed
# from crossbridge's defaults:
#
# - **The calibration.** crossbridge's Land has values from skinned fibres, and with TT06's
#   calcium transient it produces little tension. The calibration below is the intact-cell
#   values of Margara et al. (*Prog Biophys Mol Biol* 2021), which simcardemsx's own
#   ToR-ORd + Land files use, in Land's units (µM, Pa and 1/s).
# - **`B` = 1.** crossbridge's Land starts with no tropomyosin blocking (`B` = 0), so its
#   unblocked fraction $U = 1 - B - S - W$ is 1, and crossbridges form before any calcium
#   arrives. Starting fully blocked, as simcardemsx's own files do, removes that spurious
#   tension at the start. `B` is a state of the backend, so it is set in its `states_prev`.

# %%
DX_EP, DX_MECH, QUADRATURE_DEGREE = 0.5, 2.0, 2  # mm, mm, -
DT_MECH, T_END = 1.0, 150.0  # ms
LAND_CALIBRATION = {
    "ca50_ref": 0.805,
    "Tref": 120_000.0,
    "ku": 40.0,
    "kuw": 182.0,
    "kws": 12.0,
    "nTm": 2.4,
}


def rollers(V):
    """u_i = 0 on the face x_i = 0, for i = 0, 1, 2."""
    V0, _ = V.sub(0).collapse()
    zero = dolfinx.fem.Function(V0)
    bcs = []
    for i in range(3):
        facets = dolfinx.mesh.locate_entities_boundary(
            V.mesh,
            2,
            lambda x, i=i: np.isclose(x[i], 0.0),
        )
        dofs = dolfinx.fem.locate_dofs_topological((V.sub(i), V0), 2, facets)
        bcs.append(dolfinx.fem.dirichletbc(zero, dofs, V.sub(i)))
    return bcs


def mechanics():
    geo = beat.geometry.get_3D_slab_geometry(MPI.COMM_WORLD, Lx=LX, Ly=LY, Lz=LZ, dx=DX_MECH)
    backend = GeneratedActivation(
        mech_module,
        geo.mesh,
        geo.f0,
        quadrature_degree=QUADRATURE_DEGREE,
        scheme="monolithic",
        parameters=LAND_CALIBRATION,
        tension="Ta",
        tension_unit="kPa",
    )
    num_states = len(mech_module.state)
    backend.states_prev.x.array.reshape(-1, num_states)[:, mech_module.state["B"]] = 1.0

    model = pulse.CardiacModel(
        material=pulse.HolzapfelOgden(
            f0=geo.f0,
            s0=geo.s0,
            **pulse.HolzapfelOgden.transversely_isotropic_parameters(),
        ),
        active=backend,
        compressibility=pulse.Incompressible(),
    )
    petsc_options = pulse.StaticProblem.default_parameters()["petsc_options"]
    petsc_options["snes_atol"] = 1e-9
    problem = pulse.StaticProblem(
        model=model,
        geometry=pulse.Geometry(mesh=geo.mesh, metadata={"quadrature_degree": QUADRATURE_DEGREE}),
        bcs=pulse.BoundaryConditions(dirichlet=[rollers]),
        parameters={
            "base_bc": pulse.problem.BaseBC.free,
            "petsc_options": petsc_options,
            "u_space": "P_2",
        },
    )
    return geo, backend, problem


# %% [markdown]
# The controller calls a *mechanics driver* once per step. A driver is anything with a
# `problem` and an `advance(t_n, dt)` that returns whether the step converged. Passing a
# problem, as tutorial 4 did, wraps it in one that just solves; this one also times the
# solves. EP's time is taken between the EP callbacks.
#
# For R-E, `ep_solver.ode` is beat's `FootCurrentODESolver`, and the cell states live in
# the `DolfinODESolver` inside it. The controller reaches that with
# `simcardemsx.controller.dolfin_ode_solver`, to read `Ca_i` from it. Since nothing comes
# back from mechanics, the EP of each coupled run is that of the EP-only run at 0.5 mm.


# %%
class TimedSolve:
    """A mechanics driver: solve, and add up the time spent solving."""

    def __init__(self, problem):
        self.problem = problem
        self.seconds = 0.0

    def advance(self, t_n, dt):
        start = time.perf_counter()
        converged = self.problem.solve()
        self.seconds += time.perf_counter() - start
        return converged


def run_coupled(mode):
    geo_ep, tags = slab(DX_EP)
    t_a = None if mode == "monodomain" else eikonal(geo_ep, tags)
    ep = ep_solver(mode, geo_ep, tags, t_a=t_a)
    geo, backend, problem = mechanics()
    driver = TimedSolve(problem)
    controller = SimulationController(
        mechanics=driver,
        ep_solver=ep,
        backend=backend,
        ode_modules=modules,
        dt_mech=DT_MECH,
        dt_ep=DT_EP,
    )
    print(f"{MODES[mode]}: {type(ep.ode).__name__} → {type(dolfin_ode_solver(ep)).__name__}")

    # Ta at P1 and P8, λ there (averaged onto P1 from the quadrature points), and the
    # shortening of the slab: -u_x at P3, on the far face
    lmbda_p1 = dolfinx.fem.Function(dolfinx.fem.functionspace(geo.mesh, ("P", 1)))
    average_lmbda = make_averager(backend.lmbda_prev, lmbda_p1)
    probe = beat.utils.PointEvaluator(geo.mesh, POINTS[[P1, P8, P3]])
    trace = {"t": [0.0], "Ta": [[0.0, 0.0]], "lmbda": [[1.0, 1.0]], "shortening": [0.0]}
    clock = {"ep": 0.0, "mark": 0.0, "newton": 0}

    def on_ep_step(t, ep_step):
        now = time.perf_counter()
        clock["ep"] += now - clock["mark"]
        clock["mark"] = now

    def on_mech_step(t, mech_step, newton_iterations):
        average_lmbda()
        trace["t"].append(t)
        trace["Ta"].append(probe(backend.active_tension)[:2])
        trace["lmbda"].append(probe(lmbda_p1)[:2])
        trace["shortening"].append(-probe(problem.u.sub(0).collapse())[2])
        clock["newton"] += newton_iterations

    while controller.t < T_END - 1e-9:
        clock["mark"] = time.perf_counter()
        controller.step(ep_callback=on_ep_step, mech_callback=on_mech_step)
    result = {key: np.array(value) for key, value in trace.items()}
    result.update(ep_seconds=clock["ep"], mech_seconds=driver.seconds, newton=clock["newton"])
    return result


coupled = {mode: run_coupled(mode) for mode in MODES}

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
for i, (mode, result) in enumerate(coupled.items()):
    color = f"C{i}"
    for j, (point, style) in enumerate((("P1", "-"), ("P8", "--"))):
        label = f"{MODES[mode]}, {point}"
        axes[0].plot(result["t"], result["Ta"][:, j], style, color=color, label=label)
        axes[1].plot(result["t"], result["lmbda"][:, j], style, color=color, label=label)
    axes[2].plot(result["t"], result["shortening"], color=color, label=MODES[mode])
axes[0].set(ylabel="$T_a$ [kPa]", title="Active tension")
axes[1].set(ylabel="$\\lambda$ [-]", title="Fibre stretch")
axes[2].set(ylabel="shortening [mm]", title=f"Shortening of the {LX:g} mm slab")
axes[0].legend(frameon=False, fontsize="small", ncol=2)
axes[2].legend(frameon=False)
for ax in axes:
    ax.set_xlabel("time [ms]")
fig.tight_layout()
plt.show()

# %%
print(f"{'':<12}{'t (ms)':>7}{'Ta P1':>8}{'Ta P8':>8}{'λ P1':>8}{'λ P8':>8}{'short.':>8}")
for mode, result in coupled.items():
    for t in (50.0, 100.0, 150.0):
        i = int(np.argmin(np.abs(result["t"] - t)))
        values = [*result["Ta"][i], *result["lmbda"][i], result["shortening"][i]]
        print(f"{MODES[mode]:<12}{t:7.0f}" + "".join(f"{v:8.3f}" for v in values))
print()
print(f"{'wall time':<12}{'EP (s)':>8}{'mech (s)':>10}{'Newton its':>12}")
for mode, result in coupled.items():
    print(
        f"{MODES[mode]:<12}{result['ep_seconds']:8.1f}{result['mech_seconds']:10.1f}"
        f"{result['newton']:12d}",
    )

# Every run contracts: Ta at the far corner P8 exceeds 1 kPa at 150 ms
for mode, result in coupled.items():
    i = int(np.argmin(np.abs(result["t"] - 150.0)))
    assert result["Ta"][i, 1] > 1.0, (mode, result["Ta"][i])

# %% [markdown]
# P1, in the stimulated corner, follows a similar course in all three runs. At P8 the
# monodomain run's tension lags, because at 0.5 mm its wave reaches P8 well over 10 ms
# after the reference's (the table above); R-E+ and R-E− are close to each other. The
# tension is still rising when the runs stop at 150 ms. It stays far below what an
# isometric cell would develop, about 62 kPa at its peak (measured when this tutorial was
# written), because the soft slab shortens by about a fifth along its fibres (on this
# ~2 mm mechanics mesh), and Land's length and velocity dependence lower the tension as it
# does.
#
# The three runs cost about the same here. EP at 0.5 mm is cheap whichever model runs it,
# since this slab has a few thousand nodes. R-E pays off on a whole heart, where it allows
# a mesh that monodomain cannot use: Neic et al. report three orders of magnitude.
#
# ## Limits
#
# - **The activation is fixed.** The eikonal is solved once, on the undeformed slab, and
#   does not respond to the deformation, as in Strocchi et al. Here the coupling is also
#   one-way, so nothing mechanics does reaches EP.
# - **No re-entry.** The eikonal gives every point one activation time, so a wave that
#   breaks up or comes back cannot be represented. beat's `FootCurrentODESolver` can fire
#   the foot again every beat (its `period`), which is how R-E studies run several beats.
# - **R-E−'s repolarisation.** Without diffusion, neighbouring nodes do not even out
#   their action potentials, and Neic et al. found that R-E− overestimates repolarisation
#   gradients. R-E+ largely avoids it.
# - **The foot was fitted to one cell model.** Its $A$ and $\tau_F$ were fitted by Neic et
#   al. to the ten Tusscher-Noble-Noble-Panfilov model, an ancestor of TT06, so they
#   suit TT06 here. Another cell model has another latency, which is why it is measured
#   above rather than assumed.
#
# ## Where to go next
#
# - The [slab example](../applications/slab.md) takes `--ep re+` or `--ep re-` to run
#   this kind of EP in a full program, and so do the
#   [BiV](../applications/circulation-biv.md) and [Rodero](../applications/rodero.md)
#   examples.
# - The [next tutorial](08_composed_two_way_coupling.ipynb) composes a model whose
#   contraction feeds back into the calcium, ToR-ORd with Land, and splits it between EP
#   and mechanics in two ways.
