# %% [markdown]
# # One two-way coupled cell model, split two ways
#
# [The reaction-eikonal tutorial](07_reaction_eikonal_slab.ipynb) composed a cell model
# with Land's contraction model, and nothing went back from the contraction to the cell.
# In ToR-ORd and Land as Margara et al. (*Prog Biophys Mol Biol* 2021) couple them,
# calcium bound to troponin is Land's state `CaTRPN`, and the flux into it, `J_TRPN`, is
# taken out of the cytosolic calcium of ToR-ORd. The coupling goes both ways: calcium
# drives the contraction, and the contraction's troponin, whose calcium affinity depends
# on the fibre stretch, shapes the calcium transient.
#
# simcardemsx ships this model in hand-made `.ode` files, one per split
# ([the cell-model tutorial](01_cell_model.ipynb)). This tutorial composes it from the
# published models instead, once, and splits the one composed model in two places with
# gotranx's `ODE.split`. In simcardemsx a split decides two things:
#
# - **What is inside Newton.** The mechanics side is stepped at the quadrature points of
#   the mechanics mesh, inside the mechanics solve under the monolithic scheme
#   ([the single-element tutorial](03_single_element.ipynb)), so its states see the
#   stretch that Newton is converging to.
# - **What is lagged a step.** Each step, EP runs first, with what came back from the
#   previous mechanics step, the stretch $\lambda$ included. Everything on the EP side
#   that depends on what mechanics returns sees it one mechanics step late.
#
# The tutorial
#
# 1. composes ToR-ORd and Land, both fetched from a model registry, with a two-way glue;
# 2. splits the composed model two ways and prints what crosses in each;
# 3. checks one cell of the composed model against simcardemsx's hand-made files;
# 4. runs each split coupled on the slab of
#    [the reaction-eikonal tutorial](07_reaction_eikonal_slab.ipynb), beside its hand-made
#    counterpart.

# %%
import html
import io
import json
import tempfile
import time
import zipfile
from pathlib import Path

from mpi4py import MPI

import beat
import dolfinx
import gotranx
import matplotlib.pyplot as plt
import numpy as np
import odezoo
import pulse
import pyvista
from beat.eikonal import Source, solve_eikonal
from beat.reaction_eikonal import ExponentialFoot, FootCurrentODESolver
from IPython.display import HTML, display
from trame_vtk.modules.vtk.serializers import (
    SynchronizationContext,
    extract_array_hash,
    initialize_serializers,
    reference_id,
    serialize,
)
from trame_vtk.tools.vtksz2html import write_html

from simcardemsx.backends import GeneratedActivation
from simcardemsx.controller import SimulationController, dolfin_ode_solver
from simcardemsx.ode_model import EP_SCHEME, load_ode_modules
from simcardemsx.transfer_plan import resolve

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
ODEFILES = HERE.parent / "odefiles"

torord_file = odezoo.get("torord_dyn_cl", version="github-11e463d")
land_file = odezoo.get("land_2017", version="github-6e5a5aa")
print("ToR-ORd:", torord_file.name)
print("Land:   ", land_file.name)

# %% [markdown]
# ## Composing the model
#
# ToR-ORd is the model of Tomek et al. (*eLife* 2019), in its dynamic-chloride variant
# (Tomek, Bueno-Orovio & Rodriguez, *bioRxiv* 2020, doi 10.1101/2020.06.01.127043), as the
# CellML file its authors publish, and Land is crossbridge's `.ode` file, as in
# [the reaction-eikonal tutorial](07_reaction_eikonal_slab.ipynb). ToR-ORd's CellML is
# licensed GPL-3.0-only, so it is not stored in this repository: odezoo fetches it at run
# time. Its default cell type, 0, is the endocardial cell of the hand-made files.
#
# The recipe has two parts, `ep` and `land`, and nothing is split yet (gotranx's
# [guide to composing models](https://finsberg.github.io/gotranx/docs/composition.html)
# describes recipes and glue files). ToR-ORd runs in ms and Land in s. Three of Land's
# names are renamed: ToR-ORd has gates named `a` (of `Ito`) and `h` (of `INa`), and
# simcardemsx sets a `dLambda` of its own.

# %%
glue_file = ODEFILES / "torord_land.glue.ode"
recipe = {
    "time_unit": "ms",
    "glue": glue_file.name,
    "parts": [
        {"name": "ep", "file": str(torord_file), "time_unit": "ms"},
        {
            "name": "land",
            "file": str(land_file),
            "time_unit": "s",
            "rename": {"a": "a_land", "h": "h_land", "dLambda": "dLambda_land"},
        },
    ],
}
print(glue_file.read_text())

# %% [markdown]
# The glue gives Land the stretch and its rate, as in
# [the reaction-eikonal tutorial](07_reaction_eikonal_slab.ipynb), and feeds it ToR-ORd's
# calcium. Its last two lines make the coupling two-way. ToR-ORd buffers its cytosolic
# calcium on calmodulin and troponin with a rapid-equilibrium factor `Bcai`, and
# `replace Bcai` drops troponin from that factor, since troponin is now Land's state. The
# flux into Land's troponin, `J_TRPN`, is then taken out of `dcai_dt`. References to
# another part's names, such as `ep.cai`, `replace` and `+=` are what a glue file adds to
# the `.ode` grammar
# ([the glue file](https://finsberg.github.io/gotranx/docs/composition.html#the-glue-file)
# in gotranx's guide).
#
# Composing converts the CellML file, which takes a few seconds, and gotranx may warn
# that it drops a guard on a concentration, which is harmless here.

# %%
ode = gotranx.compose(recipe, base_dir=ODEFILES)
odefile = HERE / "generated_odes" / "torord_land" / "torord_land.ode"
odefile.parent.mkdir(parents=True, exist_ok=True)
ode.save(odefile)

print(f"{len(ode.states)} states, {len(ode.parameters)} parameters")
names = sorted(c.name for c in ode.components)
for part in ("ep", "land"):
    inside = [n.removeprefix(f"{part}.") for n in names if n.startswith(f"{part}.")]
    print(f"{part}: {', '.join(inside)}")

# %% [markdown]
# The saved file starts with a header holding the recipe and the sha256 of every input.
# Composing again saves the same text, byte for byte, so simcardemsx's cache of generated
# code, which is keyed on the `.ode` file's sha256, is reused on the next run. The header
# names the cached files by their absolute paths, so the text is the same on one machine,
# not across machines.

# %%
with tempfile.TemporaryDirectory() as folder:
    again = Path(folder) / odefile.name
    gotranx.compose(recipe, base_dir=ODEFILES).save(again)
    assert again.read_text() == odefile.read_text()
print("composing twice saves byte-identical text")

# %% [markdown]
# ## Two splits
#
# `ODE.split(name, inputs, outputs)` carves a new component `name` out of the model, like
# a function signature: it holds the outputs and everything they depend on, up to the
# inputs and the parameters. Naming it `mechanics` makes it the component simcardemsx puts
# on the mechanics side. gotranx's
# [worked example](https://finsberg.github.io/gotranx/examples/split-ode/main.html) splits
# O'Hara-Rudy's model with Land's the way the Ca$_i$ split below does. Two splits of the
# one composed model:
#
# - **zeta**: `S` and `W` (Land's strongly and weakly bound crossbridges) in; the two
#   distortion states `Zs`, `Zw` and the tension out. Only these are inside Newton.
# - **Ca$_i$**: the calcium in; the tension and `J_TRPN` out. All of Land's active states
#   are inside Newton.
#
# Each is saved and generated into its own folder, and its hand-made counterpart in
# `examples/odefiles` beside it, in the folder
# [the cell-model tutorial](01_cell_model.ipynb) generated it into.

# %%
SPLITS = {
    "zeta": {"inputs": ["S", "W"], "outputs": ["Zs", "Zw", "Ta"]},
    "cai": {"inputs": ["cai"], "outputs": ["Ta", "J_TRPN"]},
}
composed, hand_made = {}, {}
for split, signature in SPLITS.items():
    path = HERE / "generated_odes" / f"torord_land_{split}" / f"torord_land_{split}.ode"
    path.parent.mkdir(parents=True, exist_ok=True)
    ode.split("mechanics", **signature).save(path)
    composed[split] = load_ode_modules(path, path.parent)
    hand_file = ODEFILES / f"ToRORd_dynCl_endo_{split}split.ode"
    hand_made[split] = load_ode_modules(hand_file, HERE / "generated_odes" / hand_file.stem)

print(f"{'':16}{'mechanics states':<44}{'EP → mechanics':<16}{'mechanics → EP':<16}λ to EP")
stretch_to_ep = {}
for split in SPLITS:
    for label, modules in (("composed", composed[split]), ("hand-made", hand_made[split])):
        crossings = resolve(modules.ep, modules.mechanics)
        stretch_to_ep[split, label] = crossings.stretch_to_ep
        print(
            f"{split:<5}{label:<11}{', '.join(sorted(modules.mechanics.state)):<44}"
            f"{', '.join(crossings.forward):<16}{', '.join(crossings.backward):<16}"
            f"{crossings.stretch_to_ep}",
        )

# λ crosses to EP in every split but the hand-made Ca_i file's
assert stretch_to_ep == {
    ("zeta", "composed"): True,
    ("zeta", "hand-made"): True,
    ("cai", "composed"): True,
    ("cai", "hand-made"): False,
}, stretch_to_ep

# %% [markdown]
# The crossings are the declared inputs and outputs, under the composed model's names
# (Land's own) or the hand-made files' (`XS`, `XW`, `Zetas`, …). The tension is an output
# too, but it does not cross: the mechanics solver reads it.
#
# In the **zeta** split, λ crosses to EP. Land's troponin and crossbridge populations stay
# on the EP side, and troponin's calcium sensitivity depends on the stretch (Land's `Ca50`
# falls as λ grows), so EP needs λ, and gets it a step late. The distortion states, which
# carry the stretch rate, are inside Newton with the tension: lagging the stretch there is
# what makes the naive scheme of [the 0D coupling tutorial](02_coupling_schemes_0d.ipynb)
# unstable. Land's passive state `Cd` also stays on the EP side, where the hand-made zeta
# file keeps it in `mechanics`.
#
# In the **Ca$_i$** split all of Land's active states are inside Newton, and EP gets only
# `J_TRPN`, a step late. Yet λ still crosses to EP, where the hand-made Ca$_i$ file sends
# none. That is `Cd` again, Land's passive state: neither output needs it, so the split
# leaves it on the EP side, beside ToR-ORd, and its rate depends on λ. A split's outputs
# decide what moves. Listing `Cd` among them gives the hand-made file's layout, with all
# seven of Land's states in `mechanics` and no λ to EP:

# %%
with_cd = ode.split("mechanics", inputs=["cai"], outputs=["Ta", "J_TRPN", "Cd"])
moved = with_cd.get_component("mechanics")
moved_states = {s.name for s in moved.states}
lmbda_to_ep = "lmbda" in {p.name for p in (with_cd - moved).parameters}
print("mechanics states:", ", ".join(sorted(moved_states)))
print("λ to EP:", lmbda_to_ep)

land_states = {s.name for c in ode.components if c.name.startswith("land.") for s in c.states}
assert moved_states == land_states and len(land_states) == 7, (moved_states, land_states)
assert not lmbda_to_ep

# %% [markdown]
# `Cd` feeds only Land's passive tension, which the mechanics below do not use (their
# passive stress is pulse's Holzapfel-Ogden material), so where it sits does not change
# the active tension. The tutorial keeps the split without it.
#
# ## One cell against the hand-made files
#
# The hand-made files carry the Land values of Margara et al., in ms and kPa.
# crossbridge's Land has values from skinned fibres, so the composed model is given the
# same calibration as in [the reaction-eikonal tutorial](07_reaction_eikonal_slab.ipynb),
# in Land's own units (µM, Pa and 1/s), and starts with tropomyosin fully blocking,
# `B` = 1, as the hand-made files do.
#
# The whole model, unsplit, runs one beat as one cell at λ = 1, with ToR-ORd's own
# stimulus at $t = 0$, as in [the cell-model tutorial](01_cell_model.ipynb). The zeta and
# Ca$_i$ hand-made files differ only in which component, `mechanics` or another, their
# states, parameters and expressions belong to. Loaded whole, their dynamics are
# identical, so one of them is run.

# %%
LAND_CALIBRATION = {
    "ca50_ref": 0.805,
    "Tref": 120_000.0,
    "ku": 40.0,
    "kuw": 182.0,
    "kws": 12.0,
    "nTm": 2.4,
}
DT_EP = 0.05  # ms


def numpy_model(model):
    """The whole model as NumPy code, stepped by second-order Rush-Larsen."""
    code = gotranx.cli.gotran2py.get_code(
        model,
        scheme=[gotranx.schemes.Scheme.rush_larsen_2],
    )
    namespace: dict = {}
    exec(code, namespace)
    return namespace


def one_beat(cell, states, parameters, t_end=1000.0):
    """v (mV), cai (mM) and Ta (kPa) at every time step of one beat."""
    i_v, i_cai = cell["state_index"]("v"), cell["state_index"]("cai")
    i_Ta = cell["monitor_index"]("Ta")
    trace = np.empty((round(t_end / DT_EP) + 1, 3))
    trace[0] = states[i_v], states[i_cai], cell["monitor_values"](0.0, states, parameters)[i_Ta]
    for k in range(trace.shape[0] - 1):
        states = cell["rush_larsen_2"](states, k * DT_EP, DT_EP, parameters)
        Ta = cell["monitor_values"]((k + 1) * DT_EP, states, parameters)[i_Ta]
        trace[k + 1] = states[i_v], states[i_cai], Ta
    return trace


cell = numpy_model(ode)
composed_cell = one_beat(
    cell,
    cell["init_state_values"](B=1.0),
    cell["init_parameter_values"](**LAND_CALIBRATION),
)
cell = numpy_model(gotranx.load_ode(ODEFILES / "ToRORd_dynCl_endo_zetasplit.ode"))
hand_cell = one_beat(cell, cell["init_state_values"](), cell["init_parameter_values"]())

# %%
times = DT_EP * np.arange(composed_cell.shape[0])
fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharex=True)
for trace, label, style in ((composed_cell, "composed", "-"), (hand_cell, "hand-made", "--")):
    axes[0].plot(times, trace[:, 0], style, label=label)
    axes[1].plot(times, trace[:, 1] * 1e3, style, label=label)
    axes[2].plot(times, trace[:, 2], style, label=label)
axes[0].set(ylabel="v [mV]", title="Action potential")
axes[1].set(ylabel="[Ca$^{2+}$]$_i$ [µM]", title="Calcium transient")
axes[2].set(ylabel="$T_a$ [kPa]", title="Active tension (isometric)")
axes[2].legend(frameon=False)
for ax in axes:
    ax.set_xlabel("time [ms]")
fig.tight_layout()
plt.show()

# %% [markdown]
# The two agree within the measured model difference below, not bit for bit. The
# tolerances are twice the largest differences measured when the tutorial was written.

# %%
CELL_TOLERANCE = {"v": 3.3, "cai": 5.1e-5, "Ta": 4.5}  # mV, mM, kPa
difference = np.abs(composed_cell - hand_cell).max(axis=0)
for (name, tolerance), largest in zip(CELL_TOLERANCE.items(), difference):
    print(f"max |Δ{name}| = {largest:.3g} (tolerance {tolerance:g})")
    assert largest < tolerance, (name, largest)
print(f"largest |ΔTa| at {times[np.abs(composed_cell - hand_cell)[:, 2].argmax()]:.0f} ms")
for trace, label in ((composed_cell, "composed"), (hand_cell, "hand-made")):
    print(f"peak Ta, {label}: {trace[:, 2].max():.2f} kPa")

# %% [markdown]
# The difference comes from the models, not from the composition: the hand-made files'
# calcium balance differs from the ToR-ORd CellML's in one equation. The CellML's
# `dcai_dt` has `-(ICaL_i + IpCa + ICab - 2*INaCa_i)`, the myoplasmic parts of the L-type
# and exchanger currents with the pump and background currents. The hand-made files drop
# `ICaL_i` from it and divide `INaCa_i` by 3. They also start from another `cai` (1e-4 mM,
# against the CellML's 7.45e-5) and another `CaTRPN` (1e-8, against crossbridge's
# 0.0016). And in Land they cap `CaTRPN**(-nTm/2)` at 100, which moves the tension by
# about 5e-4 kPa. Patching these four things into the composed model brings the
# single-cell difference down to round-off, so they are the whole difference. That was
# measured once, offline, when this tutorial was written, and is not repeated here. Which
# `dcai_dt` is right is a question for the model, and this tutorial does not answer it.
#
# Land's other differences do not show in one cell at λ = 1. crossbridge writes its length
# dependence with `Min` and `Max` where the hand-made files use `Conditional`, which is
# the same function. The guards differ: crossbridge has `Max(0, Ca)`, `Max(1e-6, Ca50)`
# and `Max(1e-12, CaTRPN)`, which the hand-made files lack, but they never trigger in
# these runs. Its passive part is formulated differently, but that does not enter the
# active tension.
#
# ## Coupled, on the slab
#
# Each split now runs coupled, beside its hand-made counterpart, for 100 ms on the slab of
# [the reaction-eikonal tutorial](07_reaction_eikonal_slab.ipynb): EP at 0.5 mm under
# R-E−, and mechanics on the ~2 mm mesh with P2 displacement, the contraction states at
# the points of a degree-2 quadrature rule, the monolithic scheme, and rollers on $x = 0$,
# $y = 0$ and $z = 0$. Each Land parameter of the calibration goes to whichever side of
# the split declares it, and so does `B` = 1: the zeta split keeps `B` on the EP side, the
# Ca$_i$ split in mechanics.
#
# The eikonal is that of [the reaction-eikonal tutorial](07_reaction_eikonal_slab.ipynb),
# with its two conduction velocities and its sources in the stimulated corner, starting at
# 0.324 ms, the values measured there. Any one $t_a$ shared by the four runs will do, so it
# is computed once. ToR-ORd's own stimulus is off: the foot current fires each node.

# %%
DX_EP, DX_MECH, QUADRATURE_DEGREE = 0.5, 2.0, 2  # mm, mm, -
DT_MECH, T_END = 1.0, 100.0  # ms
LX, LY, LZ = 20.0, 7.0, 3.0  # mm
STIM_SIZE = 1.5  # mm
CV_L, CV_T = 0.6275, 0.2071  # mm/ms, measured in the reaction-eikonal tutorial
SOURCE_TIME = 0.324  # ms, the reaction-eikonal tutorial's


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


geo_ep, tags = slab(DX_EP)
t_a = solve_eikonal(
    geo_ep.mesh,
    (CV_L, CV_T, CV_T),
    (geo_ep.f0, geo_ep.s0, geo_ep.n0),
    [Source((tags, 1), SOURCE_TIME)],
)
print(f"t_a from {t_a.x.array.min():.2f} to {t_a.x.array.max():.2f} ms")


def on_side(values, module):
    """The entries of values that module declares as parameters."""
    return {name: value for name, value in values.items() if name in module.parameter}


# %% [markdown]
# `ep_solver` is the R-E− of
# [the reaction-eikonal tutorial](07_reaction_eikonal_slab.ipynb), without the other two
# kinds of EP. The EP half now has missing variables, what comes back from mechanics, so
# the ODE solver is given room for them. `mechanics` is that tutorial's, for any split.


# %%
def ep_solver(ep_module, calibration):
    """R-E− EP on the slab: every node a cell, fired by the foot current at t_a."""
    v_ode = dolfinx.fem.Function(dolfinx.fem.functionspace(geo_ep.mesh, ("P", 1)))
    num_points = v_ode.x.array.size
    states = ep_module.init_state_values(**({"B": 1.0} if "B" in ep_module.state else {}))
    parameters = ep_module.init_parameter_values(
        i_Stim_Amplitude=0.0,
        **on_side(calibration, ep_module),
    )
    num_missing = len(ep_module.missing)
    ode = beat.odesolver.DolfinODESolver(
        v_ode=v_ode,
        v_pde=dolfinx.fem.Function(v_ode.function_space),
        fun=getattr(ep_module, EP_SCHEME),
        init_states=np.tile(states[:, None], (1, num_points)),
        parameters=np.tile(parameters[:, None], (1, num_points)),
        num_states=len(ep_module.state),
        v_index=ep_module.state_index("v"),
        missing_variables=np.zeros((num_missing, num_points)),
        num_missing_variables=num_missing,
    )
    return beat.SplittingSolver(
        pde=None,
        ode=FootCurrentODESolver(ode, t_a, ExponentialFoot()),
        splitting="strang",
    )


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


def mechanics(mech_module, calibration):
    geo = beat.geometry.get_3D_slab_geometry(MPI.COMM_WORLD, Lx=LX, Ly=LY, Lz=LZ, dx=DX_MECH)
    backend = GeneratedActivation(
        mech_module,
        geo.mesh,
        geo.f0,
        quadrature_degree=QUADRATURE_DEGREE,
        scheme="monolithic",
        parameters=on_side(calibration, mech_module),
        tension="Ta",
        tension_unit="kPa",
    )
    if "B" in mech_module.state:
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
    return backend, problem


# %% [markdown]
# Each run records, after every mechanics step, $v$ and the calcium at every EP node, and
# the tension and λ at every quadrature point. At the end it also keeps the displacement
# and the tension at the nodes of the mechanics mesh, for the view below. The calibration
# goes to the composed model only: the hand-made files have their own values, in other
# units.


# %%
def run_coupled(modules, calibration):
    ep = ep_solver(modules.ep, calibration)
    backend, problem = mechanics(modules.mechanics, calibration)
    controller = SimulationController(
        mechanics=problem,
        ep_solver=ep,
        backend=backend,
        ode_modules=modules,
        dt_mech=DT_MECH,
        dt_ep=DT_EP,
    )
    i_v, i_cai = modules.ep.state_index("v"), modules.ep.state_index("cai")
    trace: dict = {"t": [], "v": [], "cai": [], "Ta": [], "lmbda": []}

    def on_mech_step(t, mech_step, newton_iterations):
        states = dolfin_ode_solver(ep).full_values
        trace["t"].append(t)
        trace["v"].append(states[i_v].copy())
        trace["cai"].append(states[i_cai].copy())
        trace["Ta"].append(backend.tension_kPa.x.array.copy())
        trace["lmbda"].append(backend.lmbda_prev.x.array.copy())

    while controller.t < T_END - 1e-9:
        controller.step(mech_callback=on_mech_step)
    result = {key: np.array(value) for key, value in trace.items()}
    space = backend.active_tension.function_space
    u_p1 = dolfinx.fem.Function(dolfinx.fem.functionspace(space.mesh, ("P", 1, (3,))))
    u_p1.interpolate(problem.u)
    result["end"] = {
        "space": space,
        "u": u_p1.x.array.reshape(-1, 3).copy(),
        "Ta": backend.active_tension.x.array.copy(),
    }
    return result


coupled = {}
for split in SPLITS:
    for label, modules, calibration in (
        ("composed", composed[split], LAND_CALIBRATION),
        ("hand-made", hand_made[split], {}),
    ):
        start = time.perf_counter()
        coupled[split, label] = run_coupled(modules, calibration)
        peak = coupled[split, label]["Ta"].mean(axis=1).max()
        print(
            f"{split:<5}{label:<10}: {time.perf_counter() - start:5.1f} s, "
            f"peak mean Ta {peak:.3f} kPa",
        )
        # The slab contracts, so the comparisons below are not between two runs at rest
        assert peak > 1.0, (split, label, peak)

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for i, split in enumerate(SPLITS):
    for label, style in (("composed", "-"), ("hand-made", "--")):
        result = coupled[split, label]
        name = f"{split}, {label}"
        axes[0].plot(result["t"], result["Ta"].mean(axis=1), style, color=f"C{i}", label=name)
        axes[1].plot(result["t"], result["lmbda"].mean(axis=1), style, color=f"C{i}", label=name)
axes[0].set(ylabel="mean $T_a$ [kPa]", title="Active tension, mean over the slab")
axes[1].set(ylabel="mean $\\lambda$ [-]", title="Fibre stretch, mean over the slab")
axes[0].legend(frameon=False)
for ax in axes:
    ax.set_xlabel("time [ms]")
fig.tight_layout()
plt.show()

# %% [markdown]
# The zeta split at 100 ms, composed beside hand-made: the mechanics mesh warped by the
# displacement at true scale and coloured by $T_a$, with the slab at rest in grey. Drag
# to turn either and scroll to zoom; the two views turn together. The tension is highest
# in the stimulated corner, which activated first, and the slab has shortened along its
# fibres. The two are hard to tell apart, and the largest differences between them,
# printed below the view, are small beside the tension and the displacement themselves.
#
# `surface` and `show_scene` are copied from
# [the reaction-eikonal tutorial](07_reaction_eikonal_slab.ipynb), which explains them.

# %%
OUTPUT = HERE / "output"
OUTPUT.mkdir(exist_ok=True)


def surface(space):
    """The surface of a P1 space's mesh, and the dof at each of its points."""
    grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(space))
    boundary = grid.extract_surface(
        pass_pointid=True,
        pass_cellid=False,
        algorithm="dataset_surface",
    )
    return boundary, boundary["vtkOriginalPointIds"]


def show_scene(plotter, name, height=400):
    """Show the plotter's scene as an interactive page, and write it to OUTPUT / name."""
    plotter.render()
    initialize_serializers()
    context = SynchronizationContext()
    context.set_ignore_last_dependencies(True)  # all of it, not what changed since a last call
    window = plotter.render_window
    scene = serialize(None, window, "scene", context, 1)
    props = [prop for renderer in window.GetRenderers() for prop in renderer.GetViewProps()]
    found = {prop["id"] for renderer in scene["dependencies"] for prop in renderer["dependencies"]}
    missing = [prop.GetClassName() for prop in props if reference_id(prop) not in found]
    assert not missing, f"trame-vtk cannot serialize {missing}"
    camera = window.GetRenderers().GetFirstRenderer().GetActiveCamera()
    scene["extra"] = {
        "vtkRefId": reference_id(window),
        "centerOfRotation": camera.GetFocalPoint(),
        "camera": reference_id(camera),
    }
    arrays = {
        item["hash"]: {**item, "content": context.get_cached_data_array(item["hash"], False)}
        for item in extract_array_hash(scene)
    }
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as file:
        file.writestr("index.json", json.dumps({"hashes": arrays, "scene": scene}))
    page = io.StringIO()
    write_html(archive.getvalue(), page)
    (OUTPUT / name).write_text(page.getvalue(), encoding="utf-8")
    attributes = f'width="100%" height="{height}" style="border:none"'
    frame = f'<iframe srcdoc="{html.escape(page.getvalue())}" {attributes}></iframe>'
    # In a <div>, since IPython warns about HTML that starts with an <iframe>
    display(HTML(f"<div>{frame}</div>"))


LABELS = ("composed", "hand-made")
top = max(coupled["zeta", label]["end"]["Ta"].max() for label in LABELS)
plotter = pyvista.Plotter(
    off_screen=True,
    shape=(1, 2),
    window_size=[900, 400],
    border=False,
    theme=pyvista.themes.DocumentTheme(),
)
for column, label in enumerate(LABELS):
    end = coupled["zeta", label]["end"]
    mesh, dofs = surface(end["space"])
    plotter.subplot(0, column)
    plotter.add_mesh(mesh.outline(), color="grey")
    mesh.points += end["u"][dofs]
    mesh["Ta"] = end["Ta"][dofs]
    plotter.add_mesh(
        mesh,
        scalars="Ta",
        clim=(0.0, top),
        cmap="viridis",
        scalar_bar_args={"title": f"Ta [kPa], zeta split, {label}"},
        ambient=0.3,
        diffuse=0.7,
        specular=0.0,
    )
plotter.link_views()
plotter.camera_position = [(-14.0, -24.0, 20.0), (LX / 2, LY / 2, LZ / 2), (0.0, 0.0, 1.0)]
show_scene(plotter, "08_tension.html")
plotter.close()
del plotter  # now, rather than at exit, where pyvista prints errors freeing it

a, b = (coupled["zeta", label]["end"] for label in LABELS)
print(
    f"at {T_END:.0f} ms, max |ΔTa| = {np.abs(a['Ta'] - b['Ta']).max():.2f} kPa, of up to "
    f"{top:.1f} kPa, and max |Δu| = {np.linalg.norm(a['u'] - b['u'], axis=1).max():.3f} mm, "
    f"of up to {np.linalg.norm(a['u'], axis=1).max():.1f} mm",
)

# %% [markdown]
# The largest differences over the run, composed against hand-made, at every node or
# quadrature point, and of the means over the slab. The tolerances are again twice those
# measured when the tutorial was written, for both splits.

# %%
COUPLED_TOLERANCE = {  # mV, mM, kPa, kPa, -, -
    "mean v": 0.22,
    "cai": 5.0e-5,
    "Ta": 0.23,
    "mean Ta": 0.054,
    "lmbda": 0.0051,
    "mean lmbda": 0.0023,
}
print(f"{'max |Δ|':<10}" + "".join(f"{name:>11}" for name in ("v", *COUPLED_TOLERANCE)))
for split in SPLITS:
    a, b = coupled[split, "composed"], coupled[split, "hand-made"]
    largest = {"v": np.abs(a["v"] - b["v"]).max()}
    for name in COUPLED_TOLERANCE:
        key = name.removeprefix("mean ")
        if name.startswith("mean "):
            largest[name] = np.abs(a[key].mean(axis=1) - b[key].mean(axis=1)).max()
        else:
            largest[name] = np.abs(a[key] - b[key]).max()
    print(f"{split:<10}" + "".join(f"{value:11.3g}" for value in largest.values()))
    for name, tolerance in COUPLED_TOLERANCE.items():
        assert largest[name] < tolerance, (split, name, largest[name])
print(f"{'tolerance':<10}{'—':>11}" + "".join(f"{t:11.3g}" for t in COUPLED_TOLERANCE.values()))
print()
for split in SPLITS:
    a, b = coupled[split, "composed"], coupled[split, "hand-made"]
    when = a["t"][np.abs(a["Ta"] - b["Ta"]).max(axis=1).argmax()]
    when_mean = a["t"][np.abs(a["Ta"].mean(axis=1) - b["Ta"].mean(axis=1)).argmax()]
    print(f"{split}: largest |ΔTa| at {when:.0f} ms, largest |Δ mean Ta| at {when_mean:.0f} ms")

# %% [markdown]
# Both splits follow their hand-made counterparts to within about 1 % of the peak mean
# tension on this ~2 mm mechanics mesh. The differences in the tension are largest at the
# end of the run (printed above), while the tension is still rising; in the single cell,
# the difference in $T_a$ peaks at 176 ms.
#
# $v$ is compared by its mean, not node by node. The composed cell's upstroke is a
# fraction of a millisecond earlier (measured when this tutorial was written), so a node
# caught in the middle of its upstroke differs by over 10 mV, which says nothing about
# the agreement.
#
# What remains is the single cell's difference, carried into the tissue. With the same
# four patches as above, the coupled runs of both splits agree with the hand-made files to
# round-off too ($|\Delta T_a|$ below 1e-9 kPa, $|\Delta\lambda|$ below 1e-11), which was
# also measured once, offline. The other differences between the models, which could
# show only once λ moves, do not reach these numbers:
#
# - `Min` and `Max` against `Conditional`: the same function, now with λ moving through
#   it, and the composed model's guards, which differ but never trigger in these runs;
# - Land's passive part, formulated differently, which the mechanics do not use;
# - `Cd`'s placement: the composed zeta split keeps it on the EP side where the hand-made
#   zeta file has it in `mechanics`, and the composed Ca$_i$ split sends λ to EP for it,
#   where the hand-made one sends none.
#
# ## Where to go next
#
# - A split is a choice of what is inside Newton, and these two are not the only ones.
#   `inputs=["CaTRPN"], outputs=["Ta"]` cuts the composed model at Land's troponin, as the
#   hand-made `catrpnsplit` file of [the cell-model tutorial](01_cell_model.ipynb) does,
#   and `resolve` says what then crosses.
# - [Why the coupling scheme matters](02_coupling_schemes_0d.ipynb) shows, in 0D, why it
#   matters what is lagged a step: a contraction model stepped at the previous step's
#   stretch does not converge once its active stiffness exceeds the passive stiffness.
