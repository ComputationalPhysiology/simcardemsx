# # A full coupling: monodomain electrophysiology and crossbridge mechanics
#
# The other demos here isolate one piece at a time and drive electrophysiology at
# the ODE level, one cell per degree of freedom. This one runs the real thing: a
# **monodomain** solve from [fenicsx-beat](https://github.com/finsberg/fenicsx-beat),
# so the activation wavefront actually propagates through the tissue, coupled to a
# `crossbridge` contraction model through the Ca_i split.
#
# That combination is the point of the package. It is also the combination that was
# not previously reachable — the crossbridge backend existed but nothing could hand
# it calcium produced by an EP solve.
#
# The two physics live on **different meshes**, which is normal: electrophysiology
# needs a fine mesh to resolve the wavefront, mechanics does not. The coupler moves
# values between them.

from pathlib import Path

from mpi4py import MPI

import beat
import dolfinx
import matplotlib.pyplot as plt
import numpy as np
import pulse
import ufl

from simcardemsx.backends import CrossbridgeSegregated
from simcardemsx.controller import SimulationController
from simcardemsx.ode_model import RuntimeODEModel, load_ode_modules

comm = MPI.COMM_WORLD

# ## Geometry
#
# A 2 x 1 x 1 mm slab. `beat` works in millimetres, so everything below is in mm,
# ms, and mV.

LX, LY, LZ = 2.0, 1.0, 1.0

ep_mesh = dolfinx.mesh.create_box(
    comm, [[0, 0, 0], [LX, LY, LZ]], [12, 6, 6], dolfinx.mesh.CellType.tetrahedron,
)
mech_mesh = dolfinx.mesh.create_box(
    comm, [[0, 0, 0], [LX, LY, LZ]], [4, 2, 2], dolfinx.mesh.CellType.tetrahedron,
)
print(f"EP mesh:        {ep_mesh.topology.index_map(3).size_global:5d} cells")
print(f"mechanics mesh: {mech_mesh.topology.index_map(3).size_global:5d} cells")

f0_ep = dolfinx.fem.Constant(ep_mesh, np.array([1.0, 0.0, 0.0]))
f0 = dolfinx.fem.Constant(mech_mesh, np.array([1.0, 0.0, 0.0]))
s0 = dolfinx.fem.Constant(mech_mesh, np.array([0.0, 1.0, 0.0]))

# ## Splitting the cell model
#
# The `.ode` file is cut at intracellular calcium. `gotranx` works out what has to
# cross from where the cut falls: calcium goes out to the contraction model, and the
# troponin buffering flux has to come back, because the contraction model now owns
# troponin and the EP model's calcium balance is missing that sink.

modules = load_ode_modules(
    Path("../numerical_experiments/odefiles/ToRORd_dynCl_endo_caisplit.ode"),
    Path("_generated/monodomain"),
)
print("mechanics needs from EP:", list(modules.mechanics.missing))
print("EP needs back:          ", list(modules.ep.missing))

ep_ode_space = dolfinx.fem.functionspace(ep_mesh, ("Lagrange", 1))
mech_ode_space = dolfinx.fem.functionspace(mech_mesh, ("DG", 1))

ode_model = RuntimeODEModel(
    ep_module_dict=modules.ep.__dict__,
    mech_module_dict=modules.mechanics.__dict__,
    mech_ode_space=mech_ode_space,
    ep_ode_space=ep_ode_space,
)

# ## The monodomain solver
#
# Conductivities and surface-to-volume ratio are the usual human-ventricle values.
# The stimulus is applied to a small block at one end, so the wave has somewhere to
# travel from.

mesh_unit = "mm"
chi = 140.0 * beat.units.ureg("mm**-1")
C_m = 0.01 * beat.units.ureg("uF/mm**2")

M = beat.conductivities.define_conductivity_tensor(
    chi=chi,
    f0=f0_ep,
    g_il=0.17 * beat.units.ureg("S/m"),
    g_it=0.019 * beat.units.ureg("S/m"),
    g_el=0.62 * beat.units.ureg("S/m"),
    g_et=0.24 * beat.units.ureg("S/m"),
)

stim_marker = 1
stim_cells = dolfinx.mesh.locate_entities(ep_mesh, 3, lambda x: x[0] <= 0.3 + 1e-6)
stim_tags = dolfinx.mesh.meshtags(
    ep_mesh, 3, stim_cells, np.full(len(stim_cells), stim_marker, dtype=np.int32),
)

time = dolfinx.fem.Constant(ep_mesh, 0.0)
I_s = beat.stimulation.define_stimulus(
    mesh=ep_mesh,
    chi=chi,
    time=time,
    subdomain_data=stim_tags,
    marker=stim_marker,
    mesh_unit=mesh_unit,
    # Higher than the shipped experiment's 50,000, because that one stimulates
    # essentially its whole mesh while this stimulates one small block in order to
    # show a wave propagating. A smaller region needs a higher current density to
    # cross threshold; at 50,000 here the tissue depolarizes to about -66 mV and
    # then quietly recovers, with no action potential and nothing to report it.
    amplitude=200_000.0 * beat.units.ureg("uA/cm**3"),
)

pde = beat.MonodomainModel(
    time=time,
    mesh=ep_mesh,
    M=M,
    I_s=I_s,
    C_m=C_m.to(f"uF/{mesh_unit}**2").magnitude,
    dx=ufl.Measure("dx", domain=ep_mesh, subdomain_data=stim_tags),
)

# The cell model's own stimulus current is switched **off**: `beat` applies the
# stimulus through the PDE, and leaving both on would stimulate twice.

n_ep = ep_ode_space.dofmap.index_map.size_local + ep_ode_space.dofmap.index_map.num_ghosts
y0 = ode_model.y()
p0 = ode_model.p(i_Stim_Amplitude=0.0)
y_ep = np.tile(y0[:, None], (1, n_ep))
p_ep = np.tile(p0[:, None], (1, n_ep))

ode = beat.odesolver.DolfinODESolver(
    v_ode=dolfinx.fem.Function(ep_ode_space),
    v_pde=pde.state,
    fun=ode_model.fgr,
    init_states=y_ep,
    parameters=p_ep,
    num_states=len(y0),
    v_index=modules.ep.state_index("v"),
    # The coupler fills this array every mechanics step. Passing it here is what
    # closes the loop: without it the EP model never sees the buffering flux.
    missing_variables=ode_model.missing_ep.values_ep,
    num_missing_variables=ode_model.missing_ep.num_values,
)

ep_solver = beat.MonodomainSplittingSolver(pde=pde, ode=ode, theta=1)

# ## The mechanics problem
#
# Stock `pulse.StaticProblem`. The activation backend *is* the active model, so
# swapping contraction models is a one-line change here and nothing else moves.

backend = CrossbridgeSegregated(f0=f0, mesh=mech_mesh, model="Land2017")

material = pulse.HolzapfelOgden(
    f0=f0, s0=s0, **pulse.HolzapfelOgden.transversely_isotropic_parameters(),
)
cardiac_model = pulse.CardiacModel(
    material=material, active=backend, compressibility=pulse.Incompressible(),
)


def dirichlet_bc(V):
    """Symmetry planes: the slab is free to shorten along the fibre."""
    V0, _ = V.sub(0).collapse()
    zero = dolfinx.fem.Function(V0)
    fdim = mech_mesh.topology.dim - 1
    return [
        dolfinx.fem.dirichletbc(
            zero,
            dolfinx.fem.locate_dofs_topological(
                (V.sub(s), V0),
                fdim,
                dolfinx.mesh.locate_entities_boundary(
                    mech_mesh, fdim, lambda x, s=s: np.isclose(x[s], 0.0),
                ),
            ),
            V.sub(s),
        )
        for s in range(3)
    ]


problem = pulse.StaticProblem(
    model=cardiac_model,
    geometry=pulse.Geometry(mesh=mech_mesh, metadata={"quadrature_degree": 4}),
    bcs=pulse.BoundaryConditions(dirichlet=[dirichlet_bc]),
    parameters={"base_bc": pulse.problem.BaseBC.free},
)

# ## Running
#
# The controller does the whole step: EP micro-steps, transfer in, advance
# activation, one mechanics solve, transfer back.

DT_EP = 0.05
DT_MECH = 1.0
DURATION = 200.0

controller = SimulationController(
    mechanics_problem=problem,
    ep_solver=ep_solver,
    ode_model=ode_model,
    backend=backend,
    dt_mech=DT_MECH,
    dt_ep=DT_EP,
)

v_index = modules.ep.state_index("v")
recorded: dict[str, list[float]] = {k: [] for k in ("t", "v", "cai", "Ta", "lmbda", "J_TRPN")}

for _ in range(int(DURATION / DT_MECH)):
    controller.step()
    states = ep_solver.ode._values
    recorded["t"].append(controller.t)
    recorded["v"].append(float(np.mean(states[v_index])))
    recorded["cai"].append(float(np.mean(backend.cai.x.array)))
    recorded["Ta"].append(float(np.mean(backend.active_tension.x.array)))
    recorded["lmbda"].append(float(np.mean(backend.lmbda.x.array)))
    recorded["J_TRPN"].append(float(np.mean(backend.J_TRPN.x.array)))

history = {k: np.asarray(v) for k, v in recorded.items()}

print(f"peak v      {history['v'].max():8.2f} mV")
print(f"peak [Ca]i  {history['cai'].max():8.3e} mM   (as transferred to crossbridge)")
print(f"peak Ta     {history['Ta'].max():8.3f} kPa")
print(f"min lambda  {history['lmbda'].min():8.5f}")

# ## What came out
#
# Four quantities, each living on a different side of the coupling: voltage is
# solved by `beat` on the EP mesh, calcium crosses to the mechanics mesh, tension is
# produced by `crossbridge`, and stretch comes back out of the `pulse` solve.

fig, axes = plt.subplots(2, 2, figsize=(9, 5), sharex=True)
for ax, (key, label, colour) in zip(
    axes.flat,
    [
        ("v", "transmembrane potential [mV]", "#12706B"),
        ("cai", r"[Ca$^{2+}$]$_i$ [mM]", "#B33F0C"),
        ("Ta", "active tension [kPa]", "#12706B"),
        ("lmbda", r"fibre stretch $\lambda$", "#B33F0C"),
    ],
):
    ax.plot(history["t"], history[key], color=colour)
    ax.set_ylabel(label, fontsize=9)
    ax.grid(alpha=0.25)
for ax in axes[1]:
    ax.set_xlabel("time [ms]")
fig.suptitle("Monodomain EP driving a crossbridge contraction model")
fig.tight_layout()
fig.savefig("monodomain_coupling.png", dpi=140)

# ## The return path is live
#
# `J_TRPN` is not diagnostic output — it is transferred back to the EP model every
# step and enters its calcium balance. A non-zero value here means the loop is
# closed.

print(f"|J_TRPN| peak {np.abs(history['J_TRPN']).max():.3e} mM/ms")
assert np.abs(history["J_TRPN"]).max() > 0.0, "the buffering flux never left the backend"

# ## Is the tension right?
#
# A quarter of a kPa looks small next to the tens of kPa Land2017 is usually quoted
# at. Those figures are for maximal activation; this is one beat reaching about
# 0.9 µM, which is submaximal, and the tissue is shortening, which lowers tension
# further through the force-length relation.
#
# Worth checking rather than assuming. Drive the same model standalone with the same
# calcium, holding sarcomere length fixed, and compare:

import crossbridge  # noqa: E402

from simcardemsx import units  # noqa: E402

standalone = crossbridge.get_model("Land2017")(num_cells=1)
SL_ref = standalone.p["SL0"]
peak_isometric = 0.0
for cai in history["cai"]:
    standalone.advance_step(
        units.ms_to_s(DT_MECH),
        units.calcium_to_crossbridge(np.array([cai])),
        np.array([SL_ref]),
        dSL_vals=np.zeros(1),
    )
    peak_isometric = max(peak_isometric, float(standalone.get_active_tension()[0]))

print(f"coupled (shortening) peak Ta  {history['Ta'].max():.4f} kPa")
print(f"standalone (isometric) peak   {peak_isometric:.4f} kPa")

# The coupled value is lower, which is the right direction: a fibre allowed to
# shorten develops less tension than one held at fixed length. If the two were
# wildly apart, or the coupled one were *higher*, something in the transfer would be
# wrong -- and that comparison is exactly what `tests/test_coupler.py` automates,
# clamping the stretch so the two must agree to solver tolerance.

# ## Swapping the contraction model
#
# Because the backend is the seam, changing which contraction model runs is a
# constructor argument. Nothing else in this file would change:
#
# ```python
# backend = CrossbridgeSegregated(f0=f0, mesh=mech_mesh, model="RDQ20MF")
# backend = CrossbridgeSegregated(f0=f0, mesh=mech_mesh, model="RDQ18", SL_ref=2.0)
# ```
#
# `RDQ18` needs an explicit `SL_ref` because it defines no slack sarcomere length of
# its own, and guessing one would silently move it along its force-length curve.
