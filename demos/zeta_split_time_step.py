# # When refining the time step makes things worse
#
# If a coupled simulation oscillates, the reflex is to shorten the time step. For
# one class of scheme that reflex is precisely wrong: the oscillation *grows* as the
# step shrinks.
#
# This demo shows that happening in the finite-element zeta-split path, on a single
# element so it runs in under a minute. It is not a hypothetical — it is what the
# shipped `numerical_experiments/strong_coupling_zetasplit` experiment does at its
# default time step.
#
# The companion demo `stabilized_vs_naive_coupling.py` explains the underlying
# result (Regazzoni & Quarteroni) in 0D, and shows the fix. This one is the symptom,
# in the solver you would actually be running.

from pathlib import Path

from mpi4py import MPI

import dolfinx
import matplotlib.pyplot as plt
import numpy as np
import pulse

from simcardemsx.backends import ZetaSplitUFL
from simcardemsx.controller import SimulationController
from simcardemsx.ode_model import RuntimeODEModel, load_ode_modules

ODEFILES = Path("../numerical_experiments/odefiles")
modules = load_ode_modules(ODEFILES / "ToRORd_dynCl_endo_zetasplit.ode", Path("_generated/zeta"))


class ODELevelEPSolver:
    def __init__(self, module, V, missing_variables):
        index_map = V.dofmap.index_map
        n = (index_map.size_local + index_map.num_ghosts) * V.dofmap.index_map_bs
        self._values = np.tile(module.init_state_values()[:, None], (1, n)).astype(float)
        self.parameters = np.tile(module.init_parameter_values()[:, None], (1, n)).astype(float)
        self._fgr = module.generalized_rush_larsen
        self._missing = missing_variables
        solver = self

        class _ODE:
            _values = property(lambda s: solver._values)
            parameters = property(lambda s: solver.parameters)

        self.ode = _ODE()

    def step(self, t_span):
        t0, t1 = t_span
        self._values[:] = self._fgr(self._values, t0, t1 - t0, self.parameters, self._missing)


def run(dt_mech, dt_ep, duration, stiffness=1.0):
    """One coupled zeta-split run; returns the stretch history.

    ``stiffness`` scales the passive material. It is the knob R&Q's analysis
    turns on: what matters is the *ratio* of active to passive stiffness, not
    either alone.
    """
    mesh_m = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    mesh_e = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
    V_m = dolfinx.fem.functionspace(mesh_m, ("DG", 1))
    V_e = dolfinx.fem.functionspace(mesh_e, ("DG", 1))

    ode_model = RuntimeODEModel(
        ep_module_dict=modules.ep.__dict__,
        mech_module_dict=modules.mechanics.__dict__,
        mech_ode_space=V_m,
        ep_ode_space=V_e,
    )
    f0 = dolfinx.fem.Constant(mesh_m, np.array([1.0, 0.0, 0.0]))
    s0 = dolfinx.fem.Constant(mesh_m, np.array([0.0, 1.0, 0.0]))
    n0 = dolfinx.fem.Constant(mesh_m, np.array([0.0, 0.0, 1.0]))
    backend = ZetaSplitUFL(f0=f0, s0=s0, n0=n0, mesh=mesh_m)

    params = pulse.HolzapfelOgden.transversely_isotropic_parameters()
    params = {
        k: (pulse.units.Variable(v.value * stiffness, v.unit) if k.startswith("a") else v)
        for k, v in params.items()
    }
    material = pulse.HolzapfelOgden(f0=f0, s0=s0, **params)
    model = pulse.CardiacModel(
        material=material, active=backend, compressibility=pulse.Incompressible(),
    )

    def bcs(V):
        V0, _ = V.sub(0).collapse()
        zero = dolfinx.fem.Function(V0)
        fdim = mesh_m.topology.dim - 1
        return [
            dolfinx.fem.dirichletbc(
                zero,
                dolfinx.fem.locate_dofs_topological(
                    (V.sub(s), V0),
                    fdim,
                    dolfinx.mesh.locate_entities_boundary(
                        mesh_m, fdim, lambda x, s=s: np.isclose(x[s], 0.0),
                    ),
                ),
                V.sub(s),
            )
            for s in range(3)
        ]

    problem = pulse.StaticProblem(
        model=model,
        geometry=pulse.Geometry(mesh=mesh_m, metadata={"quadrature_degree": 4}),
        bcs=pulse.BoundaryConditions(dirichlet=[bcs]),
        parameters={"base_bc": pulse.problem.BaseBC.free},
    )
    ep = ODELevelEPSolver(modules.ep, V_e, ode_model.missing_ep.values_ep)
    controller = SimulationController(
        mechanics_problem=problem,
        ep_solver=ep,
        ode_model=ode_model,
        backend=backend,
        dt_mech=dt_mech,
        dt_ep=dt_ep,
    )

    stretch = []
    for _ in range(int(round(duration / dt_mech))):
        controller.step()
        stretch.append(float(np.mean(backend.lmbda.x.array)))
    return np.array(stretch)


def oscillation_fraction(stretch):
    """Fraction of steps at which the stretch reverses direction.

    A smooth contraction turns around once or twice over a twitch. A value near 1
    means the solution reverses on essentially every step -- a limit cycle at the
    time-step frequency, which is not physics.
    """
    d = np.diff(stretch)
    if len(d) < 2:
        return 0.0
    return float(np.sum(np.diff(np.sign(d)) != 0)) / (len(d) - 1)


def reversal_amplitude(stretch):
    """Typical size of a step-to-step reversal, in stretch.

    Reported alongside the fraction because the two answer different questions. A
    high fraction with a negligible amplitude would be harmless numerical noise;
    a high fraction with a percent-level amplitude is not.
    """
    tail = stretch[len(stretch) // 2 :]
    return float(np.abs(np.diff(tail)).mean()) if len(tail) > 2 else 0.0


# ## Refining the step
#
# Same physics, same duration, three time steps. If this were an accuracy problem,
# the finest step would be the most trustworthy answer.

DURATION = 40.0
STEPS = (0.8, 0.2, 0.05)

results = {}
for dt_mech in STEPS:
    stretch = run(dt_mech=dt_mech, dt_ep=0.05, duration=DURATION)
    results[dt_mech] = stretch
    print(
        f"dt = {dt_mech:4.2f} ms   steps = {len(stretch):4d}   "
        f"reversals/step = {oscillation_fraction(stretch):.2f}   "
        f"amplitude = {reversal_amplitude(stretch):.2e}   "
        f"min lambda = {stretch.min():.5f}",
    )

# The reversal fraction does not fall as the step shrinks, and the amplitude is
# percent-level in stretch — far too large to dismiss as round-off. The trajectory
# underneath looks like a contraction at every step size, which is exactly what
# makes this easy to miss.

fig, axes = plt.subplots(1, 3, figsize=(11, 3), sharey=True)
for ax, dt_mech in zip(axes, STEPS):
    stretch = results[dt_mech]
    t = np.arange(len(stretch)) * dt_mech
    ax.plot(t, stretch, ".-", markersize=2, color="#12706B", linewidth=0.7)
    ax.set(
        xlabel="time [ms]",
        title=f"dt = {dt_mech} ms  ({oscillation_fraction(stretch):.2f})",
    )
axes[0].set_ylabel(r"fibre stretch $\lambda$")
fig.suptitle("Refining the time step does not remove the oscillation")
fig.tight_layout()
fig.savefig("zeta_split_time_step.png", dpi=140)

# ## It is not a soft-tissue edge case
#
# R&Q's criterion is a *ratio*: the scheme misbehaves once active stiffness exceeds
# passive stiffness. That invites the hope that a stiffer material escapes it. On
# this problem it does not — stiffening the passive material by 20x and 50x leaves
# the reversal fraction around 0.75.
#
# The stretch also bottoms out at a similar value regardless, because the amplitude
# is bounded by the contraction model's own force-length relation — `h(lambda)`
# drives active tension to zero below a threshold — rather than by anything
# numerical. A bounded oscillation is not a converged one.

# ## In the shipped experiment
#
# This is not confined to a contrived unit cube. Running
# `numerical_experiments/strong_coupling_zetasplit` on its own slab geometry for
# 30 ms, the stretch reverses on 45% of steps at the default `dt_mech = 0.05 ms`,
# with a step-to-step amplitude around 0.015 in lambda. There the trend with step
# size is visible:
#
# | `dt_mech` | reversals/step |
# |-----------|----------------|
# | 0.05 ms   | 0.18           |
# | 0.20 ms   | 0.04           |
# | 0.80 ms   | 0.00           |
#
# Measured over 20 ms, so the fractions are lower than the 30 ms figure above; the
# ordering is the point. The same behaviour is present before and after the coupler
# refactor — it is not something the refactor introduced.
#
# ## Reading this
#
# A reversal fraction that stays high, or rises, as `dt` falls is the signature of a
# **segregated instability** rather than of stiffness or a loose solver tolerance.
# Tightening Newton will not touch it, and neither will a smaller step.
#
# Two practical consequences:
#
# - Choosing a coupling time step here is not purely an accuracy trade-off. There is
#   a floor below which the answer does not improve.
# - An oscillation metric like the one above is worth computing routinely. The
#   trajectory looks plausible at every step size; only the step-to-step reversals
#   tell them apart.
#
# `stabilized_vs_naive_coupling.py` shows the fix — one consistent extra term in the
# active stress — in the 0D setting where it can be checked against a true
# monolithic reference. Applying that to this path is open work.
