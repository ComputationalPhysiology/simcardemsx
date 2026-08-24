# # The return path: why mechanics has to talk back
#
# Coupling electrophysiology to mechanics is usually described in one direction:
# calcium drives contraction. The return path gets less attention, and it is easy to
# leave out — the simulation runs perfectly well without it.
#
# It is also, depending on where the model is split, either an approximation or an
# outright error. This demo shows both cases.
#
# For most of this package's history the return path was wired but transferred
# **zeros**: the buffer it read from was never written. Every zeta-split simulation
# ran with no distortion feedback at all, and nothing reported it. That is the kind
# of bug this demo exists to make visible.

from pathlib import Path

from mpi4py import MPI

import dolfinx
import numpy as np
import pulse

from simcardemsx.backends import CrossbridgeSegregated
from simcardemsx.controller import SimulationController
from simcardemsx.ode_model import RuntimeODEModel, load_ode_modules

ODEFILES = Path("../numerical_experiments/odefiles")

# ## What crosses, and which way
#
# Under a **Ca_i split**, calcium goes to the contraction model, which then owns
# troponin. The EP model has had its own troponin removed, so its calcium balance is
# missing a sink. The buffering flux `J_TRPN` has to come back, or calcium is simply
# unbuffered.
#
# Under a **ζ's split**, the crossbridge populations `XS`/`XW` go out and the
# distortion states `Zetas`/`Zetaw` come back. That return path is genuine
# mechano-electric feedback rather than a conservation requirement.

modules = load_ode_modules(ODEFILES / "ToRORd_dynCl_endo_caisplit.ode", Path("_generated/cai"))
print("EP is missing:       ", list(modules.ep.missing))
print("mechanics is missing:", list(modules.mechanics.missing))


# ## A minimal coupled run
#
# One element, one cell's worth of electrophysiology. Small enough to run in
# seconds, and the point survives.


class ODELevelEPSolver:
    """Integrates the generated EP ODE, reading its missing variables from the
    array the coupler fills. That reference is the whole mechanism: a return path
    that delivers zeros shows up here as an EP solution that never sees them."""

    def __init__(self, module, V, missing_variables):
        index_map = V.dofmap.index_map
        n = (index_map.size_local + index_map.num_ghosts) * V.dofmap.index_map_bs
        self._values = np.tile(module.init_state_values()[:, None], (1, n)).astype(float)
        self.parameters = np.tile(module.init_parameter_values()[:, None], (1, n)).astype(float)
        self._fgr = module.generalized_rush_larsen
        self._missing = missing_variables
        self.module = module
        solver = self

        class _ODE:
            _values = property(lambda s: solver._values)
            parameters = property(lambda s: solver.parameters)

        self.ode = _ODE()

    def state(self, name):
        return self._values[self.module.state_index(name)]

    def step(self, t_span):
        t0, t1 = t_span
        self._values[:] = self._fgr(self._values, t0, t1 - t0, self.parameters, self._missing)


def build(sever_return_path=False):
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
    backend = CrossbridgeSegregated(f0=f0, mesh=mesh_m)

    material = pulse.HolzapfelOgden(
        f0=f0, s0=s0, **pulse.HolzapfelOgden.transversely_isotropic_parameters(),
    )
    model = pulse.CardiacModel(
        material=material, active=backend, compressibility=pulse.Incompressible(),
    )

    def bcs(V):
        V0, _ = V.sub(0).collapse()
        zero = dolfinx.fem.Function(V0)
        fdim = mesh_m.topology.dim - 1
        out = []
        for sub in range(3):
            facets = dolfinx.mesh.locate_entities_boundary(
                mesh_m, fdim, lambda x, s=sub: np.isclose(x[s], 0.0),
            )
            out.append(
                dolfinx.fem.dirichletbc(
                    zero,
                    dolfinx.fem.locate_dofs_topological((V.sub(sub), V0), fdim, facets),
                    V.sub(sub),
                ),
            )
        return out

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
        dt_mech=1.0,
        dt_ep=0.1,
    )

    if sever_return_path:
        # Exactly the bug that shipped: the coupler still runs the transfer, but
        # what arrives is zeros.
        original = controller.step

        def step(*a, **k):
            original(*a, **k)
            ode_model.missing_ep.values_ep[:] = 0.0

        controller.step = step

    return controller, ep, backend


# ## With and without
#
# The two runs are identical except that one discards the buffering flux on its way
# back to the EP model.

N = 15
histories = {}
for label, sever in [("with feedback", False), ("without", True)]:
    controller, ep, backend = build(sever_return_path=sever)
    cai = []
    for _ in range(N):
        controller.step()
        cai.append(float(np.mean(ep.state("cai"))))
    histories[label] = np.array(cai)
    print(f"{label:15s} final [Ca]i = {cai[-1]:.6e} mM")

drift = histories["without"][-1] - histories["with feedback"][-1]
rel = drift / histories["with feedback"][-1]
print(f"\ndiscarding the flux leaves {rel:+.2%} more calcium after {N} ms")

# The gap grows with every beat, because the missing term is a *sink*: calcium that
# should have been bound to troponin stays in the cytosol. Nothing raises, and the
# transient still looks like a calcium transient.

# ## The lesson
#
# A return path that silently delivers nothing is indistinguishable, at a glance,
# from one that works. Both give a running simulation and a plausible-looking
# result. The only defence is to assert that what arrives is what was sent —
# which is what `tests/test_coupler.py` now does for both splits.
