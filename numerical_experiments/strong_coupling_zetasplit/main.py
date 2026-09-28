"""The worked example: EP and mechanics coupled on a slab, for any split of an ``.ode`` file.

EP (fenicsx-beat, monodomain) and mechanics (fenicsx-pulse, quasistatic) are coupled
through :class:`~simcardemsx.controller.SimulationController`, with the ``mechanics``
component of ``--odefile`` stepped inside the mechanics Newton iteration by
:class:`~simcardemsx.backends.GeneratedActivation`. Output goes to
``output/<odefile stem>/``, including ``timings.json``: the wall time spent in the EP
ODE step, the EP PDE step and the mechanics solve.
"""

import argparse
import functools
import json
import logging
import time
from pathlib import Path
from typing import Callable, NamedTuple

from mpi4py import MPI

import beat
import dolfinx
import numpy as np
import pulse
import ufl

import cardiac_geometries
from simcardemsx.averaging import make_averager
from simcardemsx.backends import GeneratedActivation
from simcardemsx.controller import SimulationController
from simcardemsx.datacollector import DataCollector
from simcardemsx.ode_model import load_ode_modules

logger = logging.getLogger(__name__)
QUAD_DEGREE = 4  # Degree of quadrature for the mechanics mesh
DEFAULT_ODEFILE = Path("../odefiles/ToRORd_dynCl_endo_zetasplit.ode")
SLAB_DX = 0.5  # Resolution of the slab mesh


def default_config():
    return {
        "ep": {
            "conductivities": {
                "sigma_el": 0.62,
                "sigma_et": 0.24,
                "sigma_il": 0.17,
                "sigma_it": 0.019,
            },
            "stimulus": {
                "amplitude": 50000.0,
                "duration": 2,
                "start": 0.0,
                "xmax": 1.5,
                "xmin": 0.0,
                "ymax": 1.5,
                "ymin": 0.0,
                "zmax": 1.5,
                "zmin": 0.0,
            },
            "chi": 140.0,
            "C_m": 0.01,
        },
        "mechanics": {
            "material": {
                "a": 2.28,
                "a_f": 1.686,
                "a_fs": 0.0,
                "a_s": 0.0,
                "b": 9.726,
                "b_f": 15.779,
                "b_fs": 0.0,
                "b_s": 0.0,
            },
            "bcs": [
                {"V": "u_x", "expression": 0, "marker": 1, "param_numbers": 0, "type": "Dirichlet"},
                {"V": "u_y", "expression": 0, "marker": 3, "param_numbers": 0, "type": "Dirichlet"},
                {"V": "u_z", "expression": 0, "marker": 5, "param_numbers": 0, "type": "Dirichlet"},
            ],
        },
        "sim": {
            "N": 1,
            "dt": 0.05,
            "mech_mesh": "meshes/mesh_mech_0.5dx_0.5Lx_1.0Ly_2.0Lz",
            "markerfile": "meshes/mesh_mech_0.5dx_0.5Lx_1.0Ly_2.0Lz_surface_ffun",
            "modelfile": "../odefiles/ToRORd_dynCl_endo_zetasplit.ode",
            "outdir": "output",
            "sim_dur": 40,
            "split_scheme": "cai",
            "save_frequency_ep": 20,
            "save_frequency_mech": 1,
        },
        "output": {
            "all_ep": ["v"],
            "all_mech": ["Ta", "lambda"],
            "point_ep": [
                {"name": "v", "x": 0, "y": 0, "z": 0},
            ],
            "point_mech": [
                {"name": "Ta", "x": 0, "y": 0, "z": 0},
                {"name": "lambda", "x": 0, "y": 0, "z": 0},
            ],
        },
    }


class Geometry(NamedTuple):
    mesh: dolfinx.mesh.Mesh
    facet_tags: dolfinx.mesh.MeshTags
    markers: dict[str, tuple[int, int]]
    f0: dolfinx.fem.Function | dolfinx.fem.Constant
    s0: dolfinx.fem.Function | dolfinx.fem.Constant
    n0: dolfinx.fem.Function | dolfinx.fem.Constant
    stim_tags: dolfinx.mesh.MeshTags
    stim_marker: int

    @property
    def dx(self):
        return ufl.Measure(
            "dx",
            domain=self.mesh,
            subdomain_data=self.stim_tags,
            metadata={"quadrature_degree": QUAD_DEGREE},
        )

    @property
    def ds(self):
        return ufl.Measure(
            "ds",
            domain=self.mesh,
            subdomain_data=self.facet_tags,
            metadata={"quadrature_degree": QUAD_DEGREE},
        )

    @property
    def facet_normal(self) -> ufl.FacetNormal:
        return ufl.FacetNormal(self.mesh)

    def surface_area(self, marker: str) -> float:
        marker_id = self.markers[marker][0]
        return self.mesh.comm.allreduce(
            dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.as_ufl(1.0) * self.ds(marker_id))),
            op=MPI.SUM,
        )


def disable_logger():
    for lib in ["numba", "matplotlib"]:
        logging.getLogger(lib).setLevel(logging.WARNING)


def create_stim_tags(mesh, stim_marker=1, stimx=1.5, stimy=1.5, stimz=1.5):
    tol = 1e-6

    def S1_subdomain(x):
        return np.logical_and(
            np.logical_and(x[0] <= stimx + tol, x[1] <= stimy + tol),
            x[2] <= stimz + tol,
        )

    cells = dolfinx.mesh.locate_entities(mesh, mesh.topology.dim, S1_subdomain)

    stim_tags = dolfinx.mesh.meshtags(
        mesh,
        mesh.topology.dim,
        cells,
        np.full(len(cells), stim_marker, dtype=np.int32),
    )
    stim_tags.name = "stimulus"
    return stim_tags


def accumulate_time(fn: Callable, totals: dict[str, float], key: str) -> Callable:
    """Wrap ``fn`` so that the wall time spent in it is added to ``totals[key]``."""

    @functools.wraps(fn)
    def timed(*args, **kwargs):
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            totals[key] += time.perf_counter() - start

    return timed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--odefile",
        type=Path,
        default=DEFAULT_ODEFILE,
        help="gotranx .ode file with a 'mechanics' component (default: %(default)s). "
        "Output goes to output/<odefile stem>/.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    start_total = time.perf_counter()
    args = parse_args(argv)

    logging.basicConfig(level=logging.DEBUG)
    disable_logger()
    dolfinx.log.set_log_level(dolfinx.log.LogLevel.DEBUG)

    comm = MPI.COMM_WORLD
    config = default_config()
    odefile = args.odefile
    config["sim"]["modelfile"] = str(odefile)
    config["sim"]["outdir"] = str(Path("output") / odefile.stem)

    # ---------------------------------------------------------
    # 1. Pre-processing: Generate ODE Code
    # ---------------------------------------------------------
    out_dir = Path("generated_odes") / odefile.stem

    logger.info(f"Generating ODE modules from {odefile}")
    modules = load_ode_modules(odefile, out_dir)
    ep_module = modules.ep

    # ---------------------------------------------------------
    # 2. Setup Meshes & Geometries
    # ---------------------------------------------------------
    # One directory per resolution: the mesh is only generated when its directory is
    # missing, so a single directory would silently reuse a slab of another resolution.
    geodir = Path("meshes") / f"slab_dx{SLAB_DX}"
    if not geodir.is_dir():
        cardiac_geometries.mesh.slab(
            outdir=geodir,
            lx=2.0,
            ly=1.0,
            lz=0.5,
            dx=SLAB_DX,
            create_fibers=True,
            fiber_angle_endo=0,
            fiber_angle_epi=0,
            fiber_space="DG_1",
            comm=comm,
            use_dolfinx=True,
        )

    geo = cardiac_geometries.geometry.Geometry.from_file(comm=comm, path=geodir / "geometry.bp")

    stim_marker = 1
    stim_tags = create_stim_tags(geo.mesh, stim_marker=stim_marker)
    geo.cfun = stim_tags
    mech_geo = geo
    geo.quadrature_degree = QUAD_DEGREE
    ep_geo = geo
    mesh = mech_geo.mesh
    ep_mesh = ep_geo.mesh

    # P1: values going back to EP are averaged onto the EP ODE space, which the
    # transfer plan supports for P1 and DG0 only.
    ep_ode_space = dolfinx.fem.functionspace(ep_mesh, ("P", 1))

    # ---------------------------------------------------------
    # 3. Setup EP Solver (fenicsx-beat)
    # ---------------------------------------------------------
    mesh_unit = "mm"
    chi = config["ep"]["chi"] * beat.units.ureg("mm**-1")
    C_m = config["ep"]["C_m"] * beat.units.ureg("uF/mm**2")

    M = beat.conductivities.define_conductivity_tensor(
        chi=chi,
        f0=ep_geo.f0,
        g_il=config["ep"]["conductivities"]["sigma_il"] * beat.units.ureg("S/m"),
        g_it=config["ep"]["conductivities"]["sigma_it"] * beat.units.ureg("S/m"),
        g_el=config["ep"]["conductivities"]["sigma_el"] * beat.units.ureg("S/m"),
        g_et=config["ep"]["conductivities"]["sigma_et"] * beat.units.ureg("S/m"),
    )

    time_ep = dolfinx.fem.Constant(ep_mesh, 0.0)

    I_s = beat.stimulation.define_stimulus(
        mesh=ep_mesh,
        chi=chi,
        time=time_ep,
        subdomain_data=stim_tags,
        marker=stim_marker,
        mesh_unit=mesh_unit,
        amplitude=50_000.0 * beat.units.ureg("uA/cm**3"),
    )

    pde = beat.MonodomainModel(
        time=time_ep,
        mesh=ep_mesh,
        M=M,
        I_s=I_s,
        C_m=C_m.to(f"uF/{mesh_unit}**2").magnitude,
        dx=ep_geo.dx,
    )

    v_ode = dolfinx.fem.Function(ep_ode_space)
    num_points_ep = v_ode.x.array.size

    # One column per point, for states, parameters and missing variables alike: the
    # transfer plan writes lambda and the values EP needs back into these arrays in
    # place, and they differ between points.
    y_ep = np.tile(ep_module.init_state_values()[:, None], (1, num_points_ep))
    p_ep = np.tile(
        ep_module.init_parameter_values(i_Stim_Amplitude=0.0)[:, None],
        (1, num_points_ep),
    )
    ep_missing = getattr(ep_module, "missing", {})

    ode = beat.odesolver.DolfinODESolver(
        v_ode=v_ode,
        v_pde=pde.state,
        fun=ep_module.generalized_rush_larsen,
        init_states=y_ep,
        parameters=p_ep,
        num_states=len(y_ep),
        v_index=ep_module.state_index("v"),
        missing_variables=np.zeros((len(ep_missing), num_points_ep)) if ep_missing else None,
        num_missing_variables=len(ep_missing),
    )

    ep_solver = beat.MonodomainSplittingSolver(pde=pde, ode=ode, theta=1)

    # ---------------------------------------------------------
    # 4. Setup Mechanics Solver (fenicsx-pulse)
    # ---------------------------------------------------------
    material_params = pulse.HolzapfelOgden.transversely_isotropic_parameters()
    material = pulse.HolzapfelOgden(f0=mech_geo.f0, s0=mech_geo.s0, **material_params)
    comp_model = pulse.compressibility.Incompressible()

    # The contraction model, stepped inside Newton. Its states live on a quadrature
    # space of the same degree as the mechanics form (the controller checks this).
    backend = GeneratedActivation(
        modules.mechanics,
        mesh,
        mech_geo.f0,
        quadrature_degree=QUAD_DEGREE,
    )

    model = pulse.CardiacModel(
        material=material,
        active=backend,
        compressibility=comp_model,
    )

    def dirichlet_bc(V: dolfinx.fem.FunctionSpace) -> list[dolfinx.fem.bcs.DirichletBC]:
        V0, _ = V.sub(0).collapse()
        zero = dolfinx.fem.Function(V0)
        zero.x.array[:] = 0.0

        x0_dofs = dolfinx.fem.locate_dofs_topological(
            (V.sub(0), V0),
            mech_geo.ffun.dim,
            mech_geo.ffun.find(mech_geo.markers["X0"][0]),
        )
        y0_dofs = dolfinx.fem.locate_dofs_topological(
            (V.sub(1), V0),
            mech_geo.ffun.dim,
            mech_geo.ffun.find(mech_geo.markers["Y0"][0]),
        )
        z0_dofs = dolfinx.fem.locate_dofs_topological(
            (V.sub(2), V0),
            mech_geo.ffun.dim,
            mech_geo.ffun.find(mech_geo.markers["Z0"][0]),
        )

        return [
            dolfinx.fem.dirichletbc(zero, x0_dofs, V.sub(0)),
            dolfinx.fem.dirichletbc(zero, y0_dofs, V.sub(1)),
            dolfinx.fem.dirichletbc(zero, z0_dofs, V.sub(2)),
        ]

    bcs = pulse.BoundaryConditions(dirichlet=(dirichlet_bc,))

    geometry = pulse.Geometry.from_cardiac_geometries(
        mech_geo,
        metadata={"quadrature_degree": QUAD_DEGREE},
    )
    petsc_options = pulse.StaticProblem.default_parameters()["petsc_options"]
    # The first residual at rest (~2.5e-9) stalls at round-off below a relative tolerance.
    petsc_options["snes_atol"] = 1e-9

    # Important: BaseBC must be free since we manually constrain X, Y, Z boundaries
    problem = pulse.StaticProblem(
        model=model,
        geometry=geometry,
        bcs=bcs,
        parameters={"base_bc": pulse.problem.BaseBC.free, "petsc_options": petsc_options},
    )

    # ---------------------------------------------------------
    # 5. Initialize Coupling Controller & DataCollector
    # ---------------------------------------------------------
    dt_ep = config["sim"]["dt"]
    N_steps = config["sim"]["N"]
    dt_mech = dt_ep * N_steps

    controller = SimulationController(
        mechanics_problem=problem,
        ep_solver=ep_solver,
        backend=backend,
        ode_modules=modules,
        dt_mech=dt_mech,
        dt_ep=dt_ep,
    )

    # Timing baseline: wall time spent in the EP ODE step, the EP PDE step and the
    # mechanics solve, accumulated over the run.
    timings = {"ep_ode_s": 0.0, "ep_pde_s": 0.0, "mech_s": 0.0}
    ep_solver.ode.step = accumulate_time(ep_solver.ode.step, timings, "ep_ode_s")
    ep_solver.pde.step = accumulate_time(ep_solver.pde.step, timings, "ep_pde_s")
    problem.solve = accumulate_time(problem.solve, timings, "mech_s")

    # Ta is backend.active_tension, in kPa (ZetaSplitUFL's Ta_current was in Pa).
    # The backend's outputs live on a quadrature space, which cannot be evaluated at
    # a point, so each is recorded through a P1 copy refreshed after every step.
    # "lmbda" keeps its old output name, "lambda".
    P1 = dolfinx.fem.functionspace(mesh, ("P", 1))
    mech_variables = {"Ta": backend.active_tension}
    averagers = []
    for name, output in backend.outputs.items():
        out_name = "lambda" if name == "lmbda" else name
        mech_variables[out_name] = dolfinx.fem.Function(P1, name=out_name)
        averagers.append(make_averager(output, mech_variables[out_name]))

    collector = DataCollector(
        problem=problem,
        ep_ode_space=ep_ode_space,
        config=config,
        mech_variables=mech_variables,
    )

    # Mesh mean of the fibre stretch at the quadrature points, per mechanics step.
    volume = comm.allreduce(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.as_ufl(1.0) * geometry.dx)),
        op=MPI.SUM,
    )
    lmbda_integral = dolfinx.fem.form(backend.lmbda_prev * geometry.dx)
    lmbda_mean: list[tuple[float, float]] = []

    # ---------------------------------------------------------
    # 6. Define Callbacks & Run Simulation
    # ---------------------------------------------------------
    inds = []  # To track mechanics steps for the collector finalize

    # The controller counts steps from 1; DataCollector indexes its arrays from 0.
    def on_ep_step(current_t, ep_step_idx):
        # Update EP functions for saving
        for out_ep_var in collector.out_ep_names:
            state_idx = ep_module.state_index(out_ep_var)
            collector.out_ep_funcs[out_ep_var].x.array[:] = ode.values[state_idx]

        i = ep_step_idx - 1
        if i % config["sim"]["save_frequency_ep"] == 0:
            collector.write_ep(current_t)
            collector.write_node_data_ep(i)

    def on_mech_step(current_t, mech_step_idx, newton_iters):
        i = mech_step_idx - 1
        inds.append(i * N_steps)
        collector.timers.no_of_newton_iterations.append(newton_iters)

        integral = dolfinx.fem.assemble_scalar(lmbda_integral)
        lmbda_mean.append((current_t, comm.allreduce(integral, op=MPI.SUM) / volume))
        for average in averagers:
            average()

        if i % config["sim"]["save_frequency_mech"] == 0:
            collector.write_node_data_mech(i)
            collector.write_disp(current_t)

    # Calculate total mechanics steps needed
    total_duration = config["sim"]["sim_dur"]
    total_mech_steps = int(np.ceil(total_duration / dt_mech))

    # --- THE MAIN LOOP ---
    start_loop = time.perf_counter()
    for _ in range(total_mech_steps):
        collector.timers.start_single_loop()

        # The controller does all the interpolation, sub-stepping, and solving!
        controller.step(ep_callback=on_ep_step, mech_callback=on_mech_step)

        collector.timers.stop_single_loop()
    timings["loop_s"] = time.perf_counter() - start_loop

    collector.finalize(inds)
    np.savetxt(
        collector.outdir / "lmbda_prev_mean.txt",
        np.array(lmbda_mean),
        header="t (ms), mesh mean of backend.lmbda_prev",
    )

    timings["total_s"] = time.perf_counter() - start_total
    if comm.rank == 0:
        (collector.outdir / "timings.json").write_text(json.dumps(timings, indent=4))
    logger.info(f"Timings: {timings}")


if __name__ == "__main__":
    main()
