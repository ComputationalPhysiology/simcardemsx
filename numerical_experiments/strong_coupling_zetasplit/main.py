import importlib.util
import logging
from pathlib import Path
from typing import NamedTuple

from mpi4py import MPI

import beat
import dolfinx
import numpy as np
import pulse
import ufl

import cardiac_geometries
from simcardemsx.controller import SimulationController
from simcardemsx.datacollector import DataCollector
from simcardemsx.land import LandModel
from simcardemsx.mechanicsproblem import MechanicsProblem
from simcardemsx.ode_model import RuntimeODEModel, generate_ode_code

logger = logging.getLogger(__name__)
QUAD_DEGREE = 4  # Degree of quadrature for the mechanics mesh


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


def load_module_from_path(module_name: str, file_path: Path):
    """Cleanly loads a Python file as a module without sys.path hacks."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec is not None, f"Could not load module {module_name} from {file_path}"
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None, f"Could not load module {module_name} from {file_path}"
    spec.loader.exec_module(module)
    return module


def main():
    logging.basicConfig(level=logging.DEBUG)
    disable_logger()
    dolfinx.log.set_log_level(dolfinx.log.LogLevel.DEBUG)

    comm = MPI.COMM_WORLD
    config = default_config()

    # ---------------------------------------------------------
    # 1. Pre-processing: Generate ODE Code
    # ---------------------------------------------------------
    odefile = Path(config["sim"]["modelfile"])
    out_dir = Path("generated_odes")

    logger.info(f"Generating ODE modules from {odefile}")
    generate_ode_code(odefile, out_dir)
    ep_module = load_module_from_path("ep_model", out_dir / "ep_model.py")

    # ---------------------------------------------------------
    # 2. Setup Meshes & Geometries
    # ---------------------------------------------------------
    geodir = Path("meshes")
    if not geodir.is_dir():
        cardiac_geometries.mesh.slab(
            outdir=geodir,
            lx=2.0,
            ly=1.0,
            lz=0.5,
            dx=0.5,
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

    ode_space = "DG_1"
    family = ode_space.split("_")[0]
    degree = int(ode_space.split("_")[1])

    mech_ode_space = dolfinx.fem.functionspace(mesh, (family, degree))
    ep_ode_space = dolfinx.fem.functionspace(ep_mesh, (family, degree))

    # ---------------------------------------------------------
    # 3. Initialize Runtime ODE Model
    # ---------------------------------------------------------
    ode_model = RuntimeODEModel(
        ep_module_dict=ep_module.__dict__,
        mech_ode_space=mech_ode_space,
        ep_ode_space=ep_ode_space,
    )

    # ---------------------------------------------------------
    # 4. Setup EP Solver (fenicsx-beat)
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

    time = dolfinx.fem.Constant(ep_mesh, 0.0)

    I_s = beat.stimulation.define_stimulus(
        mesh=ep_mesh,
        chi=chi,
        time=time,
        subdomain_data=stim_tags,
        marker=stim_marker,
        mesh_unit=mesh_unit,
        amplitude=50_000.0 * beat.units.ureg("uA/cm**3"),
    )

    pde = beat.MonodomainModel(
        time=time,
        mesh=ep_mesh,
        M=M,
        I_s=I_s,
        C_m=C_m.to(f"uF/{mesh_unit}**2").magnitude,
        dx=ep_geo.dx,
    )

    v_ode = dolfinx.fem.Function(ep_ode_space)
    num_points_ep = (
        ep_ode_space.dofmap.index_map.size_local + ep_ode_space.dofmap.index_map.num_ghosts
    )

    y_ep_ = ode_model.y()
    p_ep_ = ode_model.p(i_Stim_Amplitude=0.0)

    y_ep = np.zeros((len(y_ep_), num_points_ep))
    y_ep.T[:] = y_ep_
    p_ep = np.zeros((len(p_ep_), num_points_ep))
    p_ep.T[:] = p_ep_

    ode = beat.odesolver.DolfinODESolver(
        v_ode=v_ode,
        v_pde=pde.state,
        fun=ode_model.fgr,
        init_states=y_ep,
        parameters=p_ep,
        num_states=len(y_ep),
        v_index=ode_model.ep_module_dict["state_index"]("v"),
        missing_variables=ode_model.missing_ep.values_ep,
        num_missing_variables=ode_model.missing_ep.num_values,
    )

    ep_solver = beat.MonodomainSplittingSolver(pde=pde, ode=ode, theta=1)

    # ---------------------------------------------------------
    # 5. Setup Mechanics Solver (fenicsx-pulse)
    # ---------------------------------------------------------
    material_params = pulse.HolzapfelOgden.transversely_isotropic_parameters()
    material = pulse.HolzapfelOgden(f0=mech_geo.f0, s0=mech_geo.s0, **material_params)
    comp_model = pulse.compressibility.Incompressible()

    # Pass the EP variables directly via missing_mech u_mechanics functions
    active_model = LandModel(
        f0=mech_geo.f0,
        s0=mech_geo.s0,
        n0=mech_geo.n0,
        XS=ode_model.missing_mech.u_mechanics[0],
        XW=ode_model.missing_mech.u_mechanics[1],
        mesh=mech_geo.mesh,
    )

    model = pulse.CardiacModel(
        material=material,
        active=active_model,
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

    # Important: BaseBC must be free since we manually constrain X, Y, Z boundaries
    problem = MechanicsProblem(
        model=model,
        geometry=mech_geo,
        bcs=bcs,
        parameters={"base_bc": pulse.problem.BaseBC.free},
    )
    problem.solve()

    # ---------------------------------------------------------
    # 6. Initialize Coupling Controller & DataCollector
    # ---------------------------------------------------------
    dt_ep = config["sim"]["dt"]
    N_steps = config["sim"]["N"]
    dt_mech = dt_ep * N_steps

    controller = SimulationController(
        mechanics_problem=problem,
        ep_solver=ep_solver,
        ode_model=ode_model,
        dt_mech=dt_mech,
        dt_ep=dt_ep,
    )

    mech_variables = {
        "Ta": active_model.Ta_current,
        "Zetas": active_model._Zetas,
        "Zetaw": active_model._Zetaw,
        "lambda": active_model.lmbda,
        "XS": ode_model.missing_mech.u_mechanics[0],
        "XW": ode_model.missing_mech.u_mechanics[1],
        "dLambda": active_model._dLambda,
    }

    collector = DataCollector(
        problem=problem,
        ep_ode_space=ep_ode_space,
        config=config,
        mech_variables=mech_variables,
    )

    # ---------------------------------------------------------
    # 7. Define Callbacks & Run Simulation
    # ---------------------------------------------------------
    inds = []  # To track mechanics steps for the collector finalize

    def on_ep_step(current_t, ep_step_idx):
        # Update EP functions for saving
        for out_ep_var in collector.out_ep_names:
            state_idx = ode_model.ep_module_dict["state_index"](out_ep_var)
            collector.out_ep_funcs[out_ep_var].x.array[:] = ode._values[state_idx]

        if ep_step_idx % config["sim"]["save_frequency_ep"] == 0:
            collector.write_ep(current_t)
            collector.write_node_data_ep(ep_step_idx)

    def on_mech_step(current_t, mech_step_idx, newton_iters):
        inds.append(mech_step_idx * N_steps)
        collector.timers.no_of_newton_iterations.append(newton_iters)

        if mech_step_idx % config["sim"]["save_frequency_mech"] == 0:
            collector.write_node_data_mech(mech_step_idx)
            collector.write_disp(current_t)

    # Calculate total mechanics steps needed
    total_duration = config["sim"]["sim_dur"]
    total_mech_steps = int(np.ceil(total_duration / dt_mech))

    # --- THE MAIN LOOP ---
    for _ in range(total_mech_steps):
        collector.timers.start_single_loop()

        # The controller does all the interpolation, sub-stepping, and solving!
        controller.step(ep_callback=on_ep_step, mech_callback=on_mech_step)

        collector.timers.stop_single_loop()

    collector.finalize(inds)


if __name__ == "__main__":
    main()
