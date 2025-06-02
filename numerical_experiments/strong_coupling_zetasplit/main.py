from dataclasses import dataclass
from typing import NamedTuple
import logging
from pathlib import Path

import simcardemsx.ode_model
import ufl
import numpy as np
from mpi4py import MPI
import dolfinx
import fenicsx_pulse
import beat
import gotranx
import numba

import simcardemsx
from simcardemsx.mechanicsproblem import MechanicsProblem
# from simcardemsx.land import LandModel

from simcardemsx.datacollector import DataCollector


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
            "outdir": "100ms_N1_cai_split_runcheck",
            "sim_dur": 40,
            "split_scheme": "cai",
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


def create_mesh(comm, Lx=2.0, Ly=1.0, Lz=0.5, nx=4, ny=2, nz=1, stimx=1.5, stimy=1.5, stimz=1.5):
    logger.debug("Creating mesh")
    mesh = dolfinx.mesh.create_box(
        comm,
        [[0.0, 0.0, 0.0], [Lx, Ly, Lz]],
        [nx, ny, nz],
        dolfinx.mesh.CellType.tetrahedron,
        ghost_mode=dolfinx.mesh.GhostMode.none,
    )
    fdim = mesh.topology.dim - 1
    x0_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[0], 0))
    x1_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[0], Lx))
    y0_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[1], 0))
    y1_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[1], Ly))
    z0_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[2], 0))
    z1_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[2], Lz))

    # Concatenate and sort the arrays based on facet indices.
    # Left facets marked with 1, right facets with two
    marked_facets = np.hstack([x0_facets, x1_facets, y0_facets, y1_facets, z0_facets, z1_facets])

    marked_values = np.hstack(
        [
            np.full_like(x0_facets, 1),
            np.full_like(x1_facets, 2),
            np.full_like(y0_facets, 3),
            np.full_like(y1_facets, 4),
            np.full_like(z0_facets, 5),
            np.full_like(z1_facets, 6),
        ],
    )
    sorted_facets = np.argsort(marked_facets)
    ft = dolfinx.mesh.meshtags(
        mesh,
        fdim,
        marked_facets[sorted_facets],
        marked_values[sorted_facets],
    )
    ft.name = "facet_tags"
    markers = {
        "X0": (2, 1),
        "X1": (2, 2),
        "Y0": (2, 3),
        "Y1": (2, 4),
        "Z0": (2, 5),
        "Z1": (2, 6),
    }

    f0 = dolfinx.fem.Constant(mesh, (1.0, 0.0, 0.0))
    s0 = dolfinx.fem.Constant(mesh, (0.0, 1.0, 0.0))
    n0 = dolfinx.fem.Constant(mesh, (0.0, 0.0, 1.0))

    tol = 1e-6

    def S1_subdomain(x):
        return np.logical_and(
            np.logical_and(x[0] <= stimx + tol, x[1] <= stimy + tol),
            x[2] <= stimz + tol,
        )

    cells = dolfinx.mesh.locate_entities(mesh, mesh.topology.dim, S1_subdomain)
    stim_marker = 1
    stim_tags = dolfinx.mesh.meshtags(
        mesh,
        mesh.topology.dim,
        cells,
        np.full(len(cells), stim_marker, dtype=np.int32),
    )
    stim_tags.name = "stimulus"

    with dolfinx.io.XDMFFile(mesh.comm, "tags.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(ft, mesh.geometry)
        xdmf.write_meshtags(stim_tags, mesh.geometry)

    return Geometry(
        mesh=mesh,
        facet_tags=ft,
        markers=markers,
        f0=f0,
        s0=s0,
        n0=n0,
        stim_tags=stim_tags,
        stim_marker=stim_marker,
    )


def refine(geo: Geometry) -> Geometry:
    mesh = geo.mesh
    mesh.topology.create_entities(1)
    mesh.topology.create_connectivity(2, 3)

    new_mesh, parent_cell, parent_facet = dolfinx.mesh.refine(
        mesh, partitioner=None, option=dolfinx.mesh.RefinementOption.parent_cell_and_facet
    )
    new_mesh.topology.create_entities(1)
    new_mesh.topology.create_connectivity(2, 3)
    new_stim_tags = dolfinx.mesh.transfer_meshtag(
        geo.stim_tags, new_mesh, parent_cell, parent_facet
    )
    new_facet_tags = dolfinx.mesh.transfer_meshtag(
        geo.facet_tags, new_mesh, parent_cell, parent_facet
    )

    f0 = dolfinx.fem.Constant(new_mesh, (1.0, 0.0, 0.0))
    s0 = dolfinx.fem.Constant(new_mesh, (0.0, 1.0, 0.0))
    n0 = dolfinx.fem.Constant(new_mesh, (0.0, 0.0, 1.0))

    # Create a new Geometry object with the refined mesh
    return Geometry(
        mesh=new_mesh,
        facet_tags=new_facet_tags,
        markers=geo.markers,
        f0=f0,
        s0=s0,
        n0=n0,
        stim_tags=new_stim_tags,
        stim_marker=geo.stim_marker,
    )


def setup_ep_ode_model(odefile):
    module_file = Path("ep_model.py")
    if not module_file.is_file():
        ode = gotranx.load_ode(odefile)

        mechanics_comp = ode.get_component("mechanics")
        mechanics_ode = mechanics_comp.to_ode()

        ep_ode = ode - mechanics_comp

        # Generate code for the electrophysiology model
        code_ep = gotranx.cli.gotran2py.get_code(
            ep_ode,
            scheme=[gotranx.schemes.Scheme.forward_generalized_rush_larsen],
            missing_values=mechanics_ode.missing_variables,
        )

        Path(module_file).write_text(code_ep)
        # Currently 3D mech needs to be written manually

    return __import__(str(module_file.stem)).__dict__


def disable_logger():
    for lib in ["numba", "matplotlib"]:
        logging.getLogger(lib).setLevel(logging.WARNING)


import cardiac_geometries


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

    # with dolfinx.io.XDMFFile(mesh.comm, "tags.xdmf", "w") as xdmf:
    #     xdmf.write_mesh(mesh)
    #     xdmf.write_meshtags(stim_tags, mesh.geometry)
    return stim_tags


def main():
    logging.basicConfig(level=logging.DEBUG)
    disable_logger()
    dolfinx.log.set_log_level(dolfinx.log.LogLevel.DEBUG)

    comm = MPI.COMM_WORLD
    # mech_geo = create_mesh(comm)
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

    geo = cardiac_geometries.geometry.Geometry.from_folder(
        comm=MPI.COMM_WORLD,
        folder=geodir,
    )

    stim_marker = 1
    stim_tags = create_stim_tags(geo.mesh, stim_marker=stim_marker)

    mech_geo = Geometry(
        mesh=geo.mesh,
        facet_tags=geo.ffun,
        markers=geo.markers,
        f0=geo.f0,
        s0=geo.s0,
        n0=geo.n0,
        stim_tags=stim_tags,
        stim_marker=stim_marker,
    )

    # ep_geo = refine(refine(refine(mech_geo)))
    # ep_geo = refine(refine(mech_geo))
    ep_geo = mech_geo
    mesh = mech_geo.mesh
    ep_mesh = ep_geo.mesh

    ode_space = "DG_1"
    family = ode_space.split("_")[0]
    degree = int(ode_space.split("_")[1])

    # Set the activation
    mech_ode_space = dolfinx.fem.functionspace(mesh, (family, degree))

    ep_ode_space = dolfinx.fem.functionspace(ep_mesh, (family, degree))
    v_ode = dolfinx.fem.Function(ep_ode_space)

    config = default_config()

    # FIXME: Make this work for different meshes later

    odefile = Path(config["sim"]["modelfile"])
    ode_model = simcardemsx.ode_model.ODEModel(
        odefile=odefile, mech_ode_space=mech_ode_space, ep_ode_space=ep_ode_space
    )

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
        subdomain_data=ep_geo.stim_tags,
        marker=ep_geo.stim_marker,
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

    num_points_ep = v_ode.x.array.size

    y_ep_ = ode_model.y()
    p_ep_ = ode_model.p(i_Stim_Amplitude=0.0)
    y_ep = np.zeros((len(y_ep_), num_points_ep))
    y_ep.T[:] = y_ep_  # Set to y_ep with initial values defined in ep_model
    p_ep = np.zeros((len(p_ep_), num_points_ep))
    p_ep.T[:] = p_ep_  # Initialise p_ep with initial values defined in ep_model

    ode = beat.odesolver.DolfinODESolver(
        v_ode=dolfinx.fem.Function(ep_ode_space),
        v_pde=pde.state,
        fun=ode_model.fgr,
        init_states=y_ep,
        parameters=p_ep,
        num_states=len(y_ep),
        v_index=ode_model.module["state_index"]("v"),
        missing_variables=ode_model.missing_ep.values_ep,
        num_missing_variables=ode_model.missing_ep.num_values,
    )

    ep_solver = beat.MonodomainSplittingSolver(pde=pde, ode=ode, theta=1)

    # material_params = fenicsx_pulse.HolzapfelOgden.orthotropic_parameters()
    material_params = fenicsx_pulse.HolzapfelOgden.transversely_isotropic_parameters()

    material = fenicsx_pulse.HolzapfelOgden(f0=mech_geo.f0, s0=mech_geo.s0, **material_params)
    comp_model = fenicsx_pulse.compressibility.Incompressible()

    from mechanics_model import LandModel

    active_model = LandModel(
        function_space=mech_ode_space,
        missing_values=ode_model.missing_mech.u_mechanics,
    )

    model = fenicsx_pulse.CardiacModel(
        material=material,
        active=active_model,
        compressibility=comp_model,
    )

    def dirichlet_bc(
        V: dolfinx.fem.FunctionSpace,
    ) -> list[dolfinx.fem.bcs.DirichletBC]:
        V0, _ = V.sub(0).collapse()
        zero = dolfinx.fem.Function(V0)
        zero.x.array[:] = 0.0

        x0_dofs = dolfinx.fem.locate_dofs_topological(
            (V.sub(0), V0),
            mech_geo.facet_tags.dim,
            mech_geo.facet_tags.find(mech_geo.markers["X0"][0]),
        )
        y0_dofs = dolfinx.fem.locate_dofs_topological(
            (V.sub(1), V0),
            mech_geo.facet_tags.dim,
            mech_geo.facet_tags.find(mech_geo.markers["Y0"][0]),
        )
        z0_dofs = dolfinx.fem.locate_dofs_topological(
            (V.sub(2), V0),
            mech_geo.facet_tags.dim,
            mech_geo.facet_tags.find(mech_geo.markers["Z0"][0]),
        )

        return [
            dolfinx.fem.dirichletbc(zero, x0_dofs, V.sub(0)),
            dolfinx.fem.dirichletbc(zero, y0_dofs, V.sub(1)),
            dolfinx.fem.dirichletbc(zero, z0_dofs, V.sub(2)),
        ]

    bcs = fenicsx_pulse.BoundaryConditions(
        dirichlet=(dirichlet_bc,),
    )

    problem = MechanicsProblem(model=model, geometry=mech_geo, bcs=bcs)
    problem.solve()

    mech_variables = {
        "Ta": active_model.Ta_current,
        "Zetas": active_model.y[0],
        "Zetaw": active_model.y[1],
        "lambda": active_model.lmbda,
        "XS": active_model.missing_values[0],
        "XW": active_model.missing_values[1],
        "dLambda": active_model._dLambda,
    }

    inds = []  # Array with time-steps for which we solve mechanics
    j = 0

    collector = DataCollector(
        problem=problem,
        ep_ode_space=ep_ode_space,
        config=config,
        mech_variables=mech_variables,
    )
    # t = np.arange(0, config["sim"]["sim_dur"], config["sim"]["dt"])
    for i, ti in enumerate(collector.t):
        collector.timers.start_single_loop()

        print(f"Solving time {ti:.2f} ms")
        # t_bcs.assign(ti)  # Use ti+ dt here instead?

        collector.timers.start_ep()
        ep_solver.step((ti, ti + config["sim"]["dt"]))

        collector.timers.stop_ep()

        # Assign values to ep function
        for out_ep_var in collector.out_ep_names:
            collector.out_ep_funcs[out_ep_var].x.array[:] = ode._values[
                ode_model.module["state_index"](out_ep_var)
            ]

        collector.write_node_data_ep(i)

        if i % config["sim"]["N"] != 0:
            collector.timers.stop_single_loop()
            continue

        collector.timers.start_var_transfer()

        # Assign the extracted values as missing_mech for the mech step (ep function space)
        ode_model.update_ep_missing_values(
            ti + config["sim"]["dt"],
            ode._values,
            ode.parameters,
        )
        # Interpolate missing variables from ep to mech function space
        ode_model.missing_mech.interpolate_ep_to_mechanics()
        ode_model.missing_mech.mechanics_function_to_values()
        inds.append(i)

        collector.timers.stop_var_transfer()

        print("Solve mechanics")
        collector.timers.start_mech()

        active_model.t.value = ti + config["sim"]["N"] * config["sim"]["dt"]  # Addition!
        nit = problem.solve()  # ti, config["sim"]["N"] * config["sim"]["dt"])
        problem.post_solve()
        collector.timers.no_of_newton_iterations.append(nit)
        print(f"No of iterations: {nit}")
        active_model.update_prev()
        collector.timers.stop_mech()

        collector.timers.start_var_transfer()
        # Do we need to handle more cases here?
        # if config["sim"]["split_scheme"] == "cai":
        #     missing_ep.u_mechanics_int[0].interpolate(active_model._J_TRPN)
        if ode_model.missing_ep is not None:
            ode_model.missing_ep.interpolate_mechanics_to_ep()
            ode_model.missing_ep.ep_function_to_values()
        collector.timers.stop_var_transfer()

        collector.write_node_data_mech(i)

        collector.timers.start_var_transfer
        # Use previous cai in mech to be consistent with zeta split
        ode_model.update_prev_missing_mech()

        collector.timers.stop_var_transfer()
        collector.timers.collect_var_transfer()

        collector.write_disp(j)

        j += 1
        collector.timers.stop_single_loop()

    collector.finalize(inds)


if __name__ == "__main__":
    main()
