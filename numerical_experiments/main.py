from typing import NamedTuple
import logging
from pathlib import Path

import ufl
import numpy as np
from mpi4py import MPI
import dolfinx
import fenicsx_pulse
import beat
import gotranx
import numba

from simcardemsx.mechanicsproblem import MechanicsProblem
from simcardemsx.land import LandModel
from simcardemsx.interpolation import MissingValue
from simcardemsx.datacollector import DataCollector


logger = logging.getLogger(__name__)


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
            "N": 2,
            "dt": 0.05,
            "mech_mesh": "meshes/mesh_mech_0.5dx_0.5Lx_1.0Ly_2.0Lz",
            "markerfile": "meshes/mesh_mech_0.5dx_0.5Lx_1.0Ly_2.0Lz_surface_ffun",
            "modelfile": "../odefiles/ToRORd_dynCl_endo_caisplit.ode",
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
        return ufl.Measure("dx", domain=self.mesh, subdomain_data=self.stim_tags)

    @property
    def ds(self):
        return ufl.Measure("ds", domain=self.mesh, subdomain_data=self.facet_tags)

    @property
    def facet_normal(self) -> ufl.FacetNormal:
        return ufl.FacetNormal(self.mesh)

    def surface_area(self, marker: str) -> float:
        marker_id = self.markers[marker][0]
        return self.mesh.comm.allreduce(
            dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.as_ufl(1.0) * self.ds(marker_id))),
            op=MPI.SUM,
        )


def create_mesh(comm, Lx=0.5, Ly=1.0, Lz=2.0, nx=2, ny=4, nz=8, stimx=0.5, stimy=0.5, stimz=0.5):
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


def main():
    logging.basicConfig(level=logging.DEBUG)
    disable_logger()

    comm = MPI.COMM_WORLD
    mech_geo = create_mesh(comm)
    # ep_geo = refine(refine(refine(mech_geo)))
    ep_geo = refine(refine(mech_geo))

    mesh = mech_geo.mesh
    ep_mesh = ep_geo.mesh

    config = default_config()

    # FIXME: Make this work for different meshes later

    # breakpoint()

    odefile = Path("ToRORd_dynCl_endo_zetasplit.ode")
    ep_model = setup_ep_ode_model(odefile)
    # fgr_ep = numba.jit(nopython=True)(ep_model["forward_generalized_rush_larsen"])
    fgr_ep = ep_model["forward_generalized_rush_larsen"]

    mv_ep = ep_model["missing_values"]

    # Get initial values from the EP model
    y_ep_ = ep_model["init_state_values"]()
    p_ep_ = ep_model["init_parameter_values"](i_Stim_Amplitude=0.0)

    ep_missing_values_ = np.zeros(len(ep_model["missing"]))
    mechanics_missing_values_ = np.zeros(2)

    ep_ode_space = dolfinx.fem.functionspace(ep_mesh, ("DG", 1))
    v_ode = dolfinx.fem.Function(ep_ode_space)
    num_points_ep = v_ode.x.array.size

    y_ep = np.zeros((len(y_ep_), num_points_ep))
    y_ep.T[:] = y_ep_  # Set to y_ep with initial values defined in ep_model

    # Set the activation
    activation_space = dolfinx.fem.functionspace(mesh, ("DG", 1))
    activation = dolfinx.fem.Function(activation_space)

    missing_mech = MissingValue(
        element=activation.ufl_element(),
        interpolation_element=ep_ode_space.ufl_element(),
        mechanics_mesh=mesh,
        ep_mesh=ep_mesh,
        num_values=len(mechanics_missing_values_),
    )

    missing_ep = MissingValue(
        element=ep_ode_space.ufl_element(),
        interpolation_element=activation.ufl_element(),
        mechanics_mesh=mesh,
        ep_mesh=ep_mesh,
        num_values=len(ep_missing_values_),
    )

    missing_ep.values_mechanics.T[:] = ep_missing_values_
    missing_ep.values_ep.T[:] = ep_missing_values_
    ode_missing_variables = missing_ep.values_ep
    missing_ep_args = (missing_ep.values_ep,)

    missing_mech.values_ep.T[:] = mechanics_missing_values_
    missing_mech.values_mechanics.T[:] = mechanics_missing_values_
    missing_mech.mechanics_values_to_function()  # Assign initial values to mech functions

    # Use previous cai in mech to be consistent across splitting schemes
    prev_missing_mech = MissingValue(
        element=activation.ufl_element(),
        interpolation_element=ep_ode_space.ufl_element(),
        mechanics_mesh=mesh,
        ep_mesh=ep_mesh,
        num_values=len(mechanics_missing_values_),
    )

    for i in range(len(mechanics_missing_values_)):
        prev_missing_mech.u_mechanics[i].x.array[:] = missing_mech.values_mechanics[i]

    p_ep = np.zeros((len(p_ep_), num_points_ep))
    p_ep.T[:] = p_ep_  # Initialise p_ep with initial values defined in ep_model

    mesh_unit = "cm"
    chi = 1400.0 * beat.units.ureg("cm**-1")
    s_l = 0.24 * beat.units.ureg("S/cm")
    s_t = 0.0456 * beat.units.ureg("S/cm")
    s_l = (s_l / chi).to("uA/mV").magnitude
    s_t = (s_t / chi).to("uA/mV").magnitude
    M = s_l * ufl.outer(ep_geo.f0, ep_geo.f0) + s_t * (
        ufl.Identity(3) - ufl.outer(ep_geo.f0, ep_geo.f0)
    )

    C_m = 1.0 * beat.units.ureg("uF/cm**2")

    time = dolfinx.fem.Constant(ep_mesh, 0.0)

    I_s = beat.stimulation.define_stimulus(
        mesh=ep_mesh,
        chi=chi,
        time=time,
        subdomain_data=ep_geo.stim_tags,
        marker=ep_geo.stim_marker,
        mesh_unit=mesh_unit,
        amplitude=50_000.0,
    )

    pde = beat.MonodomainModel(
        time=time,
        mesh=ep_mesh,
        M=M,
        I_s=I_s,
        C_m=C_m.to(f"uF/{mesh_unit}**2").magnitude,
        dx=ep_geo.dx,
    )

    ode = beat.odesolver.DolfinODESolver(
        v_ode=dolfinx.fem.Function(ep_ode_space),
        v_pde=pde.state,
        fun=fgr_ep,
        init_states=y_ep,
        parameters=p_ep,
        num_states=len(y_ep),
        v_index=ep_model["state_index"]("v"),
        missing_variables=ode_missing_variables,
        num_missing_variables=len(ep_missing_values_),
    )

    ep_solver = beat.MonodomainSplittingSolver(pde=pde, ode=ode, theta=1)

    # material_params = fenicsx_pulse.HolzapfelOgden.orthotropic_parameters()
    material_params = fenicsx_pulse.HolzapfelOgden.transversely_isotropic_parameters()
    material = fenicsx_pulse.HolzapfelOgden(f0=mech_geo.f0, s0=mech_geo.s0, **material_params)
    comp_model = fenicsx_pulse.compressibility.Incompressible()
    active_model = LandModel(
        f0=mech_geo.f0,
        s0=mech_geo.s0,
        n0=mech_geo.n0,
        XS=missing_mech.u_mechanics[0],
        XW=missing_mech.u_mechanics[1],
        mesh=mesh,
        dLambda_tol=1e-12,
        eta=0.0,
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
            mech_geo.facet_tags.find(mech_geo.markers["X0"][1]),
        )
        y0_dofs = dolfinx.fem.locate_dofs_topological(
            (V.sub(1), V0),
            mech_geo.facet_tags.dim,
            mech_geo.facet_tags.find(mech_geo.markers["Y0"][1]),
        )
        z0_dofs = dolfinx.fem.locate_dofs_topological(
            (V.sub(2), V0),
            mech_geo.facet_tags.dim,
            mech_geo.facet_tags.find(mech_geo.markers["Z0"][1]),
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
        "Zetas": active_model._Zetas,
        "Zetaw": active_model._Zetaw,
        "lambda": active_model.lmbda,
        "XS": active_model.XS,
        "XW": active_model.XW,
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
                ep_model["state_index"](out_ep_var)
            ]

        collector.write_node_data_ep(i)

        if i % config["sim"]["N"] != 0:
            collector.timers.stop_single_loop()
            continue

        collector.timers.start_var_transfer()
        # Extract missing values for the mechanics step from the ep model (ep function space)
        missing_ep_values = mv_ep(
            ti + config["sim"]["dt"],
            ode._values,
            ode.parameters,
            *missing_ep_args,
        )
        # Assign the extracted values as missing_mech for the mech step (ep function space)
        for k in range(missing_mech.num_values):
            missing_mech.u_ep_int[k].x.array[:] = missing_ep_values[k, :]

        # Interpolate missing variables from ep to mech function space
        missing_mech.interpolate_ep_to_mechanics()
        missing_mech.mechanics_function_to_values()
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
        if missing_ep is not None:
            missing_ep.interpolate_mechanics_to_ep()
            missing_ep.ep_function_to_values()
        collector.timers.stop_var_transfer()

        collector.write_node_data_mech(i)

        collector.timers.start_var_transfer
        # Use previous cai in mech to be consistent with zeta split
        for i in range(len(mechanics_missing_values_)):
            prev_missing_mech.u_mechanics[i].x.array[:] = missing_mech.values_mechanics[i]
        collector.timers.stop_var_transfer()
        collector.timers.collect_var_transfer()

        collector.write_disp(j)

        j += 1
        collector.timers.stop_single_loop()

    collector.finalize(inds)


if __name__ == "__main__":
    main()
