from dataclasses import dataclass, field
import typing
from pathlib import Path
import json
import matplotlib.pyplot as plt

from mpi4py import MPI
import dolfinx
import ufl
import toml
import numpy as np
import adios4dolfinx

from .mechanicsproblem import MechanicsProblem


def compute_function_average_over_mesh(func, mesh):
    volume = mesh.comm.allreduce(
        dolfinx.fem.assemble_scalar(
            dolfinx.fem.form(dolfinx.fem.Constant(mesh, 1.0) * ufl.dx(domain=mesh))
        ),
        op=MPI.SUM,
    )

    return (
        mesh.comm.allreduce(
            dolfinx.fem.assemble_scalar(dolfinx.fem.form(func * ufl.dx(domain=mesh))), op=MPI.SUM
        )
        / volume
    )


class Timing(typing.NamedTuple):
    reps: int
    wall_tot: float
    usr_tot: float
    sys_tot: float

    @property
    def wall_avg(self):
        return self.wall_tot / self.reps

    @property
    def usr_avg(self):
        return self.usr_tot / self.reps

    @property
    def sys_avg(self):
        return self.sys_tot / self.reps


@dataclass
class Timers:
    timings_solveloop: list[float] = field(default_factory=list)
    timings_ep_steps: list[float] = field(default_factory=list)
    timings_mech_steps: list[float] = field(default_factory=list)
    no_of_newton_iterations: list[float] = field(default_factory=list)
    timings_var_transfer: list[float] = field(default_factory=list)

    def __post_init__(self):
        self.start_total()

    def start_total(self):
        self.total_timer = dolfinx.common.Timer("total")

    def stop_total(self):
        self.total_timer.stop()
        self.timings_total = self.total_timer.elapsed

    def start_ep(self):
        self.ep_timer = dolfinx.common.Timer("ep")

    def stop_ep(self):
        self.ep_timer.stop()
        self.timings_ep_steps.append(self.ep_timer.elapsed()[0])

    def start_single_loop(self):
        self.timing_single_loop = dolfinx.common.Timer("single_loop")

    def stop_single_loop(self):
        self.timing_single_loop.stop()
        self.timings_solveloop.append(self.timing_single_loop.elapsed()[0])

    def start_var_transfer(self):
        self.timing_var_transfer = dolfinx.common.Timer("mv and lambda transfer time")

    def stop_var_transfer(self):
        self.timing_var_transfer.stop()

    def collect_var_transfer(self):
        self.timings_var_transfer.append(self.timing_var_transfer.elapsed()[0])

    def start_mech(self):
        self.mech_timer = dolfinx.common.Timer("mech time")

    def stop_mech(self):
        self.mech_timer.stop()
        self.timings_mech_steps.append(self.mech_timer.elapsed()[0])

    def finalize(self, comm, outdir: Path):
        self.stop_total()

        # from io import StringIO
        # from wurlitzer import pipes

        # out = StringIO()
        # with pipes(stdout=out):
        #     dolfinx.common.list_timings(
        #         comm,
        #         [
        #             dolfinx.common.TimingType.wall,
        #             dolfinx.common.TimingType.user,
        #             dolfinx.common.TimingType.system,
        #         ],
        #     )
        # timings = out.getvalue()
        # print(timings)

        timings = {}
        for task_name in [
            "Build BoxMesh (tetrahedra)",
            "Build dofmap data",
            "Build sparsity",
            "Compute connectivity 0-0",
            "Compute connectivity 1-0",
            "Compute connectivity 2-0",
            "Compute connectivity 3-0",
            "Compute dof reordering map",
            "Compute entities of dim = 1",
            "Compute entities of dim = 2",
            "Compute local part of mesh dual graph (mixed)",
            "Compute local-to-local map",
            "Compute-local-to-global links for global/local adjacency list",
            "Distribute row-wise data (scalable)",
            "GPS: create_level_structure",
            "Gibbs-Poole-Stockmeyer ordering",
            "Init dofmap from element dofmap",
            "PLAZA: Enforce rules",
            "PLAZA: refine",
            "SparsityPattern::finalize",
            "Topology: create",
            "Topology: determine shared index ownership",
            "Topology: determine vertex ownership groups (owned, undetermined, unowned)",
            "mech time",
            "mv and lambda transfer time",
            "single_loop",
            "total",
            "Loop total times",
            "Ep steps times",
            "Mech steps times",
            "No of mech iterations",
            "mv and lambda transfer time",
            "Total time",
        ]:
            try:
                timings[task_name] = Timing(
                    *dolfinx.common.timing(task_name),
                )
            except RuntimeError:
                continue

        (outdir / "solve_timings.json").write_text(
            json.dumps(
                {
                    "Loop total times": self.timings_solveloop,
                    "Ep steps times": self.timings_ep_steps,
                    "Mech steps times": self.timings_mech_steps,
                    "No of mech iterations": self.no_of_newton_iterations,
                    "mv and lambda transfer time": self.timings_var_transfer,
                    "Total time": self.total_timer.elapsed()[0],
                    "timings": timings,
                },
                indent=4,
            )
        )


@dataclass
class DataCollector:
    problem: MechanicsProblem
    ep_ode_space: dolfinx.fem.FunctionSpace
    config: dict
    mech_variables: dict[str, dolfinx.fem.Function]

    def __post_init__(self):
        self.outdir.mkdir(exist_ok=True, parents=True)

        self._t = np.arange(0, self.config["sim"]["sim_dur"], self.config["sim"]["dt"])
        (self.outdir / "config.txt").write_text(toml.dumps(self.config))

        self.out_ep_var_names = self.config["output"]["all_ep"]
        self.out_mech_var_names = self.config["output"]["all_mech"]

        self.out_ep_coord_names = [f["name"] for f in self.config["output"]["point_ep"]]
        self.ep_coords = [
            np.array([[f[f"{coord}"] for coord in ["x", "y", "z"]]], dtype=np.float64)
            for f in self.config["output"]["point_ep"]
        ]

        self.out_mech_coord_names = [f["name"] for f in self.config["output"]["point_mech"]]
        self.mech_coords = [
            np.array([[f[f"{coord}"] for coord in ["x", "y", "z"]]], dtype=np.float64)
            for f in self.config["output"]["point_mech"]
        ]

        # Create function spaces for ep variables to output
        self.out_ep_funcs = {}
        for out_ep_var in self.out_ep_names:
            self.out_ep_funcs[out_ep_var] = dolfinx.fem.Function(self.ep_ode_space, name=out_ep_var)

        import shutil

        shutil.rmtree(self.outdir / "disp.bp", ignore_errors=True)
        shutil.rmtree(self.outdir / "ep.bp", ignore_errors=True)
        self.vtx_disp = dolfinx.io.VTXWriter(
            self.comm, self.outdir / "disp.bp", [self.problem.u], engine="BP5"
        )
        self.vtx_ep = dolfinx.io.VTXWriter(
            self.comm,
            self.outdir / "ep.bp",
            [self.out_ep_funcs[out_ep_var] for out_ep_var in self.out_ep_var_names],
            engine="BP5",
        )

        self.out_ep_files = {}
        for out_ep_var in self.out_ep_var_names:
            self.out_ep_files[out_ep_var] = self.outdir / f"{out_ep_var}_out_ep.xdmf"
            self.out_ep_files[out_ep_var].unlink(missing_ok=True)
            self.out_ep_files[out_ep_var].with_suffix(".h5").unlink(missing_ok=True)

        self.out_mech_files = {}
        for out_mech_var in self.out_mech_var_names:
            self.out_mech_files[out_mech_var] = self.outdir / f"{out_mech_var}_out_mech.xdmf"
            self.out_mech_files[out_mech_var].unlink(missing_ok=True)
            self.out_mech_files[out_mech_var].with_suffix(".h5").unlink(missing_ok=True)

        self.out_ep_example_nodes = {}
        self.out_ep_volume_average_timeseries = {}
        for out_ep_var in self.out_ep_coord_names:
            self.out_ep_example_nodes[out_ep_var] = np.zeros(len(self.t))
            self.out_ep_volume_average_timeseries[out_ep_var] = np.zeros(len(self.t))

        self.out_mech_example_nodes = {}
        self.out_mech_volume_average_timeseries = {}
        for out_mech_var in self.out_mech_coord_names:
            self.out_mech_example_nodes[out_mech_var] = np.zeros(len(self.t))
            self.out_mech_volume_average_timeseries[out_mech_var] = np.zeros(len(self.t))

        self.timers = Timers()
        # adios4dolfinx.write_mesh(self.disp_file, self.problem.geometry.mesh)

        self._setup_eval_mech()
        self._setup_eval_ep()

    def _setup_eval_mech(self):
        bb_tree_mech = dolfinx.geometry.bb_tree(self.mech_mesh, self.mech_mesh.topology.dim)
        potential_colliding_cells_mech = [
            dolfinx.geometry.compute_collisions_points(bb_tree_mech, coord)
            for coord in self.mech_coords
        ]
        adj_mech = [
            dolfinx.geometry.compute_colliding_cells(self.mech_mesh, cells, coords)
            for cells, coords in zip(potential_colliding_cells_mech, self.mech_coords)
        ]
        self.indices_mech = [
            np.flatnonzero(adj.offsets[1:] - adj.offsets[:-1]) for adj in adj_mech
        ]  # Get indices of cells that are colliding with the point
        self.cells_mech = [
            adj.array[adj.offsets[indices]] for adj, indices in zip(adj_mech, self.indices_mech)
        ]  # Get the cells that are colliding with the point

        self.points_on_proc_mech = [
            coords[indices] for coords, indices in zip(self.mech_coords, self.indices_mech)
        ]

    def _setup_eval_ep(self):
        bb_tree_ep = dolfinx.geometry.bb_tree(self.ep_mesh, self.ep_mesh.topology.dim)
        potential_colliding_cells_ep = [
            dolfinx.geometry.compute_collisions_points(bb_tree_ep, coord)
            for coord in self.ep_coords
        ]
        adj_ep = [
            dolfinx.geometry.compute_colliding_cells(self.ep_mesh, cells, coords)
            for cells, coords in zip(potential_colliding_cells_ep, self.ep_coords)
        ]
        self.indices_ep = [np.flatnonzero(adj.offsets[1:] - adj.offsets[:-1]) for adj in adj_ep]
        self.cells_ep = [
            adj.array[adj.offsets[indices]] for adj, indices in zip(adj_ep, self.indices_ep)
        ]  # Get the cells that are colliding with the point

        self.points_on_proc_ep = [
            coords[indices] for coords, indices in zip(self.ep_coords, self.indices_ep)
        ]  # Get the points that are colliding with the cell

    @property
    def ep_mesh(self) -> dolfinx.mesh.Mesh:
        """Get the mesh for EP."""
        return self.ep_ode_space.mesh

    @property
    def mech_mesh(self) -> dolfinx.mesh.Mesh:
        """Get the mesh for mechanics."""
        return self.problem.geometry.mesh

    @property
    def comm(self):
        return self.ep_ode_space.mesh.comm

    @property
    def t(self):
        return self._t

    @property
    def out_ep_names(self):
        return list(set(self.out_ep_var_names) | set(self.out_ep_coord_names))

    @property
    def outdir(self):
        return Path(self.config["sim"]["outdir"])

    @property
    def disp_file(self):
        return self.outdir / "displacement.xdmf"

    def _broadcast(self, values, u, indices):
        bs = u.function_space.dofmap.index_map_bs
        # Create array to store values and fill with -inf
        # to ensure that all points are included in the allreduce
        # with op=MPI.MAX
        u_out = np.ones((1, bs), dtype=np.float64) * -np.inf
        # Fill in values for points on this process
        u_out[indices, :] = values
        # Now loop over all processes and find the maximum value

        if bs > 1:
            # If block size is larger than 1, loop over blocks
            for j in range(bs):
                u_out[0, j] = self.comm.allreduce(u_out[0, j], op=MPI.MAX)
        else:
            u_out[0] = self.comm.allreduce(u_out[0], op=MPI.MAX)

        return np.squeeze(u_out)

    def _eval_mech(self, u, i):
        values = u.eval(self.points_on_proc_mech[i], self.cells_mech[i])
        indices = self.indices_mech[i]
        return self._broadcast(values, u, indices)

    def _eval_ep(self, u, i):
        values = u.eval(self.ep_coords[i], self.cells_ep[i])
        indices = self.indices_ep[i]
        return self._broadcast(values, u, indices)

    def write_node_data_mech(self, i):
        for var_nr, data in enumerate(self.config["output"]["point_mech"]):
            out_mech_var = data["name"]
            # Trace variable in coordinate
            self.out_mech_example_nodes[out_mech_var][i] = self._eval_mech(
                self.mech_variables[out_mech_var], var_nr
            )

            # Compute volume averages
            self.out_mech_volume_average_timeseries[out_mech_var][i] = (
                compute_function_average_over_mesh(
                    self.mech_variables[out_mech_var], self.mech_mesh
                )
            )

    def write_node_data_ep(self, i):
        for var_nr, data in enumerate(self.config["output"]["point_ep"]):
            out_ep_var = data["name"]
            # Trace variable in coordinate
            self.out_ep_example_nodes[out_ep_var][i] = self._eval_ep(
                self.out_ep_funcs[out_ep_var], var_nr
            )

            # Compute volume averages
            self.out_ep_volume_average_timeseries[out_ep_var][i] = (
                compute_function_average_over_mesh(self.out_ep_funcs[out_ep_var], self.ep_mesh)
            )

    def write_disp(self, j):
        self.vtx_disp.write(j)
        self.vtx_ep.write(j)

    def write_ep(self, j):
        for out_ep_var in self.out_ep_var_names:
            with dolfin.XDMFFile(self.out_ep_files[out_ep_var].as_posix()) as file:
                file.write_checkpoint(
                    self.out_ep_funcs[out_ep_var],
                    out_ep_var,
                    j,
                    dolfin.XDMFFile.Encoding.HDF5,
                    True,
                )
        for out_mech_var in self.out_mech_var_names:
            with dolfin.XDMFFile(self.out_mech_files[out_mech_var].as_posix()) as file:
                file.write_checkpoint(
                    self.mech_variables[out_mech_var],
                    out_mech_var,
                    j,
                    dolfin.XDMFFile.Encoding.HDF5,
                    True,
                )

    def finalize(self, inds, plot_results=True):
        self.timers.finalize(comm=self.comm, outdir=self.outdir)
        # Write averaged results for later analysis
        for out_ep_var in self.out_ep_coord_names:
            # with open(Path(outdir / f"{out_ep_var}_out_ep_volume_average.txt"), "w") as f:
            np.savetxt(
                self.outdir / f"{out_ep_var}_out_ep_volume_average.txt",
                self.out_ep_volume_average_timeseries[out_ep_var][inds],
            )

        for out_mech_var in self.out_mech_coord_names:
            # with open(Path(outdir / f"{out_mech_var}_out_mech_volume_average.txt"), "w") as f:
            np.savetxt(
                self.outdir / f"{out_mech_var}_out_mech_volume_average.txt",
                self.out_mech_volume_average_timeseries[out_mech_var][inds],
            )

        # Write point traces for later analysis
        for var_nr, data in enumerate(self.config["output"]["point_ep"]):
            out_ep_var = data["name"]
            x = np.squeeze(self.ep_coords[var_nr])
            path = self.outdir / f"{out_ep_var}_ep_coord{x[0]},{x[1]},{x[2]}.txt".replace(  # noqa: E501
                " ", ""
            )
            np.savetxt(path, self.out_ep_example_nodes[out_ep_var][inds])

        for var_nr, data in enumerate(self.config["output"]["point_mech"]):
            out_mech_var = data["name"]
            x = np.squeeze(self.mech_coords[var_nr])
            path = (
                self.outdir / f"{out_mech_var}_mech_coord{x[0]},{x[1]},{x[2]}.txt"  # noqa: E501
            )
            np.savetxt(path, self.out_mech_example_nodes[out_mech_var][inds])

        print(f"Solved on {100 * len(inds) / len(self.t)}% of the time steps")
        inds = np.array(inds)

        if plot_results:
            fig, ax = plt.subplots(len(self.out_ep_coord_names), 1, figsize=(10, 10))
            if len(self.out_ep_coord_names) == 1:
                ax = np.array([ax])
            for i, out_ep_var in enumerate(self.out_ep_coord_names):
                ax[i].plot(self.t[inds], self.out_ep_volume_average_timeseries[out_ep_var][inds])
                ax[i].set_title(f"{out_ep_var} volume average")
                ax[i].set_xlabel("Time (ms)")
            fig.tight_layout()
            fig.savefig(self.outdir / "out_ep_volume_averages.png")

            fig, ax = plt.subplots(len(self.out_ep_coord_names), 1, figsize=(10, 10))
            if len(self.out_ep_coord_names) == 1:
                ax = np.array([ax])
            for var_nr, data in enumerate(self.config["output"]["point_ep"]):
                out_ep_var = data["name"]
                ax[var_nr].plot(self.t[inds], self.out_ep_example_nodes[out_ep_var][inds])
                ax[var_nr].set_title(f"{out_ep_var} in coord {self.ep_coords[var_nr]}")
                ax[var_nr].set_xlabel("Time (ms)")
            fig.tight_layout()
            fig.savefig(self.outdir / "out_ep_coord.png")

            fig, ax = plt.subplots(len(self.out_mech_coord_names), 1, figsize=(10, 10))
            if len(self.out_mech_coord_names) == 1:
                ax = np.array([ax])
            for i, out_mech_var in enumerate(self.out_mech_coord_names):
                ax[i].plot(
                    self.t[inds], self.out_mech_volume_average_timeseries[out_mech_var][inds]
                )
                ax[i].set_title(f"{out_mech_var} volume average")
                ax[i].set_xlabel("Time (ms)")
            fig.tight_layout()
            fig.savefig(self.outdir / "out_mech_volume_averages.png")

            fig, ax = plt.subplots(len(self.out_mech_coord_names), 1, figsize=(10, 10))
            if len(self.out_mech_coord_names) == 1:
                ax = np.array([ax])

            for var_nr, data in enumerate(self.config["output"]["point_mech"]):
                out_mech_var = data["name"]
                ax[var_nr].plot(self.t[inds], self.out_mech_example_nodes[out_mech_var][inds])
                ax[var_nr].set_title(f"{out_mech_var} in coord {self.mech_coords[var_nr]}")
                ax[var_nr].set_xlabel("Time (ms)")
            fig.tight_layout()
            fig.savefig(self.outdir / "out_mech_coord.png")
