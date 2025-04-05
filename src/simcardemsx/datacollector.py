from dataclasses import dataclass, field
from pathlib import Path
import json
import matplotlib.pyplot as plt

import dolfinx
import ufl
import toml
import numpy as np
import adios4dolfinx

from .mechanicsproblem import MechanicsProblem


def compute_function_average_over_mesh(func, mesh):
    volume = dolfinx.fem.assemble_scalar(dolfinx.fem.Constant(mesh, 1.0) * ufl.dx(domain=mesh))
    return dolfinx.fem.assemble_scalar(func * ufl.dx(domain=mesh)) / volume


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

    def finalize(self, outdir: Path):
        self.stop_total()
        timings = dolfinx.common.timings(
            dolfinx.common.TimingClear.keep,
            [
                dolfinx.common.TimingType.wall,
                dolfinx.common.TimingType.user,
                dolfinx.common.TimingType.system,
            ],
        ).str(True)
        print(timings)

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
            [f[f"{coord}"] for coord in ["x", "y", "z"]] for f in self.config["output"]["point_ep"]
        ]

        self.out_mech_coord_names = [f["name"] for f in self.config["output"]["point_mech"]]
        self.mech_coords = [
            [f[f"{coord}"] for coord in ["x", "y", "z"]]
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
        adios4dolfinx.write_mesh(self.disp_file, self.problem.geometry.mesh)

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

    @property
    def ep_mesh(self):
        return self.ep_ode_space.mesh()

    def write_node_data_ep(self, i):
        # Store values to plot time series for given coord
        for var_nr, data in enumerate(self.config["output"]["point_ep"]):
            # Trace variable in coordinate
            out_ep_var = data["name"]
            self.out_ep_example_nodes[out_ep_var][i] = self.out_ep_funcs[out_ep_var](
                self.ep_coords[var_nr]
            )
            # Compute volume averages
            self.out_ep_volume_average_timeseries[out_ep_var][i] = (
                compute_function_average_over_mesh(self.out_ep_funcs[out_ep_var], self.ep_mesh)
            )

    def write_node_data_mech(self, i):
        for var_nr, data in enumerate(self.config["output"]["point_mech"]):
            out_mech_var = data["name"]
            # Trace variable in coordinate
            self.out_mech_example_nodes[out_mech_var][i] = self.mech_variables[out_mech_var](
                self.mech_coords[var_nr]
            )

            # Compute volume averages
            self.out_mech_volume_average_timeseries[out_mech_var][i] = (
                compute_function_average_over_mesh(
                    self.mech_variables[out_mech_var], self.problem.geometry.mesh
                )
            )

    def write_disp(self, j):
        self.vtx_disp.write(j)
        self.vtx_ep.write(j)
        # U, p = self.problem.state.split(deepcopy=True)
        # with dolfin.XDMFFile(self.disp_file.as_posix()) as file:
        #     file.write_checkpoint(U, "disp", j, dolfin.XDMFFile.Encoding.HDF5, True)

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
        self.timers.finalize(outdir=self.outdir)
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
            path = (
                self.outdir
                / f"{out_ep_var}_ep_coord{self.ep_coords[var_nr][0]},{self.ep_coords[var_nr][1]},{self.ep_coords[var_nr][2]}.txt".replace(  # noqa: E501
                    " ", ""
                )
            )
            np.savetxt(path, self.out_ep_example_nodes[out_ep_var][inds])

        for var_nr, data in enumerate(self.config["output"]["point_mech"]):
            out_mech_var = data["name"]
            path = (
                self.outdir
                / f"{out_mech_var}_mech_coord{self.mech_coords[var_nr][0]},{self.mech_coords[var_nr][1]},{self.mech_coords[var_nr][2]}.txt"  # noqa: E501
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
