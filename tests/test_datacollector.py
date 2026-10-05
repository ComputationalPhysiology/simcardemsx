"""What :class:`DataCollector` and its :class:`Timers` write.

The collector is deprecated: the demos write ``results.bp`` and ``log.csv``
(:mod:`simcardemsx.results`) and their ``post.py`` replots from them. It is wired to
:class:`SimulationController`'s callbacks as the slab example used to wire it: the
controller counts steps from 1, and the collector is handed ``step_idx - 1`` whenever
that index is a multiple of the save frequency.
"""

import json

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest

from simcardemsx.controller import SimulationController
from simcardemsx.datacollector import DataCollector, Timers

# Binary fractions, so the times below are exact.
DT_EP = 0.125
N = 2  # EP steps per mechanics step
DT_MECH = N * DT_EP
NUM_MECH_STEPS = 6
SAVE_FREQUENCY_EP = 3
SAVE_FREQUENCY_MECH = 2


def _unit_cube(n: int) -> dolfinx.mesh.Mesh:
    return dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n)


def test_time_axes_are_the_times_the_samples_were_taken(
    split_modules,
    make_ep_solver,
    make_mechanics,
    tmp_path,
):
    modules = split_modules["zetasplit"]
    ep_solver = make_ep_solver(modules.ep, _unit_cube(1))
    problem, backend = make_mechanics(modules.mechanics, _unit_cube(1))
    controller = SimulationController(problem, ep_solver, backend, modules, DT_MECH, DT_EP)

    origin = {"x": 0, "y": 0, "z": 0}
    config = {
        "sim": {
            "dt": DT_EP,
            "N": N,
            "sim_dur": NUM_MECH_STEPS * DT_MECH,
            "save_frequency_ep": SAVE_FREQUENCY_EP,
            "save_frequency_mech": SAVE_FREQUENCY_MECH,
            "outdir": str(tmp_path),
        },
        "output": {
            "all_ep": ["v"],
            "all_mech": ["Ta"],
            "point_ep": [{"name": "v", **origin}],
            "point_mech": [{"name": "Ta", **origin}],
        },
    }
    with pytest.warns(DeprecationWarning, match="results.bp"):
        collector = DataCollector(
            problem=problem,
            ep_ode_space=ep_solver.ode.v_ode.function_space,
            config=config,
            mech_variables={"Ta": backend.active_tension},
        )

    ep_sampled_at: list[float] = []
    mech_sampled_at: list[float] = []

    def on_ep_step(t, ep_step_idx):
        i = ep_step_idx - 1
        if i % SAVE_FREQUENCY_EP == 0:
            ep_sampled_at.append(t)
            collector.write_node_data_ep(i)

    def on_mech_step(t, mech_step_idx, newton_iterations):
        i = mech_step_idx - 1
        if i % SAVE_FREQUENCY_MECH == 0:
            mech_sampled_at.append(t)
            collector.write_node_data_mech(i)

    for _ in range(NUM_MECH_STEPS):
        controller.step(ep_callback=on_ep_step, mech_callback=on_mech_step)
    collector.finalize([], plot_results=False)

    # EP steps 0, 3, 6, 9 (of 12) end at 1, 4, 7, 10 times DT_EP; mechanics steps 0, 2,
    # 4 (of 6) end at 1, 3, 5 times DT_MECH.
    expected_ep = [0.125, 0.5, 0.875, 1.25]
    expected_mech = [0.25, 0.75, 1.25]
    assert ep_sampled_at == expected_ep
    assert mech_sampled_at == expected_mech

    written = {name: np.loadtxt(tmp_path / f"{name}.txt").tolist() for name in ("t_ep", "t_mech")}
    assert written == {"t_ep": expected_ep, "t_mech": expected_mech}


def test_timers_finalize_reports_the_timers_dolfinx_registered(tmp_path):
    # dolfinx registers this one itself, and it is among the names finalize reports.
    _unit_cube(1)
    Timers().finalize(MPI.COMM_WORLD, tmp_path)

    timings = json.loads((tmp_path / "solve_timings.json").read_text())["timings"]
    # dolfinx's timer registry is global to the process: other tests add repetitions.
    reps, wall_seconds = timings["Build BoxMesh (tetrahedra)"]
    assert reps >= 1
    assert wall_seconds >= 0.0
    # Listed among the names to report, but never a dolfinx timer: skipped, not an error.
    assert "Loop total times" not in timings
