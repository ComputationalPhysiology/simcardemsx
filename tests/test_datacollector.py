"""The save-index arithmetic in DataCollector.

These writers are called with a 1-based step counter, only on multiples of the
save frequency, into arrays sized for exactly that many saves. Getting the
shift wrong is silent in the middle of a run and fatal at the end of one: slot
zero stays zero and the final save runs off the array.
"""

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest

from simcardemsx.datacollector import DataCollector

from .test_coupler import _problem, _roller_bcs, _zeta_backend


def _config(tmp_path, sim_dur=4.0, dt=0.05, freq_ep=20, freq_mech=1, N=1):
    return {
        "sim": {
            "N": N,
            "dt": dt,
            "sim_dur": sim_dur,
            "outdir": str(tmp_path),
            "save_frequency_ep": freq_ep,
            "save_frequency_mech": freq_mech,
        },
        "output": {
            "all_ep": ["v"],
            "all_mech": ["Ta"],
            "point_ep": [{"name": "v", "x": 0, "y": 0, "z": 0}],
            "point_mech": [{"name": "Ta", "x": 0, "y": 0, "z": 0}],
        },
    }


@pytest.fixture
def collector(tmp_path):
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    backend = _zeta_backend(mesh)
    problem = _problem(mesh, backend, _roller_bcs(mesh))
    ep_space = dolfinx.fem.functionspace(mesh, ("DG", 1))
    return DataCollector(
        problem=problem,
        ep_ode_space=ep_space,
        config=_config(tmp_path),
        mech_variables={"Ta": backend.active_tension},
    )


def test_every_ep_save_lands_in_the_array_including_the_last(collector):
    """A run saves exactly as many times as the array has slots.

    The last save used to run off the end -- at any sim_dur, not just unusual
    ones -- so this failed on the final step of every simulation.
    """
    freq = collector.config["sim"]["save_frequency_ep"]
    n_slots = len(collector.out_ep_example_nodes["v"])
    total_steps = len(collector.t)

    saves = [i for i in range(1, total_steps + 1) if i % freq == 0]
    assert len(saves) == n_slots, "test premise: one save per slot"

    for step in saves:
        collector.write_node_data_ep(step)

    assert len(collector._t_ep) == n_slots


def test_every_mech_save_lands_in_the_array_including_the_last(collector):
    freq = collector.config["sim"]["save_frequency_mech"]
    n_slots = len(collector.out_mech_example_nodes["Ta"])
    saves = [i for i in range(1, n_slots * freq + 1) if i % freq == 0]

    for step in saves:
        collector.write_node_data_mech(step)

    assert len(collector._t_mech) == n_slots


def test_the_first_save_is_not_left_empty(collector):
    """Slot zero used to be skipped, leaving a leading zero in every trace."""
    freq = collector.config["sim"]["save_frequency_ep"]
    collector.out_ep_example_nodes["v"][:] = np.nan

    collector.write_node_data_ep(freq)

    assert not np.isnan(collector.out_ep_example_nodes["v"][0]), (
        "the first save did not write slot 0"
    )
