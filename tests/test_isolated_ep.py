# tests/test_isolated_ep.py


from mpi4py import MPI

import basix.ufl
import dolfinx
import numpy as np

from simcardemsx.ode_model import RuntimeODEModel, generate_ode_code
from simcardemsx.utils import load_module_from_path


def test_mechano_electric_feedback(tmp_path):
    """
    Test that the EP ODE state responds to changes in mechanical stretch (lambda).
    """
    # 1. Create a minimal ODE where stretch (lmbda) affects voltage (v)
    # Make lmbda a state in the mechanics component so it becomes a true missing variable!
    ode_content = """
    parameters("ep", a=1.0)
    states("ep", v=0.0)

    states("mechanics", lmbda=1.0)
    expressions("mechanics")
    dlmbda_dt = 0.0

    expressions("ep")
    dv_dt = a * lmbda
    """

    ode_file = tmp_path / "mef_test.ode"
    ode_file.write_text(ode_content)

    # 2. Generate and load the code dynamically
    generate_ode_code(ode_file, tmp_path)
    ep_module = load_module_from_path("ep_model", tmp_path / "ep_model.py")

    # 3. Setup FEniCSx spaces
    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_unit_cube(comm, 1, 1, 1)
    element = basix.ufl.element(basix.ElementFamily.P, mesh.basix_cell(), 1)
    V = dolfinx.fem.functionspace(mesh, element)

    # 4. Initialize RuntimeODEModel
    model = RuntimeODEModel(ep_module_dict=ep_module.__dict__, mech_ode_space=V, ep_ode_space=V)

    # 5. Extract initial states and parameters
    num_dofs = V.dofmap.index_map.size_local
    state_baseline = np.zeros((1, num_dofs))
    state_stretched = np.zeros((1, num_dofs))

    state_baseline[0, :] = ep_module.init_state_values(v=1.0)
    state_stretched[0, :] = ep_module.init_state_values(v=1.0)

    params = np.zeros((1, num_dofs))
    params[0, :] = ep_module.init_parameter_values(a=1.0)

    # --- SIMULATION ---

    # Step A: Baseline (lambda = 1.0)
    baseline_lambda = np.ones((1, num_dofs)) * 1.0

    # Capture the returned updated states!
    state_baseline = model.fgr(
        states=state_baseline,
        t=0.0,
        parameters=params,
        missing_variables=baseline_lambda,
        dt=1.0,
    )

    # Step B: Stretched (lambda = 1.2)
    stretched_lambda = np.ones((1, num_dofs)) * 1.2

    # Capture the returned updated states!
    state_stretched = model.fgr(
        states=state_stretched,
        t=0.0,
        parameters=params,
        missing_variables=stretched_lambda,
        dt=1.0,
    )

    # --- ASSERTIONS ---

    assert np.all(state_baseline > 0.0), "Baseline state did not advance."

    # Prove MEF: Stretched tissue must result in a different voltage than baseline tissue
    difference = np.abs(state_stretched - state_baseline)
    assert np.all(difference > 1e-6), "EP state did not respond to mechanical stretch!"
