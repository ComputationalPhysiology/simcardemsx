# tests/test_isolated_ep.py


import numpy as np

from simcardemsx.ode_model import load_ode_modules


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
    modules = load_ode_modules(ode_file, tmp_path)
    ep_module = modules.ep

    # 3. Extract initial states and parameters, at the 8 vertices of a unit cube
    num_dofs = 8
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
    state_baseline = ep_module.generalized_rush_larsen(
        states=state_baseline,
        t=0.0,
        parameters=params,
        missing_variables=baseline_lambda,
        dt=1.0,
    )

    # Step B: Stretched (lambda = 1.2)
    stretched_lambda = np.ones((1, num_dofs)) * 1.2

    # Capture the returned updated states!
    state_stretched = ep_module.generalized_rush_larsen(
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
