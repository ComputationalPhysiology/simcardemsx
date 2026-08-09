from mpi4py import MPI

import basix.ufl
import dolfinx
import numpy as np

from simcardemsx.ode_model import RuntimeODEModel, generate_ode_code


def test_runtime_ode_model_with_mock_dict():
    """
    Test that the runtime ODE model can correctly initialize function
    spaces and pass values using a fake generated module dictionary.
    """
    comm = MPI.COMM_WORLD
    mesh_mech = dolfinx.mesh.create_unit_cube(comm, 1, 1, 1)
    mesh_ep = dolfinx.mesh.create_unit_cube(comm, 1, 1, 1)

    element_mech = basix.ufl.element(basix.ElementFamily.P, mesh_mech.basix_cell(), 1)
    element_ep = basix.ufl.element(basix.ElementFamily.P, mesh_ep.basix_cell(), 1)

    V_mech = dolfinx.fem.functionspace(mesh_mech, element_mech)
    V_ep = dolfinx.fem.functionspace(mesh_ep, element_ep)

    # 1. Create a fake generated module dictionary
    def mock_missing_values(t, values, parameters, missing_ep_values):
        # Simulate returning mechanics missing variables (shape: 2, num_dofs)
        return np.ones((2, values.shape[1])) * 7.5

    mock_ep_module = {
        "missing": ["some_missing_ep_variable"],  # Simulating 1 missing EP value
        "missing_values": mock_missing_values,
        "generalized_rush_larsen": lambda: None,
        "init_state_values": lambda: None,
        "init_parameter_values": lambda: None,
    }

    # 2. Initialize the model entirely in memory
    model = RuntimeODEModel(ep_module_dict=mock_ep_module, mech_ode_space=V_mech, ep_ode_space=V_ep)

    assert model.missing_ep.num_values == 1
    assert model.missing_mech.num_values == 2

    # 3. Test value passing logic
    local_size = V_ep.dofmap.index_map.size_local
    ghost_size = V_ep.dofmap.index_map.num_ghosts
    dummy_values = np.zeros((1, local_size + ghost_size))
    model.update_ep_missing_values(t=0.0, values=dummy_values, parameters=None)

    # Assert the mock function was called and values applied to the interpolation function
    assert np.allclose(model.missing_mech.u_ep_int[0].x.array, 7.5)


def test_code_generator_smoke_test(tmp_path):
    """
    Test that the code generator correctly parses a gotranx file
    and writes out the python modules.
    """
    # 1. Create a minimal valid gotranx ODE string with a mechanics component
    ode_content = """
    states("ep", v=0)
    parameters("ep", a=1)
    expressions("ep")
    dv_dt = a

    states("mechanics", XS=0)
    expressions("mechanics")
    dXS_dt = v - XS
    """

    # 2. Write it to a temporary test directory
    ode_file = tmp_path / "test.ode"
    ode_file.write_text(ode_content)

    out_dir = tmp_path / "compiled_odes"

    # 3. Run the generator
    generate_ode_code(ode_file, out_dir)

    # 4. Verify outputs exist and contain Python code
    ep_file = out_dir / "ep_model.py"
    mech_file = out_dir / "mechanics_model.py"

    assert ep_file.exists()
    assert mech_file.exists()

    ep_code = ep_file.read_text()
    assert "def generalized_rush_larsen" in ep_code

    mech_code = mech_file.read_text()
    assert "def" in mech_code
