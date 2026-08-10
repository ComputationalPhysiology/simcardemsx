from pathlib import Path

from mpi4py import MPI

import basix.ufl
import dolfinx
import numpy as np
import pytest

from simcardemsx.ode_model import RuntimeODEModel, generate_ode_code, load_ode_modules


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
    # The mechanics side is missing 2 variables, e.g. XS and XW in the zeta split
    mock_mech_module = {"missing": ["XS", "XW"]}

    # 2. Initialize the model entirely in memory
    model = RuntimeODEModel(
        ep_module_dict=mock_ep_module,
        mech_module_dict=mock_mech_module,
        mech_ode_space=V_mech,
        ep_ode_space=V_ep,
    )

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


# Each generated module's `missing` dict names what that side needs *from the
# other*, so the two counts are independent. The pair below is deliberately
# asymmetric: 1 variable crossing EP -> mechanics, 2 crossing back.
SPLIT_ODE = """
parameters("ep", a=1.0)
states("ep", v=0.0, cai=0.0001)
states("mechanics", XS=0.0, XW=0.0)
expressions("mechanics")
dXS_dt = cai - XS
dXW_dt = cai - XW
expressions("ep")
dv_dt = a * XS * XW
dcai_dt = 0.001 * v
"""


@pytest.mark.parametrize(
    "ode_content, n_ep_missing, n_mech_missing",
    [
        # mechanics needs cai (1); EP needs XS and XW (2)
        pytest.param(SPLIT_ODE, 2, 1, id="2-back-1-forward"),
        # ...and a variant needing only one back, so no single hardcoded
        # count can satisfy both cases
        pytest.param(
            SPLIT_ODE.replace("dv_dt = a * XS * XW", "dv_dt = a * XS"),
            1,
            1,
            id="1-back-1-forward",
        ),
    ],
)
def test_missing_counts_follow_the_ode_split(tmp_path, ode_content, n_ep_missing, n_mech_missing):
    """Transfer buffers must be sized from the ODE file, not hardcoded.

    Regression test: the mechanics-side count used to be fixed at 2, which is
    right only for the zeta split. Both caisplit and catrpnsplit need 1, so
    they were silently allocating a buffer of the wrong width.
    """
    ode_file = tmp_path / "split.ode"
    ode_file.write_text(ode_content)

    modules = load_ode_modules(ode_file, tmp_path / "generated")

    assert len(modules.ep.missing) == n_ep_missing
    assert len(modules.mechanics.missing) == n_mech_missing

    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_unit_cube(comm, 1, 1, 1)
    element = basix.ufl.element(basix.ElementFamily.P, mesh.basix_cell(), 1)
    V = dolfinx.fem.functionspace(mesh, element)

    model = RuntimeODEModel(
        ep_module_dict=modules.ep.__dict__,
        mech_module_dict=modules.mechanics.__dict__,
        mech_ode_space=V,
        ep_ode_space=V,
    )

    assert model.missing_ep.num_values == n_ep_missing
    assert model.missing_mech.num_values == n_mech_missing


def test_real_split_files_have_the_expected_interfaces(tmp_path):
    """Pin the interface of the three ODE files shipped in the repository.

    These are what the FIXME's hardcoded 2 was silently wrong about: only the
    zeta split has two variables crossing into mechanics.

    Note the CaTrpn split needs nothing back from mechanics, and gotranx then
    omits the `missing` attribute entirely rather than emitting an empty dict
    -- hence getattr. Any consumer of these modules has to tolerate that.
    """
    odefiles = Path(__file__).parent.parent / "numerical_experiments" / "odefiles"
    expected = {
        "ToRORd_dynCl_endo_caisplit": ({"J_TRPN"}, {"cai"}),
        "ToRORd_dynCl_endo_catrpnsplit": (set(), {"CaTrpn"}),
        "ToRORd_dynCl_endo_zetasplit": ({"Zetas", "Zetaw"}, {"XS", "XW"}),
    }

    for stem, (ep_missing, mech_missing) in expected.items():
        odefile = odefiles / f"{stem}.ode"
        if not odefile.is_file():
            pytest.skip(f"{odefile} not available")
        modules = load_ode_modules(odefile, tmp_path / stem)
        assert set(getattr(modules.ep, "missing", {})) == ep_missing, stem
        assert set(getattr(modules.mechanics, "missing", {})) == mech_missing, stem
