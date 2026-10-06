import json
import os
import subprocess
import sys
from pathlib import Path

from mpi4py import MPI

import dolfinx
import pytest
import ufl

from simcardemsx.ode_model import generate_ode_code, load_ode_modules


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
    assert "def generalized_rush_larsen" in mech_code
    assert "import ufl" in mech_code


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


@pytest.mark.parametrize(
    "split, mech_missing, ep_missing",
    [
        ("caisplit", {"cai"}, {"J_TRPN"}),
        ("zetasplit", {"XS", "XW"}, {"Zetas", "Zetaw"}),
        ("catrpnsplit", {"CaTrpn"}, set()),
    ],
)
def test_generated_modules_describe_the_split(split_modules, split, mech_missing, ep_missing):
    """`provides` on each module names what it hands to the *other* side.

    It is derived from the other side's `missing` dict at generation time
    (Task 3), so `mech.provides` mirrors `ep.missing` and vice versa -- here
    that means `mech.provides` == `ep_missing` and `ep.provides` ==
    `mech_missing`.
    """
    ep, mech = split_modules[split]
    assert set(getattr(mech, "missing", {})) == mech_missing
    assert set(mech.provides) == ep_missing
    assert set(ep.provides) == mech_missing
    assert "Ta" in mech.monitor
    assert {"lmbda", "dLambda"} <= set(mech.parameter)


def test_mechanics_module_emits_ufl(split_modules):
    """The mechanics module's generated scheme must build a UFL expression
    tree, not compute a numpy value -- that is the whole point of switching
    to `gotranx.cli.gotran2ufl`. Every parameter/state/missing-variable is
    passed in as a `dolfinx.fem.Constant` here, never a Python float: the
    generated code uses `ufl.Or`/`ufl.lt` etc., which raise on plain Python
    bools produced by comparing floats directly.
    """
    _, mech = split_modules["caisplit"]
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    c = lambda v: dolfinx.fem.Constant(mesh, float(v))
    p = [c(v) for v in mech.init_parameter_values()]
    s = [c(v) for v in mech.init_state_values()]
    out = mech.generalized_rush_larsen(s, c(0.0), c(1.0), p, [c(1e-4)])
    assert all(isinstance(e, ufl.core.expr.Expr) for e in out)


#: Two hash seeds under which gotranx (1.8.0) generates the Ca_i split's EP states in
#: different orders.
HASH_SEEDS = ("1", "3")


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="gotranx's sort_assignments adds each assignment's dependencies to its "
    "TopologicalSorter in set order, so the generated state order depends on "
    "PYTHONHASHSEED. Restarts no longer depend on it (a checkpoint is read by name), "
    "but the fix belongs in gotranx: sort the dependencies. Remove this mark then.",
)
def test_generated_state_order_does_not_depend_on_the_hash_seed(tmp_path):
    """The generated modules' state order is the same in every process, whatever its
    ``PYTHONHASHSEED``.

    The two orders give the same numbers (only the order of the states and of the
    statements differs), and a restart maps a checkpoint's states by name, so nothing in
    simcardemsx depends on this any more. It tracks gotranx's nondeterminism.
    """
    odefile = (
        Path(__file__).parent.parent
        / "numerical_experiments"
        / "odefiles"
        / "ToRORd_dynCl_endo_caisplit.ode"
    )
    script = (
        "import json, sys; from pathlib import Path; "
        "from simcardemsx.ode_model import load_ode_modules; "
        "m = load_ode_modules(Path(sys.argv[1]), Path(sys.argv[2])); "
        "print(json.dumps([m.ep.state, m.mechanics.state]))"
    )
    orders = []
    for seed in HASH_SEEDS:
        result = subprocess.run(
            [sys.executable, "-c", script, str(odefile), str(tmp_path / seed)],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
            timeout=300,
        )
        # Not an assertion: a crash in the subprocess fails, not satisfies, the xfail.
        if result.returncode != 0:
            pytest.fail(f"PYTHONHASHSEED={seed}: {result.stderr[-3000:]}")
        orders.append(json.loads(result.stdout.strip().splitlines()[-1]))
    assert orders[0] == orders[1]
