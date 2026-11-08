import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

from mpi4py import MPI

import dolfinx
import pytest
import ufl

from simcardemsx import ode_model
from simcardemsx.ode_model import generate_ode_code
from simcardemsx.utils import load_module_from_path


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


_TINY_ODE = """
states("ep", v=0)
parameters("ep", a=1)
expressions("ep")
dv_dt = a

states("mechanics", XS=0)
expressions("mechanics")
dXS_dt = v - XS
"""


def test_generated_code_is_cached_until_its_inputs_change(tmp_path, monkeypatch):
    """`generate_ode_code` reuses a complete earlier generation of the same `.ode`
    content, and regenerates when the content changes, when the stamp is missing
    (a generation interrupted before it was written) or when forced."""
    import gotranx

    calls = []
    load_ode = gotranx.load_ode
    monkeypatch.setattr(gotranx, "load_ode", lambda f: calls.append(f) or load_ode(f))

    ode_file = tmp_path / "tiny.ode"
    ode_file.write_text(_TINY_ODE)
    out_dir = tmp_path / "generated"

    generate_ode_code(ode_file, out_dir)
    assert len(calls) == 1
    assert (out_dir / ode_model.STAMP_FILE).is_file()

    paths = generate_ode_code(ode_file, out_dir)
    assert len(calls) == 1, "the same content must be reused"
    assert paths.ep.is_file() and paths.mechanics.is_file()

    ode_file.write_text(_TINY_ODE.replace("a=1", "a=2"))
    generate_ode_code(ode_file, out_dir)
    assert len(calls) == 2, "changed content must regenerate"
    ep = load_module_from_path("ep_tiny", out_dir / "ep_model.py")
    assert ep.init_parameter_values()[ep.parameter["a"]] == 2

    (out_dir / ode_model.STAMP_FILE).unlink()
    generate_ode_code(ode_file, out_dir)
    assert len(calls) == 3, "no stamp must regenerate"

    generate_ode_code(ode_file, out_dir, force=True)
    assert len(calls) == 4, "force must regenerate"


def test_the_stamp_names_gotranx_and_the_generator(tmp_path):
    """A gotranx upgrade or a change to `ode_model.py` must invalidate the cache, so
    both are in the stamp, beside the `.ode` content."""
    ode_file = tmp_path / "tiny.ode"
    ode_file.write_text(_TINY_ODE)
    stamp = ode_model._stamp(ode_file)
    assert set(stamp) == {"ode_sha256", "gotranx", "generator_sha256"}
    assert stamp["gotranx"] == importlib.metadata.version("gotranx")


def test_real_split_files_have_the_expected_interfaces(split_modules):
    """Pin the interface of the three ODE files shipped in the repository.

    These are what the FIXME's hardcoded 2 was silently wrong about: only the
    zeta split has two variables crossing into mechanics.

    Note the CaTrpn split needs nothing back from mechanics, and gotranx then
    omits the `missing` attribute entirely rather than emitting an empty dict
    -- hence getattr. Any consumer of these modules has to tolerate that.
    """
    expected = {
        "caisplit": ({"J_TRPN"}, {"cai"}),
        "catrpnsplit": (set(), {"CaTrpn"}),
        "zetasplit": ({"Zetas", "Zetaw"}, {"XS", "XW"}),
    }
    assert set(split_modules) == set(expected)

    for split, (ep_missing, mech_missing) in expected.items():
        modules = split_modules[split]
        assert set(getattr(modules.ep, "missing", {})) == ep_missing, split
        assert set(getattr(modules.mechanics, "missing", {})) == mech_missing, split


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


#: Two hash seeds under which gotranx before 2.2.0 generated the Ca_i split's EP states
#: in different orders.
HASH_SEEDS = ("1", "3")


def test_generated_state_order_does_not_depend_on_the_hash_seed(tmp_path):
    """The generated modules' state order is the same in every process, whatever its
    ``PYTHONHASHSEED``.

    Before 2.2.0, gotranx's sort_assignments added each assignment's dependencies to its
    TopologicalSorter in set order, so the order depended on the hash seed; 2.2.0 sorts
    them. A restart maps a checkpoint's states by name, so it does not rely on this.
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
        if result.returncode != 0:
            pytest.fail(f"PYTHONHASHSEED={seed}: {result.stderr[-3000:]}")
        orders.append(json.loads(result.stdout.strip().splitlines()[-1]))
    assert orders[0] == orders[1]
