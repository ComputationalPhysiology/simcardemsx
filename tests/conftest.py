from pathlib import Path

import pytest

from simcardemsx.ode_model import ODEModules, load_ode_modules

ODEFILES_DIR = Path(__file__).parent.parent / "numerical_experiments" / "odefiles"

SPLITS = ("caisplit", "zetasplit", "catrpnsplit")


@pytest.fixture(scope="session")
def split_modules(tmp_path_factory) -> dict[str, ODEModules]:
    """Generate and load the EP/mechanics module pair for each of the three
    ODE splits shipped in `numerical_experiments/odefiles`.

    Session-scoped: code generation goes through gotranx (parsing + black
    formatting), which is slow, and every test that only reads module state
    (parameters/monitor/missing/provides) can share the result. Each split
    gets its own directory from `tmp_path_factory` so the three
    `ep_model.py`/`mechanics_model.py` files don't collide on disk.
    """
    modules = {}
    for split in SPLITS:
        odefile = ODEFILES_DIR / f"ToRORd_dynCl_endo_{split}.ode"
        output_dir = tmp_path_factory.mktemp(split)
        modules[split] = load_ode_modules(odefile, output_dir)
    return modules
