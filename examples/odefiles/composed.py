"""ToR-ORd + Land, composed from odezoo at first run, split three ways.

ToR-ORd's CellML is licensed GPL-3.0-only and this repository is MIT, so the composed
``.ode`` files are never committed: :func:`torord_land` fetches both models from odezoo
(pinned by version), composes them with ``torord_land.glue.ode`` and splits the result,
writing the file into ``COMPOSED_DIR`` (gitignored) or a folder of the caller's.

:func:`place` puts Land's calibration on each side of a split, and :func:`initial_states`
gives the initial states the composed model needs set.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import gotranx

try:
    import odezoo
except ImportError:  # pragma: no cover - reported by torord_land
    odezoo = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from simcardemsx.ode_model import ODEModules

ODEFILES = Path(__file__).resolve().parent
COMPOSED_DIR = ODEFILES / "composed"

# odezoo model name -> the version the composition was made and checked with
ODEZOO_VERSIONS: dict[str, str] = {
    "torord_dyn_cl": "github-11e463d",
    "land_2017": "github-6e5a5aa",
}

# ``ODE.split("mechanics", inputs=, outputs=)``: what each split puts on the mechanics side
SPLITS: dict[str, dict[str, list[str]]] = {
    "zeta": {"inputs": ["S", "W"], "outputs": ["Zs", "Zw", "Ta"]},
    "cai": {"inputs": ["cai"], "outputs": ["Ta", "J_TRPN"]},
    "catrpn": {"inputs": ["CaTRPN"], "outputs": ["Ta"]},
}

# Land's calibration for ToR-ORd (Tref in Pa)
LAND_CALIBRATION: dict[str, float] = {
    "ca50_ref": 0.805,
    "Tref": 120_000.0,
    "ku": 40.0,
    "kuw": 182.0,
    "kws": 12.0,
    "nTm": 2.4,
}

# Land starts with all its crossbridges unbound: TmB = 1
INITIAL_STATES: dict[str, float] = {"B": 1.0}

_RENAME = {"a": "a_land", "h": "h_land", "dLambda": "dLambda_land"}


def _fetch(name: str) -> Path:
    version = ODEZOO_VERSIONS[name]
    if odezoo is None:
        raise RuntimeError(f"odezoo is not installed: it fetches {name} {version}")
    try:
        return Path(odezoo.get(name, version=version))
    except Exception as e:
        raise RuntimeError(
            f"odezoo could not fetch the model {name} at version {version}: {e}. "
            "The first run needs network access; odezoo caches what it fetches.",
        ) from e


def torord_land(split: str, folder: Path | None = None) -> Path:
    """Compose ToR-ORd and Land, split them, and return the saved ``.ode`` file.

    The file is ``<folder or COMPOSED_DIR>/torord_land_<split>.ode``. Composing is
    deterministic, and the file is only written when its text differs from the file's, so
    a correct file keeps its mtime and sha256 (which key the cache of generated code).

    Raises
    ------
    ValueError
        For a split not in :data:`SPLITS`.
    RuntimeError
        If odezoo cannot fetch a model.
    """
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}: choose one of {sorted(SPLITS)} (SPLITS)")
    torord_file = _fetch("torord_dyn_cl")
    land_file = _fetch("land_2017")
    glue_file = ODEFILES / "torord_land.glue.ode"
    recipe = {
        "time_unit": "ms",
        "glue": glue_file.name,
        "parts": [
            {"name": "ep", "file": str(torord_file), "time_unit": "ms"},
            {"name": "land", "file": str(land_file), "time_unit": "s", "rename": _RENAME},
        ],
    }
    ode = gotranx.compose(recipe, base_dir=ODEFILES)
    path = Path(folder if folder is not None else COMPOSED_DIR) / f"torord_land_{split}.ode"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    ode.split("mechanics", **SPLITS[split]).save(tmp)
    try:
        new = tmp.read_text()
        if not path.exists() or path.read_text() != new:
            tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def place(values: Mapping[str, float], modules: ODEModules) -> dict[str, dict[str, float]]:
    """Each value, for every half of the split that declares it as a parameter.

    ``"ep"`` holds the names the EP module declares, ``"mechanics"`` those the
    ``mechanics`` module declares, and a name may be in both.

    Raises
    ------
    KeyError
        If a name is declared by neither half: it would change nothing.
    """
    placed = {
        "ep": {k: v for k, v in values.items() if k in modules.ep.parameter},
        "mechanics": {k: v for k, v in values.items() if k in modules.mechanics.parameter},
    }
    unused = sorted(set(values) - set(placed["ep"]) - set(placed["mechanics"]))
    if unused:
        raise KeyError(f"Parameters {unused} are declared by neither half of the split")
    return placed


def initial_states(module) -> dict[str, float]:
    """:data:`INITIAL_STATES` restricted to the module's own states."""
    return {k: v for k, v in INITIAL_STATES.items() if k in module.state}
