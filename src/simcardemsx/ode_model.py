from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, NamedTuple

import dolfinx
import gotranx
import numpy as np

from .interpolation import MissingValue
from .ode2mechanics import ode2mechanics
from .utils import load_module_from_path


class ODEModulePaths(NamedTuple):
    """Paths of the two modules written by :func:`generate_ode_code`."""

    ep: Path
    mechanics: Path


class ODEModules(NamedTuple):
    """The two generated modules, loaded.

    Both are needed to set up the coupling: each declares, in its own
    ``missing`` dict, the variables it needs *from the other side*.
    """

    ep: ModuleType
    mechanics: ModuleType


def _unit_map(ode) -> Dict[str, str | None]:
    """Name-to-unit for everything the ODE declares.

    gotranx does not propagate units into generated code, so the only record of
    them is the ``.ode`` source. Built from the *whole* ODE rather than per
    side, because each side needs the units of what it receives from the other:
    the EP module has to know what ``J_TRPN`` arrives in, and ``J_TRPN`` is
    defined on the mechanics side.

    A variable the source gives no unit for maps to ``None``. That is common --
    the shipped ToR-ORd files declare units on many parameters but on none of
    the variables that actually cross -- and it means "unknown", never
    "dimensionless". A consumer must not treat the two as the same.
    """
    units: Dict[str, str | None] = {}
    for group in (ode.states, ode.parameters, ode.intermediates):
        for item in group:
            units.setdefault(item.name, getattr(item, "unit_str", None))
    return units


def _append_units(path: Path, units: Dict[str, str | None]) -> None:
    """Record the unit map in an already-written generated module."""
    with path.open("a") as fh:
        fh.write("\n\n# Units as declared in the .ode source; None means the\n")
        fh.write("# source did not say, which is not the same as dimensionless.\n")
        fh.write(f"units = {units!r}\n")


def generate_ode_code(odefile: Path, output_dir: Path) -> ODEModulePaths:
    """
    Pre-processing step: Generates EP and Mechanics Python modules
    from a gotranx ODE file and writes them to the specified directory.

    Returns the paths of both written modules. Prefer :func:`load_ode_modules`,
    which also loads them -- the mechanics module is not optional, since it is
    the only place that records how many variables the mechanics side needs
    from EP.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ep_module_file = output_dir / "ep_model.py"
    mechanics_module_file = output_dir / "mechanics_model.py"

    ode = gotranx.load_ode(odefile)
    mechanics_comp = ode.get_component("mechanics")
    mechanics_ode = mechanics_comp.to_ode()
    ep_ode = ode - mechanics_comp

    # Generate Mechanics ODE
    code_mech = ode2mechanics(mechanics_ode, missing_values=ep_ode.missing_variables)
    mechanics_module_file.write_text(code_mech)

    # Generate EP ODE
    code_ep = gotranx.cli.gotran2py.get_code(
        ep_ode,
        scheme=[gotranx.schemes.Scheme.generalized_rush_larsen],
        missing_values=mechanics_ode.missing_variables,
    )
    ep_module_file.write_text(code_ep)

    units = _unit_map(ode)
    _append_units(ep_module_file, units)
    _append_units(mechanics_module_file, units)

    return ODEModulePaths(ep=ep_module_file, mechanics=mechanics_module_file)


def load_ode_modules(odefile: Path, output_dir: Path) -> ODEModules:
    """Generate both ODE modules from ``odefile`` and load them.

    Use this rather than calling :func:`generate_ode_code` and loading only
    ``ep_model``: the split is described jointly by the two modules, and
    :class:`RuntimeODEModel` needs both to size its transfer buffers.
    """
    paths = generate_ode_code(odefile, output_dir)
    return ODEModules(
        ep=load_module_from_path("ep_model", paths.ep),
        mechanics=load_module_from_path("mechanics_model", paths.mechanics),
    )


@dataclass
class RuntimeODEModel:
    """
    Runtime class that handles data structures and FEniCSx function spaces
    for the ODE components, entirely decoupled from code generation.
    """

    ep_module_dict: Dict[str, Any]
    mech_module_dict: Dict[str, Any]
    mech_ode_space: dolfinx.fem.FunctionSpace
    ep_ode_space: dolfinx.fem.FunctionSpace

    def __post_init__(self):
        self._setup_missing_values()

    def _setup_missing_values(self):
        # Each generated module's `missing` dict names the variables that side
        # needs *from the other*, so the two counts come from opposite modules
        # and are generally different. For the ODE files in
        # numerical_experiments/odefiles they are:
        #
        #     split          ep.missing        mechanics.missing
        #     caisplit       J_TRPN            cai
        #     catrpnsplit    (none)            CaTrpn
        #     zetasplit      Zetas, Zetaw      XS, XW
        #
        # A side that needs nothing gets no `missing` entry at all from gotranx
        # -- hence .get(), not [] -- which is why CaTrpn appears twice above
        # with an empty left column.
        ep_missing_values_ = np.zeros(len(self.ep_module_dict.get("missing", ())))
        mechanics_missing_values_ = np.zeros(len(self.mech_module_dict.get("missing", ())))

        self.missing_mech = MissingValue(
            element=self.mech_ode_space.ufl_element(),
            interpolation_element=self.ep_ode_space.ufl_element(),
            ep_mesh=self.ep_ode_space.mesh,
            num_values=len(mechanics_missing_values_),
        )

        self.missing_ep = MissingValue(
            element=self.ep_ode_space.ufl_element(),
            interpolation_element=self.mech_ode_space.ufl_element(),
            ep_mesh=self.ep_ode_space.mesh,
            num_values=len(ep_missing_values_),
        )

        self.missing_ep.values_ep.T[:] = ep_missing_values_

        self.missing_mech.values_ep.T[:] = mechanics_missing_values_

    # -- the split, as the generated modules describe it ---------------------

    @property
    def ep_missing(self) -> Dict[str, int]:
        """What the EP side needs from mechanics, name to row."""
        return self.ep_module_dict.get("missing", {}) or {}

    @property
    def mech_missing(self) -> Dict[str, int]:
        """What the mechanics side needs from EP, name to row."""
        return self.mech_module_dict.get("missing", {}) or {}

    @property
    def units(self) -> Dict[str, Any]:
        """Units as the ODE source declared them; see :func:`generate_ode_code`."""
        return self.ep_module_dict.get("units", {}) or {}

    # -- EP-mesh Functions the transfers pass through ------------------------

    def ep_transfer_sources(self) -> Dict[str, dolfinx.fem.Function]:
        """EP-mesh Functions holding what mechanics is missing, keyed by name."""
        return {name: self.missing_mech.u_ep_int[i] for name, i in self.mech_missing.items()}

    def ep_transfer_targets(self) -> Dict[str, dolfinx.fem.Function]:
        """EP-mesh Functions receiving what EP is missing, keyed by name."""
        return {name: self.missing_ep.u_ep[i] for name, i in self.ep_missing.items()}

    def ep_transfer_source_functions(self):
        """The same, positionally -- row order is the generated module's."""
        return self.missing_mech.u_ep_int

    def ep_transfer_target_functions(self):
        return self.missing_ep.u_ep

    def commit_ep_missing_values(self) -> None:
        """Read the EP-mesh Functions into the array the EP solver consumes."""
        self.missing_ep.ep_function_to_values()

    def update_ep_missing_values(self, t, values, parameters):
        # Calls the function from the injected dictionary
        missing_ep_values = self.mv(t, values, parameters, self.missing_ep.values_ep)

        for k in range(self.missing_mech.num_values):
            self.missing_mech.u_ep_int[k].x.array[:] = missing_ep_values[k, :]

    @property
    def fgr(self):
        return self.ep_module_dict["generalized_rush_larsen"]

    @property
    def mv(self):
        return self.ep_module_dict["missing_values"]

    @property
    def y(self):
        return self.ep_module_dict["init_state_values"]

    @property
    def p(self):
        return self.ep_module_dict["init_parameter_values"]
