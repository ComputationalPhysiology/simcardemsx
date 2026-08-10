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
            mechanics_mesh=self.mech_ode_space.mesh,
            ep_mesh=self.ep_ode_space.mesh,
            num_values=len(mechanics_missing_values_),
        )

        self.missing_ep = MissingValue(
            element=self.ep_ode_space.ufl_element(),
            interpolation_element=self.mech_ode_space.ufl_element(),
            mechanics_mesh=self.mech_ode_space.mesh,
            ep_mesh=self.ep_ode_space.mesh,
            num_values=len(ep_missing_values_),
        )

        self.missing_ep.values_mechanics.T[:] = ep_missing_values_
        self.missing_ep.values_ep.T[:] = ep_missing_values_

        self.missing_mech.values_ep.T[:] = mechanics_missing_values_
        self.missing_mech.values_mechanics.T[:] = mechanics_missing_values_
        self.missing_mech.mechanics_values_to_function()

        self.prev_missing_mech = MissingValue(
            element=self.mech_ode_space.ufl_element(),
            interpolation_element=self.ep_ode_space.ufl_element(),
            mechanics_mesh=self.mech_ode_space.mesh,
            ep_mesh=self.ep_ode_space.mesh,
            num_values=len(mechanics_missing_values_),
        )
        self.update_prev_missing_mech()

    def update_prev_missing_mech(self):
        for i in range(self.missing_mech.num_values):
            self.prev_missing_mech.u_mechanics[i].x.array[:] = self.missing_mech.values_mechanics[i]

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
