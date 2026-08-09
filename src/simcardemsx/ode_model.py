from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import dolfinx
import gotranx
import numpy as np

from .interpolation import MissingValue
from .ode2mechanics import ode2mechanics


def generate_ode_code(odefile: Path, output_dir: Path) -> None:
    """
    Pre-processing step: Generates EP and Mechanics Python modules
    from a gotranx ODE file and writes them to the specified directory.
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


@dataclass
class RuntimeODEModel:
    """
    Runtime class that handles data structures and FEniCSx function spaces
    for the ODE components, entirely decoupled from code generation.
    """

    ep_module_dict: Dict[str, Any]
    mech_ode_space: dolfinx.fem.FunctionSpace
    ep_ode_space: dolfinx.fem.FunctionSpace

    def __post_init__(self):
        self._setup_missing_values()

    def _setup_missing_values(self):
        ep_missing_values_ = np.zeros(len(self.ep_module_dict["missing"]))
        # FIXME: This should depend on the specific split, hardcoded for now
        mechanics_missing_values_ = np.zeros(2)

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
