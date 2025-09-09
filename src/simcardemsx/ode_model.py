from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import dolfinx
import gotranx
import numpy as np

from .interpolation import MissingValue
from .ode2mechanics import ode2mechanics


def setup_ep_ode_model(
    odefile, ep_module_file=Path("ep_model.py"), mechanics_module_file=Path("mechanics_model.py"),
):
    if not (ep_module_file.is_file() and mechanics_module_file.is_file()):
        ode = gotranx.load_ode(odefile)

        mechanics_comp = ode.get_component("mechanics")
        mechanics_ode = mechanics_comp.to_ode()
        ep_ode = ode - mechanics_comp

        code_mech = ode2mechanics(mechanics_ode, missing_values=ep_ode.missing_variables)
        Path(mechanics_module_file).write_text(code_mech)

        # Generate code for the electrophysiology model
        code_ep = gotranx.cli.gotran2py.get_code(
            ep_ode,
            scheme=[gotranx.schemes.Scheme.generalized_rush_larsen],
            missing_values=mechanics_ode.missing_variables,
        )

        Path(ep_module_file).write_text(code_ep)
        # Currently 3D mech needs to be written manually

    return __import__(str(ep_module_file.stem)).__dict__, __import__(
        str(mechanics_module_file.stem),
    ).__dict__


@dataclass
class ODEModel:
    odefile: Path
    mech_ode_space: dolfinx.fem.FunctionSpace
    ep_ode_space: dolfinx.fem.FunctionSpace
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        parameters = type(self).default_parameters()
        parameters.update(self.parameters)
        self.parameters = parameters

        self.ep_module, self.mechanics_module = setup_ep_ode_model(
            self.odefile,
            ep_module_file=self.parameters["ep_module_file"],
            mechanics_module_file=self.parameters["mechanics_module_file"],
        )
        self._setup_missing_values()
        # fgr_ep = ep_model["forward_generalized_rush_larsen"]

        # mv_ep = ep_model["missing_values"]

        # Get initial values from the EP model
        # y_ep_ = self.module["init_state_values"]()
        # p_ep_ = self.module["init_parameter_values"](i_Stim_Amplitude=0.0)

    @staticmethod
    def default_parameters() -> dict[str, Any]:
        return {
            "ep_module_file": Path("ep_model.py"),
            "mechanics_module_file": Path("mechanics_model.py"),
        }

    def _setup_missing_values(self):
        ep_missing_values_ = np.zeros(len(self.ep_module["missing"]))
        mechanics_missing_values_ = np.zeros(len(self.mechanics_module["missing"]))

        self.missing_mech = MissingValue(
            element=self.mech_ode_space.ufl_element(),
            interpolation_element=self.ep_ode_space.ufl_element(),
            mechanics_mesh=self.mech_ode_space.mesh,
            ep_mesh=self.ep_ode_space.mesh,
            num_values=len(mechanics_missing_values_),
            names=list(self.mechanics_module["missing"].keys()),
        )

        self.missing_ep = MissingValue(
            element=self.ep_ode_space.ufl_element(),
            interpolation_element=self.mech_ode_space.ufl_element(),
            mechanics_mesh=self.mech_ode_space.mesh,
            ep_mesh=self.ep_ode_space.mesh,
            num_values=len(ep_missing_values_),
            names=list(self.ep_module["missing"].keys()),
        )

        self.missing_ep.values_mechanics.T[:] = ep_missing_values_
        self.missing_ep.values_ep.T[:] = ep_missing_values_
        # ode_missing_variables = missing_ep.values_ep
        # missing_ep_args = (missing_ep.values_ep,)

        self.missing_mech.values_ep.T[:] = mechanics_missing_values_
        self.missing_mech.values_mechanics.T[:] = mechanics_missing_values_
        self.missing_mech.mechanics_values_to_function()  # Assign initial values to mech functions

        # Use previous cai in mech to be consistent across splitting schemes
        self.prev_missing_mech = MissingValue(
            element=self.mech_ode_space.ufl_element(),
            interpolation_element=self.ep_ode_space.ufl_element(),
            mechanics_mesh=self.mech_ode_space.mesh,
            ep_mesh=self.ep_ode_space.mesh,
            num_values=len(mechanics_missing_values_),
            names=list(self.mechanics_module["missing"].keys()),
        )
        self.update_prev_missing_mech()

    def update_prev_missing_mech(self):
        for i in range(self.missing_mech.num_values):
            self.prev_missing_mech.u_mechanics[i].x.array[:] = self.missing_mech.values_mechanics[i]

    def update_ep_missing_values(self, t, values, parameters):
        # Extract missing values for the mechanics step from the ep model (ep function space)
        missing_ep_values = self.mv(t, values, parameters, self.missing_ep.values_ep)

        for k in range(self.missing_mech.num_values):
            self.missing_mech.u_ep_int[k].x.array[:] = missing_ep_values[k, :]

    @property
    def fgr(self):
        return self.ep_module["generalized_rush_larsen"]

    @property
    def mv(self):
        return self.ep_module["missing_values"]

    @property
    def y(self):
        return self.ep_module["init_state_values"]

    @property
    def p(self):
        return self.ep_module["init_parameter_values"]
