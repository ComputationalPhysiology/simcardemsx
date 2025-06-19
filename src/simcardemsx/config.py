from typing import Annotated

from pint import Quantity
from pydantic import BaseModel, Field
from pydantic_pint import PydanticPintQuantity


class Conductivity(BaseModel):
    sigma_el: Annotated[Quantity, PydanticPintQuantity("S/m")] = Field(
        ...,
        description="Extracellular conductivity in the longitudinal direction (S/m)",
    )
    sigma_et: Annotated[Quantity, PydanticPintQuantity("S/m")] = Field(
        ...,
        description="Extracellular conductivity in the transverse direction (S/m)",
    )
    sigma_il: Annotated[Quantity, PydanticPintQuantity("S/m")] = Field(
        ...,
        description="Intracellular conductivity in the longitudinal direction (S/m)",
    )
    sigma_it: Annotated[Quantity, PydanticPintQuantity("S/m")] = Field(
        ...,
        description="Intracellular conductivity in the transverse direction (S/m)",
    )


# class Stimulus(BaseModel):
#     amplitude: NumberWithUnit = Field(..., description="Amplitude of the stimulus (A/m^2)")
#     duration: NumberWithUnit = Field(..., description="Duration of the stimulus (s)")
#     start: NumberWithUnit = Field(..., description="Start time of the stimulus (s)")
#     xmax: float = Field(..., description="Maximum x-coordinate for the stimulus region (m)")
#     xmin: float = Field(..., description="Minimum x-coordinate for the stimulus region (m)")
#     ymax: float = Field(..., description="Maximum y-coordinate for the stimulus region (m)")
#     ymin: float = Field(..., description="Minimum y-coordinate for the stimulus region (m)")
#     zmax: float = Field(..., description="Maximum z-coordinate for the stimulus region (m)")
#     zmin: float = Field(..., description="Minimum z-coordinate for the stimulus region (m)")


# class EPConfig(BaseModel):
#     conductivities: Conductivity
#     stimulus: Stimulus
#     chi: float = Field(..., description="Chi value for the electrophysiology model (S/m)")
#     C_m: float = Field(..., description="Membrane capacitance (F/m^2)")
