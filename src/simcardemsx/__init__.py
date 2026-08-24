from importlib.metadata import metadata

meta = metadata("simcardemsx")
__version__ = meta.get("Version")
__author__ = meta.get("Author-email")
__license__ = meta.get("license-expression")
__email__ = meta.get("Author-email")
__program_name__ = meta.get("Name")

from . import backends, controller, interpolation, land, ode_model, utils

__all__ = [
    "backends",
    "land",
    "controller",
    "ode_model",
    "utils",
    "interpolation",
]
