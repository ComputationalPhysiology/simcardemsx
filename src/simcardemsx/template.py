from __future__ import annotations
from textwrap import dedent, indent
import functools
from structlog import get_logger

from gotranx.templates.python import (
    # acc,
    state_index,
    parameter_index,
    monitor_index,
    missing_index,
    # method,
    init_state_values,
    init_parameter_values,
    init_parameter_values,
)


__all__ = [
    "init_state_values",
    "init_parameter_values",
    "method",
    "parameter_index",
    "state_index",
    "monitor_index",
    "missing_index",
]


logger = get_logger()


def method(
    name,
    args,
    states,
    parameters,
    values,
    num_return_values: int,
    missing_variables: str = "",
    **kwargs,
):
    logger.debug(f"Generating method '{name}', with {num_return_values} return values.")
    if len(kwargs) > 0:
        logger.debug(f"Unused kwargs: {kwargs}")

    if name == "missing_values":
        return_name_lst = ["["] + [f"_values_{i}, " for i in range(num_return_values)] + ["]"]
    else:
        return_name_lst = ["["] + [f"_values_{i}, " for i in range(num_return_values)] + ["]"]
    indent_return = indent(f"return {''.join(return_name_lst)}", "    ")
    indent_missing_variables = indent(missing_variables, "    ")
    indent_states = indent(states, "    ")
    indent_parameters = indent(parameters, "    ")
    indent_values = indent(values, "    ")

    return dedent(
        f"""
def {name}({args}):

    # Assign states
{indent_states}

    # Assign parameters
{indent_parameters}
{indent_missing_variables}
    # Assign expressions
{indent_values}

{indent_return}
""",
    )
