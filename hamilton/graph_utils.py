from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from types import ModuleType


def is_submodule(child: ModuleType, parent: ModuleType):
    return parent.__name__ in child.__name__


def find_functions(function_module: ModuleType) -> list[tuple[str, Callable]]:
    """Function to determine the set of functions we want to build a graph from.

    This iterates through the function module and grabs all function definitions.
    :return: list of tuples of (func_name, function).
    """

    def valid_fn(fn):
        return (
            inspect.isfunction(fn)
            and not fn.__name__.startswith("_")
            and is_submodule(inspect.getmodule(fn), function_module)
        )

    return [f for f in inspect.getmembers(function_module, predicate=valid_fn)]
