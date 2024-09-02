from __future__ import annotations

import asyncio
import inspect
import typing
from typing import Any


async def await_dict_of_tasks(task_dict: dict[str, typing.Awaitable]) -> dict[str, Any]:
    """Util to await a dictionary of tasks as asyncio.gather is kind of garbage"""
    keys = sorted(task_dict.keys())
    coroutines = [task_dict[key] for key in keys]
    coroutines_gathered = await asyncio.gather(*coroutines)
    return dict(zip(keys, coroutines_gathered))


async def process_value(val: Any) -> Any:
    """Helper function to process the value of a potential awaitable.
    This is very simple -- all it does is await the value if its not already resolved.

    :param val: Value to process.
    :return: The value (awaited if it is a coroutine, raw otherwise).
    """
    if not inspect.isawaitable(val):
        return val
    return await val
