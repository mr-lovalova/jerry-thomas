import importlib.metadata as metadata
from collections.abc import Callable
from functools import lru_cache
from typing import Any


PARSERS_EP = "jerrythomas.parsers"
LOADERS_EP = "jerrythomas.loaders"
MAPPERS_EP = "jerrythomas.mappers"
COMBINERS_EP = "jerrythomas.combiners"
BUILD_OPERATIONS_EP = "jerrythomas.operations.build"
RUNTIME_OPERATIONS_EP = "jerrythomas.operations.runtime"


@lru_cache
def load_entrypoint(group: str, name: str) -> Callable[..., Any]:
    entrypoints = metadata.entry_points().select(group=group, name=name)
    if not entrypoints:
        available = ", ".join(
            sorted(
                entrypoint.name
                for entrypoint in metadata.entry_points().select(group=group)
            )
        )
        raise ValueError(
            f"No entry point '{name}' in '{group}'. Available: {available or '(none)'}"
        )
    if len(entrypoints) > 1:
        targets = ", ".join(entrypoint.value for entrypoint in entrypoints)
        raise ValueError(f"Ambiguous entry point '{name}' in '{group}': {targets}")
    entrypoint = next(iter(entrypoints)).load()
    if not callable(entrypoint):
        raise TypeError(f"Entry point '{name}' in '{group}' must be callable")
    return entrypoint
