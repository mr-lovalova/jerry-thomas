import importlib.metadata as metadata
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from urllib.request import url2pathname


PARSERS_EP = "jerrythomas.parsers"
LOADERS_EP = "jerrythomas.loaders"
MAPPERS_EP = "jerrythomas.mappers"
COMBINERS_EP = "jerrythomas.combiners"
TRANSFORMS_EP = "jerrythomas.transforms"
ARTIFACT_OPERATIONS_EP = "jerrythomas.operations.artifact"
OUTPUT_OPERATIONS_EP = "jerrythomas.operations.output"
PLUGIN_GROUPS = frozenset(
    {
        PARSERS_EP,
        LOADERS_EP,
        MAPPERS_EP,
        COMBINERS_EP,
        TRANSFORMS_EP,
        ARTIFACT_OPERATIONS_EP,
        OUTPUT_OPERATIONS_EP,
    }
)


@dataclass(frozen=True)
class PluginDistribution:
    name: str
    version: str
    editable_path: Path | None
    entrypoints: tuple[metadata.EntryPoint, ...]


def plugin_distributions(
    selected: set[tuple[str, str]] | None = None,
) -> tuple[PluginDistribution, ...]:
    """Read installed Jerry/plugin metadata without importing plugin code.

    None selects all Jerry entry points; an empty set selects only Jerry itself.
    Keep competing packages visible: loading an ambiguous entry point fails.
    """
    distributions = []
    seen: set[str] = set()
    for dist in metadata.distributions():
        # Match importlib.metadata.entry_points(): first distribution per name.
        name = re.sub(r"[-_.]+", "-", dist.name).lower()
        if name in seen:
            continue
        seen.add(name)
        entrypoints = tuple(
            sorted(
                (
                    ep
                    for ep in dist.entry_points
                    if ep.group in PLUGIN_GROUPS
                    and (selected is None or (ep.group, ep.name) in selected)
                ),
                key=lambda ep: (ep.group, ep.name, ep.value),
            )
        )
        if name == "jerry-thomas" or entrypoints:
            distributions.append(
                PluginDistribution(
                    dist.name, dist.version, _editable_path(dist), entrypoints
                )
            )
    return tuple(sorted(distributions, key=lambda dist: (dist.name, dist.version)))


def _editable_path(dist: metadata.Distribution) -> Path | None:
    direct_url = dist.read_text("direct_url.json")
    if direct_url:
        try:
            origin = json.loads(direct_url)
            if not isinstance(origin, dict):
                return None
            directory = origin.get("dir_info")
            source_url = origin.get("url")
            if (
                isinstance(directory, dict)
                and directory.get("editable") is True
                and isinstance(source_url, str)
            ):
                url = urlsplit(source_url)
                if url.scheme == "file" and url.netloc in {"", "localhost"}:
                    path = Path(url2pathname(url.path))
                    return path if path.is_absolute() else None
        except ValueError:
            pass
    return None


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
