import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import jerrythomas
from jerrythomas.plugins import plugin_distributions


DIST_NAME = "jerry-thomas"


def installed_version() -> str:
    try:
        return version(DIST_NAME)
    except PackageNotFoundError:
        return "unknown"


def short_version() -> str:
    return f"{DIST_NAME} {installed_version()}"


def version_report() -> str:
    base_executable = getattr(sys, "_base_executable", sys.executable)
    lines = [
        f"{DIST_NAME}: {installed_version()}",
        f"jerrythomas: {Path(jerrythomas.__file__).resolve()}",
        f"python: {sys.executable}",
        f"python-prefix: {sys.prefix}",
        f"base-python: {Path(base_executable).resolve()}",
        f"python-version: {platform.python_version()}",
    ]
    lines.append("installed plugin providers:")
    for dist in plugin_distributions():
        lines.append(f"  {dist.name} {dist.version}")
        if dist.editable_path is not None:
            lines.append(f"    editable: {dist.editable_path}")
        for ep in dist.entrypoints:
            lines.append(f"    {ep.group}/{ep.name} -> {ep.value}")
    return "\n".join(lines)
