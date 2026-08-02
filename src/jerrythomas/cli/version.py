import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import jerrythomas


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
    return "\n".join(
        [
            f"{DIST_NAME}: {installed_version()}",
            f"jerrythomas: {Path(jerrythomas.__file__).resolve()}",
            f"python: {sys.executable}",
            f"python-prefix: {sys.prefix}",
            f"base-python: {Path(base_executable).resolve()}",
            f"python-version: {platform.python_version()}",
        ]
    )
