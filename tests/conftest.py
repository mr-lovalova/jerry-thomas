from pathlib import Path
import shutil
from functools import lru_cache
import importlib

import pytest

import jerrythomas.plugins as dp_plugins

# Test-only entrypoint declarations.
_TEST_EP_TARGETS = {
    (
        "jerrythomas.parsers",
        "core.temporal.csv",
    ): "tests.parsers.temporal_csv:TemporalCsvValueParser",
    (
        "jerrythomas.combiners",
        "combine_valuation_inputs",
    ): "tests.combiners:combine_valuation_inputs",
    (
        "jerrythomas.combiners",
        "combine_humidity_with_baseline",
    ): "tests.combiners:combine_humidity_with_baseline",
}
_ORIGINAL_LOAD_ENTRYPOINT = dp_plugins.load_entrypoint


@lru_cache(maxsize=None)
def _load_entrypoint_with_test_targets(group: str, name: str):
    target = _TEST_EP_TARGETS.get((group, name))
    if target:
        module, attr = target.split(":")
        return getattr(importlib.import_module(module), attr)
    return _ORIGINAL_LOAD_ENTRYPOINT(group, name)


dp_plugins.load_entrypoint = _load_entrypoint_with_test_targets


@pytest.fixture
def copy_fixture(tmp_path: Path):
    """Return a helper that copies a fixture directory into a temp location."""

    def _copy(name: str) -> Path:
        src = Path(__file__).parent / "fixtures" / name
        dest = tmp_path / name
        shutil.copytree(
            src,
            dest,
            ignore=shutil.ignore_patterns(
                "build",
                "__pycache__",
                "*.pyc",
                "*.pyo",
                ".DS_Store",
            ),
        )
        return dest

    return _copy
