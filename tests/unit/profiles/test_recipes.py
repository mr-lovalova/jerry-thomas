from importlib.metadata import EntryPoint
from types import SimpleNamespace

import pytest

from jerrythomas import plugins
from jerrythomas.profiles import recipes


def test_implementation_identifies_plugins_by_group_and_ignores_argument_payloads(
    tmp_path,
    monkeypatch,
):
    def distribution(name, group, entrypoint):
        return SimpleNamespace(
            name=name,
            version="1.0",
            read_text=lambda _: None,
            entry_points=[EntryPoint(name=entrypoint, value="plugin:run", group=group)],
        )

    monkeypatch.setattr(recipes, "_git_identity", lambda _: None)
    monkeypatch.setattr(
        plugins.metadata,
        "distributions",
        lambda: [
            distribution("used-parser", "jerrythomas.parsers", "shared-name"),
            distribution("unused-loader", "jerrythomas.loaders", "shared-name"),
            distribution("unused-payload", "jerrythomas.mappers", "payload-name"),
        ],
    )
    result = recipes._implementation(
        tmp_path,
        {
            "sources": {
                "source": {
                    "parser": {
                        "entrypoint": "shared-name",
                        "args": {"entrypoint": "payload-name"},
                    },
                    "loader": {"transport": "fs"},
                }
            },
            "streams": {
                "stream": {"map": {"entrypoint": "shared-name"}, "transforms": []}
            },
            "artifacts": {},
            "jobs": [],
        },
    )
    assert result["packages"] == {"used-parser": "1.0"}
    assert result["entrypoints"] == [
        {
            "group": "jerrythomas.parsers",
            "name": "shared-name",
            "value": "plugin:run",
            "distribution": "used-parser",
        }
    ]
    assert result["unidentified_entrypoints"] == [
        {
            "group": "jerrythomas.mappers",
            "name": "shared-name",
        }
    ]


@pytest.mark.parametrize(
    ("entrypoint", "target"),
    [
        (
            "core.records",
            "jerrythomas.operations.runtime.records:run_records_operation",
        ),
        (
            "core.dataset",
            "jerrythomas.operations.runtime.dataset:run_dataset_operation",
        ),
        (
            "core.availability_matrix",
            "jerrythomas.operations.runtime.matrix:run_matrix_operation",
        ),
        (
            "core.coverage_report",
            "jerrythomas.operations.runtime.coverage:run_coverage_operation",
        ),
    ],
)
def test_recipe_identifies_loadable_builtin_output_registrations(
    tmp_path, monkeypatch, entrypoint, target
):
    monkeypatch.setattr(recipes, "_git_identity", lambda _: None)
    result = recipes._implementation(
        tmp_path,
        {
            "sources": {},
            "streams": {},
            "artifacts": {},
            "jobs": [{"operation": {"kind": "output", "entrypoint": entrypoint}}],
        },
    )
    assert result["unidentified_entrypoints"] == []
    assert result["entrypoints"] == [
        {
            "group": plugins.RUNTIME_OPERATIONS_EP,
            "name": entrypoint,
            "value": target,
            "distribution": "jerry-thomas",
        }
    ]
    runner = plugins.load_entrypoint(plugins.RUNTIME_OPERATIONS_EP, entrypoint)
    assert f"{runner.__module__}:{runner.__name__}" == target
