from importlib.metadata import EntryPoint
from types import SimpleNamespace

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
        recipes.metadata,
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
