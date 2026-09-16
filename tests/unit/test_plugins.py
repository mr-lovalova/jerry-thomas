import importlib.metadata as metadata
import json
from types import SimpleNamespace

import pytest

from jerrythomas import plugins


def _distribution(name, group=plugins.PARSERS_EP, direct_url=None):
    return SimpleNamespace(
        name=name,
        version="2.3.0",
        entry_points=[
            metadata.EntryPoint(name="csv", value=f"{name}:parse", group=group)
        ],
        read_text=lambda _: direct_url,
    )


def test_plugin_distributions_filters_without_loading_and_keeps_duplicate_providers(
    monkeypatch, tmp_path
):
    editable = tmp_path / "plugin source"
    monkeypatch.setattr(
        metadata,
        "distributions",
        lambda: [
            _distribution("provider-b"),
            _distribution("unrelated", group="other.parsers"),
            _distribution("loader", group=plugins.LOADERS_EP),
            _distribution(
                "provider-a",
                direct_url=json.dumps(
                    {"url": editable.as_uri(), "dir_info": {"editable": True}}
                ),
            ),
            _distribution("jerry-thomas", group=plugins.MAPPERS_EP),
        ],
    )

    def unexpected_load(self):
        pytest.fail("Metadata inspection must not load plugin code")

    monkeypatch.setattr(metadata.EntryPoint, "load", unexpected_load)
    selected = plugins.plugin_distributions({(plugins.PARSERS_EP, "csv")})
    assert [dist.name for dist in selected] == [
        "jerry-thomas",
        "provider-a",
        "provider-b",
    ]
    assert selected[0].entrypoints == ()
    assert selected[1].version == "2.3.0"
    assert selected[1].editable_path == editable
    assert selected[2].editable_path is None
    assert [dist.name for dist in plugins.plugin_distributions()] == [
        "jerry-thomas",
        "loader",
        "provider-a",
        "provider-b",
    ]
    assert [dist.name for dist in plugins.plugin_distributions(set())] == [
        "jerry-thomas"
    ]


@pytest.mark.parametrize(
    "direct_url",
    [
        "invalid json",
        "[]",
        '{"url": "file:///source", "dir_info": null}',
        '{"url": "file:///source", "dir_info": {"editable": false}}',
        '{"url": "https://example.org/source", "dir_info": {"editable": true}}',
        '{"url": 123, "dir_info": {"editable": true}}',
        '{"url": "file:relative", "dir_info": {"editable": true}}',
        '{"url": "file://", "dir_info": {"editable": true}}',
    ],
)
def test_plugin_distributions_omit_unavailable_editable_path(monkeypatch, direct_url):
    monkeypatch.setattr(
        metadata,
        "distributions",
        lambda: [_distribution("provider", direct_url=direct_url)],
    )
    assert plugins.plugin_distributions()[0].editable_path is None


def test_plugin_distributions_match_entrypoint_precedence_for_same_package(monkeypatch):
    installed = _distribution("my-plugin")
    shadowed = _distribution("My_Plugin")
    shadowed.version = "3.0.0"
    monkeypatch.setattr(metadata, "distributions", lambda: [installed, shadowed])

    providers = plugins.plugin_distributions()

    assert len(providers) == 1
    assert providers[0].name == "my-plugin"
    assert providers[0].version == "2.3.0"
    assert providers[0].entrypoints == tuple(installed.entry_points)


def test_load_entrypoint_rejects_noncallable_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entrypoints = metadata.EntryPoints(
        [
            metadata.EntryPoint(
                name="invalid",
                value="builtins:Ellipsis",
                group="test.group",
            )
        ]
    )
    monkeypatch.setattr(metadata, "entry_points", lambda: entrypoints)
    plugins.load_entrypoint.cache_clear()

    with pytest.raises(TypeError, match="Entry point 'invalid'.*must be callable"):
        plugins.load_entrypoint("test.group", "invalid")

    plugins.load_entrypoint.cache_clear()
