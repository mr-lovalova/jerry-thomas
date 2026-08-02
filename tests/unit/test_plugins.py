import importlib.metadata as metadata

import pytest

from jerrythomas import plugins


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
