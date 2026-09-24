from dataclasses import replace

import pytest

from jerrythomas.artifacts.registry import (
    ArtifactRegistry,
    ArtifactSpec,
    SCALER_SPEC,
    VECTOR_METADATA_SPEC,
)


def test_registered_metadata_is_a_stable_read_only_mapping(tmp_path) -> None:
    source = {"rows": 1}
    artifacts = ArtifactRegistry(tmp_path)

    artifacts.register("schedule", "schedule.jsonl", source)
    source["rows"] = 2
    metadata = artifacts.require("schedule").meta

    assert metadata == {"rows": 1}
    assert not hasattr(metadata, "__setitem__")


def test_loaded_artifact_is_reused_until_registration_changes(tmp_path) -> None:
    calls = 0

    def load(path):
        nonlocal calls
        calls += 1
        return path.read_text(encoding="utf-8")

    (tmp_path / "first.txt").write_text("first", encoding="utf-8")
    (tmp_path / "second.txt").write_text("second", encoding="utf-8")
    artifacts = ArtifactRegistry(tmp_path)
    spec = ArtifactSpec(key="value", loader=load)

    artifacts.register("value", "first.txt")
    assert artifacts.load(spec) == "first"
    assert artifacts.load(spec) == "first"
    assert calls == 1

    artifacts.register("value", "second.txt")
    assert artifacts.load(spec) == "second"
    assert calls == 2

    artifacts.clear()
    artifacts.register("value", "first.txt")
    assert artifacts.load(spec) == "first"
    assert calls == 3


@pytest.mark.parametrize("core_key", ["metadata", "scaler"])
def test_dataset_specs_do_not_shadow_shared_artifact_names_or_cached_values(
    tmp_path, core_key
) -> None:
    dataset_id = f"dataset.alpha.{core_key}"
    artifacts = ArtifactRegistry(tmp_path, aliases={core_key: dataset_id})
    (tmp_path / "shared.txt").write_text("shared schedule", encoding="utf-8")
    (tmp_path / "dataset.txt").write_text("dataset artifact", encoding="utf-8")
    artifacts.register(core_key, "shared.txt", {"partition_by": ["security_id"]})
    artifacts.register(dataset_id, "dataset.txt")
    shared = ArtifactSpec(key=core_key, loader=lambda path: path.read_text())
    selected = replace(
        SCALER_SPEC if core_key == "scaler" else VECTOR_METADATA_SPEC,
        loader=lambda path: path.read_text(),
    )

    assert artifacts.has(core_key)
    assert artifacts.has(selected)
    assert artifacts.require(core_key).meta == {"partition_by": ["security_id"]}
    assert artifacts.optional(core_key).relative_path == "shared.txt"
    assert artifacts.optional(selected).relative_path == "dataset.txt"
    assert artifacts.resolve_path(core_key) == tmp_path / "shared.txt"
    assert artifacts.resolve_path(selected) == tmp_path / "dataset.txt"
    assert artifacts.load(shared) == "shared schedule"
    assert artifacts.load(selected) == "dataset artifact"
    assert artifacts.load(shared) == "shared schedule"


def test_dataset_alias_does_not_make_an_absent_explicit_artifact_available(tmp_path):
    artifacts = ArtifactRegistry(
        tmp_path, aliases={"metadata": "dataset.alpha.metadata"}
    )
    artifacts.register("dataset.alpha.metadata", "metadata.json")
    selected = ArtifactSpec(
        key="metadata", loader=lambda path: path.read_text(), dataset_scoped=True
    )

    assert artifacts.has(selected)
    assert artifacts.optional(selected) is not None
    assert not artifacts.has("metadata")
    assert artifacts.optional("metadata") is None
