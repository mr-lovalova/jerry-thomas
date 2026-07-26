from datapipeline.artifacts.registry import ArtifactRegistry, ArtifactSpec


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
