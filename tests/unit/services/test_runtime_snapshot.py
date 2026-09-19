import multiprocessing
import pickle
from concurrent.futures import ProcessPoolExecutor

import pytest

from jerrythomas.artifacts.registry import ArtifactRegistry, ArtifactSpec
from jerrythomas.config.execution import ExecutionConfig
from jerrythomas.config.sources import EntryPointConfig
from jerrythomas.pipelines.stream.pipeline import run_stream_pipeline
from jerrythomas.runtime import Runtime
from jerrythomas.services.project_definition import load_project_definition
from jerrythomas.services.runtime_compiler import (
    compile_runtime,
    compile_runtime_snapshot,
    snapshot_runtime,
)
from jerrythomas.sources.loader import BaseDataLoader


class MutatingValuesLoader(BaseDataLoader):
    def __init__(self, settings):
        settings["values"].append(2)
        self.values = settings["values"]

    def load(self):
        for index, value in enumerate(self.values):
            yield {
                "time": f"2024-01-01T{index:02}:00:00Z",
                "ticker": "A",
                "value": value,
            }


@pytest.fixture
def compiled_runtime(tmp_path):
    (tmp_path / "sources").mkdir()
    (tmp_path / "streams").mkdir()
    (tmp_path / "project.yaml").write_text(
        "schema_version: 6\nartifact_revision: 1\nname: snapshot-test\n"
        "paths:\n  sources: sources\n  streams: streams\n"
        "  dataset: dataset.yaml\n  artifacts: artifacts\n"
        "globals:\n  data_file: prices.jsonl\n"
    )
    (tmp_path / "dataset.yaml").write_text(
        "sample: {cadence: 1h, rounding: exact, keys: [ticker]}\n"
        "features: [{id: value, stream: selected, field: value}]\n"
    )
    (tmp_path / "prices.jsonl").write_text(
        '{"time":"2024-01-01T01:00:00Z","ticker":"A","value":2}\n'
        '{"time":"2024-01-01T00:00:00Z","ticker":"A","value":1}\n'
        '{"time":"2024-01-01T02:00:00Z","ticker":"A","value":3}\n'
    )
    for source_id in ("prices", "unused"):
        (tmp_path / "sources" / f"{source_id}.yaml").write_text(
            f"id: {source_id}\nparser: {{entrypoint: core.temporal_record}}\n"
            "loader:\n  transport: fs\n  path: ${data_file}\n"
            "  reader: {format: jsonl}\n"
        )
        (tmp_path / "streams" / f"{source_id}.yaml").write_text(
            f"id: {source_id}\nfrom: {{source: {source_id}}}\n"
            "map: {entrypoint: identity}\npartition_by: [ticker]\n"
        )
    (tmp_path / "streams" / "selected.yaml").write_text(
        "id: selected\nfrom: {stream: prices}\n"
        "transforms:\n"
        "- {operation: where, field: value, operator: in, comparand: [1, 2]}\n"
    )
    return compile_runtime(load_project_definition(tmp_path / "project.yaml"))


def _snapshot_values(snapshot):
    runtime = compile_runtime_snapshot(snapshot)
    return [record.value for record in run_stream_pipeline(runtime, "selected")]


def test_snapshot_roundtrip_rebuilds_only_selected_dependencies(compiled_runtime):
    runtime = compiled_runtime
    runtime.execution = ExecutionConfig(workers=3, sort_buffer_mb=7)
    runtime.observe_node_events = False
    runtime.heartbeat_interval_seconds = 2.5
    runtime.artifacts = ArtifactRegistry(runtime.artifacts_root / "registered")
    runtime.artifacts.register("ready", "value.json", {"fields": ["value"]})
    # Loaded artifacts may contain objects that cannot be sent to a process.
    runtime.artifacts.load(ArtifactSpec("ready", lambda _: lambda: None))

    snapshot = pickle.loads(pickle.dumps(snapshot_runtime(runtime, ["selected"])))
    assert set(snapshot.streams.streams) == {"prices", "selected"}
    assert set(snapshot.streams.sources) == {"prices"}
    assert snapshot.streams.sources["prices"].loader.path == "prices.jsonl"

    first = compile_runtime_snapshot(snapshot)
    second = compile_runtime_snapshot(snapshot)
    assert first.execution == ExecutionConfig(workers=1, sort_buffer_mb=7)
    assert first.dataset == runtime.dataset
    assert first.observe_node_events is False
    assert first.heartbeat_interval_seconds == 2.5
    assert first.artifacts.root == runtime.artifacts.root
    assert first.artifacts.resolve_path("ready") == runtime.artifacts.resolve_path(
        "ready"
    )
    assert first.artifacts.require("ready").meta == {"fields": ["value"]}
    assert (
        first.streams["prices"].source.loader
        is not runtime.streams["prices"].source.loader
    )
    assert (
        first.streams["prices"].source.loader
        is not second.streams["prices"].source.loader
    )
    assert (
        first.streams["prices"].source.parser
        is not second.streams["prices"].source.parser
    )
    assert _snapshot_values(snapshot) == [1, 2]


def test_snapshot_isolates_nested_configuration_and_artifact_metadata(compiled_runtime):
    runtime = compiled_runtime
    runtime.artifacts.register("ready", "value.json", {"fields": ["value"]})
    snapshot = snapshot_runtime(runtime, ["selected"])
    first = compile_runtime_snapshot(snapshot)
    first.streams["selected"].transforms[0].comparand.append(3)
    first.dataset.features.clear()
    first.artifacts.require("ready").meta["fields"].append("changed")

    assert [record.value for record in run_stream_pipeline(first, "selected")] == [
        1,
        2,
        3,
    ]
    assert _snapshot_values(snapshot) == [1, 2]
    assert snapshot.dataset.features == runtime.dataset.features
    assert snapshot.artifact_registrations["ready"].meta == {"fields": ["value"]}
    assert runtime.artifacts.require("ready").meta == {"fields": ["value"]}
    snapshot.streams.streams["selected"].transforms[0].comparand.append(3)
    assert _snapshot_values(snapshot_runtime(runtime, ["selected"])) == [1, 2]


def test_snapshot_runs_in_spawn_child_without_project_files(
    compiled_runtime, monkeypatch
):
    snapshot = snapshot_runtime(compiled_runtime, ["selected"])
    for path in compiled_runtime.project_yaml.parent.rglob("*.yaml"):
        path.unlink()
    monkeypatch.setenv("data_file", "missing.jsonl")

    with ProcessPoolExecutor(
        max_workers=1, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        assert executor.submit(_snapshot_values, snapshot).result(timeout=30) == [1, 2]


def test_snapshot_preserves_plugin_arguments_before_loader_mutation(
    compiled_runtime, tmp_path, monkeypatch
):
    distribution = tmp_path / "snapshot_test-1.0.dist-info"
    distribution.mkdir()
    (distribution / "METADATA").write_text("Name: snapshot-test\nVersion: 1.0\n")
    (distribution / "entry_points.txt").write_text(
        "[jerrythomas.loaders]\n"
        "snapshot.mutable-values = "
        "tests.unit.services.test_runtime_snapshot:MutatingValuesLoader\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    definition = load_project_definition(compiled_runtime.project_yaml)
    definition.streams.sources["prices"].loader = EntryPointConfig(
        entrypoint="snapshot.mutable-values", args={"settings": {"values": [1]}}
    )
    runtime = compile_runtime(definition)
    snapshot = pickle.loads(pickle.dumps(snapshot_runtime(runtime, ["selected"])))

    assert snapshot.streams.sources["prices"].loader.args == {
        "settings": {"values": [1]}
    }
    assert _snapshot_values(snapshot) == [1, 2]
    with ProcessPoolExecutor(
        max_workers=1, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        assert executor.submit(_snapshot_values, snapshot).result(timeout=30) == [1, 2]


def test_snapshot_rejects_runtime_without_resolved_configuration(compiled_runtime):
    runtime = Runtime(
        project_yaml=compiled_runtime.project_yaml,
        artifacts_root=compiled_runtime.artifacts_root,
        dataset=compiled_runtime.dataset,
    )
    with pytest.raises(ValueError, match=r"compile_runtime.*execution.workers=1"):
        snapshot_runtime(runtime, ["selected"])
