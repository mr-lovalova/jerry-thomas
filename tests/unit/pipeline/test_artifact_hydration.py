from types import SimpleNamespace

from datapipeline.artifacts.hydration import (
    hydrate_runtime_artifacts,
    hydrate_runtime_artifacts_for_pipeline,
)
from datapipeline.artifacts.planning import build_artifact_graph
from datapipeline.artifacts.specs import (
    SERIES,
    VECTOR_METADATA,
)
from datapipeline.artifacts.validation import NestedTickDependency
from datapipeline.build.state import (
    ArtifactFileFingerprint,
    BuildState,
    save_build_state,
)
from datapipeline.config.dataset.dataset import DatasetConfig, SampleConfig
from datapipeline.config.dataset.series import SeriesConfig
from datapipeline.config.streams import StreamsConfig
from datapipeline.config.tasks.base import ArtifactTask
from datapipeline.config.tasks.metadata import MetadataTask
from datapipeline.config.tasks.series import SeriesTask
from datapipeline.config.tasks.ticks import TicksTask
from datapipeline.runtime import Runtime
from datapipeline.services.definitions import ArtifactHashes
from datapipeline.services.project_definition import load_project_definition
from datapipeline.services.runtime_compiler import compile_runtime


def _current_hashes(*keys: str) -> ArtifactHashes:
    return ArtifactHashes({key: "current" for key in keys})


def test_hydration_replaces_registry_with_dependency_current_artifacts(
    tmp_path,
) -> None:
    custom = ArtifactTask(
        id="custom_snapshot",
        entrypoint="plugin.snapshot",
        output="build/custom.json",
    )
    graph = build_artifact_graph(
        [
            SeriesTask(id="series"),
            MetadataTask(id="metadata"),
            custom,
        ]
    )
    runtime = Runtime(
        project_yaml=tmp_path / "project.yaml",
        artifacts_root=tmp_path / "artifacts",
        dataset=DatasetConfig(sample=SampleConfig(cadence="1h")),
    )
    state = BuildState()
    paths = {
        SERIES: "build/series.json",
        VECTOR_METADATA: "build/missing-metadata.json",
        "custom_snapshot": "build/custom.json",
    }
    for relative_path in (
        paths[SERIES],
        paths["custom_snapshot"],
    ):
        destination = runtime.artifacts_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("{}", encoding="utf-8")

    for key in (SERIES, "custom_snapshot"):
        relative_path = paths[key]
        state.register(
            key,
            relative_path,
            artifact_hash="current",
            files=(
                ArtifactFileFingerprint.from_path(
                    relative_path,
                    runtime.artifacts_root / relative_path,
                ),
            ),
        )
    state.register(
        VECTOR_METADATA,
        paths[VECTOR_METADATA],
        artifact_hash="current",
        files=(
            ArtifactFileFingerprint(
                relative_path=paths[VECTOR_METADATA],
                size=0,
                mtime_ns=0,
                ctime_ns=0,
            ),
        ),
    )
    state.artifacts[SERIES].artifact_hash = "old"

    for key, relative_path in paths.items():
        runtime.artifacts.register(key, relative_path)
    runtime.artifacts.register("orphan", "build/orphan.json")

    hydrated = hydrate_runtime_artifacts(
        runtime=runtime,
        graph=graph,
        state=state,
        artifact_hashes=_current_hashes(*paths),
        artifact_keys=graph.dependency_closure(paths),
    )

    assert hydrated == ("custom_snapshot",)
    assert runtime.artifacts.has("custom_snapshot")
    assert not runtime.artifacts.has(SERIES)
    assert not runtime.artifacts.has(VECTOR_METADATA)
    assert not runtime.artifacts.has("orphan")


def test_hydration_skips_incomplete_unrelated_artifact_chain(tmp_path) -> None:
    custom = ArtifactTask(
        id="custom_snapshot",
        entrypoint="plugin.snapshot",
        output="build/custom.json",
    )
    metadata = MetadataTask(id="metadata")
    graph = build_artifact_graph([custom, metadata])
    runtime = Runtime(
        project_yaml=tmp_path / "project.yaml",
        artifacts_root=tmp_path / "artifacts",
        dataset=DatasetConfig(sample=SampleConfig(cadence="1h")),
    )
    state = BuildState()
    paths = {
        "custom_snapshot": "build/custom.json",
        SERIES: "build/series.json",
        VECTOR_METADATA: "build/metadata.json",
    }
    for key, relative_path in paths.items():
        destination = runtime.artifacts_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("{}", encoding="utf-8")
        state.register(
            key,
            relative_path,
            artifact_hash="current",
            files=(ArtifactFileFingerprint.from_path(relative_path, destination),),
        )

    hydrated = hydrate_runtime_artifacts(
        runtime=runtime,
        graph=graph,
        state=state,
        artifact_hashes=_current_hashes(*paths),
        artifact_keys=graph.dependency_closure(paths),
    )

    assert hydrated == ("custom_snapshot",)
    assert runtime.artifacts.has("custom_snapshot")
    assert not runtime.artifacts.has(SERIES)
    assert not runtime.artifacts.has(VECTOR_METADATA)


def test_project_hydration_excludes_nested_tick_and_dependents(
    monkeypatch,
    tmp_path,
) -> None:
    tick = TicksTask(
        id="derived_ticks",
        stream="derived",
        output="build/derived-ticks.jsonl",
    )
    series = SeriesTask(id="series")
    dataset = DatasetConfig(
        sample=SampleConfig(cadence="1h"),
        features=[SeriesConfig(id="price", stream="feature", field="close")],
    )
    streams = StreamsConfig.model_validate(
        {
            "streams": {
                "feature": {
                    "id": "feature",
                    "from": {"source": "raw"},
                    "map": {"entrypoint": "identity"},
                    "transforms": [
                        {"operation": "ensure_ticks", "artifact": "derived_ticks"}
                    ],
                }
            }
        }
    )
    graph = build_artifact_graph([tick, series], dataset, streams)
    runtime = Runtime(
        project_yaml=tmp_path / "project.yaml",
        artifacts_root=tmp_path / "artifacts",
        dataset=dataset,
    )
    state = BuildState()
    for key, relative_path in (
        ("derived_ticks", tick.output),
        (SERIES, series.output),
    ):
        destination = runtime.artifacts_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("{}", encoding="utf-8")
        state.register(
            key,
            relative_path,
            artifact_hash="current",
            files=(ArtifactFileFingerprint.from_path(relative_path, destination),),
        )
    runtime.artifacts.register("derived_ticks", tick.output)
    runtime.artifacts.register(SERIES, series.output)
    monkeypatch.setattr(
        "datapipeline.artifacts.hydration.load_build_state",
        lambda _state_path: state,
    )
    monkeypatch.setattr(
        "datapipeline.artifacts.hydration.nested_tick_dependencies",
        lambda *_args: (
            NestedTickDependency(
                task=tick,
                tick_artifacts=frozenset({"base_ticks"}),
            ),
        ),
    )

    definition = SimpleNamespace(
        project=SimpleNamespace(artifacts_root=runtime.artifacts_root),
        artifact_operations=(),
        artifact_hashes=_current_hashes("derived_ticks", SERIES),
        dataset=runtime.dataset,
        streams=streams,
    )
    hydrated = hydrate_runtime_artifacts_for_pipeline(
        runtime,
        definition,
        graph=graph,
    )

    assert hydrated == ()
    assert not runtime.artifacts.has("derived_ticks")
    assert not runtime.artifacts.has(SERIES)


def test_project_hydration_uses_semantic_artifact_hash(tmp_path) -> None:
    project_path = tmp_path / "project.yaml"
    project_path.write_text(
        "\n".join(
            [
                "schema_version: 4",
                "artifact_revision: 1",
                "paths:",
                "  streams: ./streams",
                "  sources: ./sources",
                "  dataset: ./dataset.yaml",
                "  artifacts: ./artifacts",
                "  operations: ./operations",
                "  profiles: ./profiles",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "dataset.yaml").write_text("sample:\n  cadence: 1h\n", encoding="utf-8")
    for directory in ("streams", "sources"):
        (tmp_path / directory).mkdir()
    operations = tmp_path / "operations"
    operations.mkdir(parents=True)
    task_path = operations / "custom_snapshot.yaml"
    task_path.write_text(
        "\n".join(
            [
                "kind: artifact",
                "entrypoint: plugin.snapshot",
                "output: build/custom.json",
            ]
        ),
        encoding="utf-8",
    )
    definition = load_project_definition(project_path)
    runtime = compile_runtime(definition)
    output = runtime.artifacts_root / "build/custom.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("{}", encoding="utf-8")
    state = BuildState()
    state.register(
        "custom_snapshot",
        "build/custom.json",
        artifact_hash=definition.artifact_hashes.for_artifact("custom_snapshot"),
        files=(ArtifactFileFingerprint.from_path("build/custom.json", output),),
    )
    save_build_state(state, runtime.artifacts_root)

    hydrate_runtime_artifacts_for_pipeline(runtime, definition)
    assert runtime.artifacts.has("custom_snapshot")

    task_path.write_text(task_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    whitespace_definition = load_project_definition(project_path)
    whitespace_runtime = compile_runtime(whitespace_definition)
    assert whitespace_definition.artifact_hashes == definition.artifact_hashes
    assert hydrate_runtime_artifacts_for_pipeline(
        whitespace_runtime,
        whitespace_definition,
    ) == ("custom_snapshot",)
    assert whitespace_runtime.artifacts.has("custom_snapshot")

    task_path.write_text(
        task_path.read_text(encoding="utf-8").replace(
            "entrypoint: plugin.snapshot",
            "entrypoint: plugin.snapshot_v2",
        ),
        encoding="utf-8",
    )

    changed_definition = load_project_definition(project_path)
    changed_runtime = compile_runtime(changed_definition)
    assert changed_definition.artifact_hashes != definition.artifact_hashes
    assert (
        hydrate_runtime_artifacts_for_pipeline(
            changed_runtime,
            changed_definition,
        )
        == ()
    )
    assert not changed_runtime.artifacts.has("custom_snapshot")
