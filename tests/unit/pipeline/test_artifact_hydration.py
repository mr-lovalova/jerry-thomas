from types import SimpleNamespace

from jerrythomas.artifacts.hydration import (
    hydrate_runtime_artifacts,
    hydrate_runtime_artifacts_for_pipeline,
)
from jerrythomas.artifacts.planning import build_artifact_graph
from jerrythomas.artifacts.specs import (
    SCALER_STATISTICS,
    SERIES,
    VECTOR_METADATA,
)
from jerrythomas.artifacts.validation import NestedScheduleDependency
from jerrythomas.artifacts.state import (
    ArtifactFileFingerprint,
    BuildState,
    save_build_state,
)
from jerrythomas.config.dataset.dataset import DatasetConfig, SampleConfig
from jerrythomas.config.dataset.series import SeriesConfig
from jerrythomas.config.streams import StreamsConfig
from jerrythomas.config.tasks.base import ArtifactTask
from jerrythomas.config.tasks.metadata import MetadataTask
from jerrythomas.config.tasks.scaler import ScalerTask
from jerrythomas.config.tasks.series import SeriesTask
from jerrythomas.config.tasks.schedule import ScheduleTask
from jerrythomas.runtime import Runtime
from jerrythomas.services.definitions import ArtifactHashes
from jerrythomas.services.project_definition import load_project_definition
from jerrythomas.services.runtime_compiler import compile_runtime


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
            SeriesTask(dataset="default", id="series"),
            MetadataTask(dataset="default", id="metadata"),
            custom,
        ],
        {"default": DatasetConfig(sample=SampleConfig(rounding="ceil", cadence="1h"))},
    )
    runtime = Runtime(
        project_yaml=tmp_path / "project.yaml",
        artifacts_root=tmp_path / "artifacts",
        dataset_id="default",
        dataset=graph.datasets["default"],
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
        artifact_hash="current",
        files=(
            ArtifactFileFingerprint(
                relative_path=paths[VECTOR_METADATA],
                size=0,
                mtime_ns=0,
                ctime_ns=0,
                sha256="0" * 64,
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
    metadata = MetadataTask(dataset="default", id="metadata")
    graph = build_artifact_graph(
        [custom, metadata],
        {"default": DatasetConfig(sample=SampleConfig(rounding="ceil", cadence="1h"))},
    )
    runtime = Runtime(
        project_yaml=tmp_path / "project.yaml",
        artifacts_root=tmp_path / "artifacts",
        dataset_id="default",
        dataset=graph.datasets["default"],
    )
    state = BuildState()
    paths = {
        "custom_snapshot": "build/custom.json",
        "dataset.default.series": "build/series.json",
        VECTOR_METADATA: "build/metadata.json",
    }
    for key, relative_path in paths.items():
        destination = runtime.artifacts_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("{}", encoding="utf-8")
        state.register(
            key,
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


def test_project_hydration_excludes_inactive_scaler(
    monkeypatch,
    tmp_path,
) -> None:
    scaler = ScalerTask(
        dataset="default",
    )
    dataset = DatasetConfig(sample=SampleConfig(rounding="ceil", cadence="1h"))
    streams = StreamsConfig()
    graph = build_artifact_graph([scaler], {"default": dataset})
    runtime = Runtime(
        project_yaml=tmp_path / "project.yaml",
        artifacts_root=tmp_path / "artifacts",
        dataset_id="default",
        dataset=graph.datasets["default"],
    )
    output = runtime.artifacts_root / scaler.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("{}", encoding="utf-8")
    state = BuildState()
    state.register(
        SCALER_STATISTICS,
        artifact_hash="current",
        files=(ArtifactFileFingerprint.from_path(scaler.output, output),),
    )
    runtime.artifacts.register(SCALER_STATISTICS, scaler.output)
    monkeypatch.setattr(
        "jerrythomas.artifacts.hydration.load_build_state",
        lambda _state_path: state,
    )
    definition = SimpleNamespace(
        project=SimpleNamespace(artifacts_root=runtime.artifacts_root),
        artifact_graph=graph,
        artifact_hashes=_current_hashes(SCALER_STATISTICS),
        streams=streams,
    )

    assert (
        hydrate_runtime_artifacts_for_pipeline(
            runtime,
            definition,
        )
        == ()
    )
    assert not runtime.artifacts.has(SCALER_STATISTICS)


def test_project_hydration_excludes_nested_schedule_and_dependents(
    monkeypatch,
    tmp_path,
) -> None:
    schedule = ScheduleTask(
        id="derived_schedule",
        stream="derived",
        partition_by=[],
        output="build/derived-schedule.jsonl",
    )
    series = SeriesTask(dataset="default", id="series")
    dataset = DatasetConfig(
        sample=SampleConfig(rounding="ceil", cadence="1h"),
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
                        {
                            "operation": "ensure_schedule",
                            "schedule": "derived_schedule",
                        }
                    ],
                }
            }
        }
    )
    graph = build_artifact_graph([schedule, series], {"default": dataset}, streams)
    runtime = Runtime(
        project_yaml=tmp_path / "project.yaml",
        artifacts_root=tmp_path / "artifacts",
        dataset_id="default",
        dataset=graph.datasets["default"],
    )
    state = BuildState()
    for key, relative_path in (
        ("derived_schedule", schedule.output),
        (SERIES, series.output),
    ):
        destination = runtime.artifacts_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("{}", encoding="utf-8")
        state.register(
            key,
            artifact_hash="current",
            files=(ArtifactFileFingerprint.from_path(relative_path, destination),),
        )
    runtime.artifacts.register("derived_schedule", schedule.output)
    runtime.artifacts.register(SERIES, series.output)
    monkeypatch.setattr(
        "jerrythomas.artifacts.hydration.load_build_state",
        lambda _state_path: state,
    )
    monkeypatch.setattr(
        "jerrythomas.artifacts.hydration.nested_schedule_dependencies",
        lambda *_args: (
            NestedScheduleDependency(
                task=schedule,
                schedule_artifacts=frozenset({"base_schedule"}),
            ),
        ),
    )

    definition = SimpleNamespace(
        project=SimpleNamespace(artifacts_root=runtime.artifacts_root),
        artifact_graph=graph,
        artifact_hashes=_current_hashes("derived_schedule", SERIES),
        streams=streams,
    )
    hydrated = hydrate_runtime_artifacts_for_pipeline(
        runtime,
        definition,
    )

    assert hydrated == ()
    assert not runtime.artifacts.has("derived_schedule")
    assert not runtime.artifacts.has(SERIES)


def test_project_hydration_uses_semantic_artifact_hash(tmp_path) -> None:
    project_path = tmp_path / "project.yaml"
    project_path.write_text(
        "\n".join(
            [
                "schema_version: 7",
                "artifact_revision: 1",
                "paths:",
                "  streams: ./streams",
                "  sources: ./sources",
                "  datasets: ./datasets",
                "  artifacts: ./artifacts",
                "  operations: ./operations",
                "  profiles: ./profiles",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "datasets").mkdir()
    (tmp_path / "datasets/default.yaml").write_text(
        "version: v1\nsample:\n  rounding: ceil\n  cadence: 1h\n", encoding="utf-8"
    )
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
                "path: build/custom.json",
            ]
        ),
        encoding="utf-8",
    )
    definition = load_project_definition(project_path)
    runtime = compile_runtime(definition, dataset_id="default")
    output = runtime.artifacts_root / "build/custom.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("{}", encoding="utf-8")
    state = BuildState()
    state.register(
        "custom_snapshot",
        artifact_hash=definition.artifact_hashes.for_artifact("custom_snapshot"),
        files=(ArtifactFileFingerprint.from_path("build/custom.json", output),),
    )
    save_build_state(state, runtime.artifacts_root)

    hydrate_runtime_artifacts_for_pipeline(runtime, definition)
    assert runtime.artifacts.has("custom_snapshot")

    task_path.write_text(task_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    whitespace_definition = load_project_definition(project_path)
    whitespace_runtime = compile_runtime(whitespace_definition, dataset_id="default")
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
    changed_runtime = compile_runtime(changed_definition, dataset_id="default")
    assert changed_definition.artifact_hashes != definition.artifact_hashes
    assert (
        hydrate_runtime_artifacts_for_pipeline(
            changed_runtime,
            changed_definition,
        )
        == ()
    )
    assert not changed_runtime.artifacts.has("custom_snapshot")
