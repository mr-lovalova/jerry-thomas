import os
import tomllib
from pathlib import Path

import pytest

from datapipeline.artifacts.planning import (
    ArtifactGraph,
    build_artifact_graph,
)
from datapipeline.artifacts.specs import (
    ARTIFACT_DEFINITIONS,
    SCALER_STATISTICS,
    SERIES,
    VECTOR_METADATA,
    COVERAGE_STATS,
    ArtifactDefinition,
    dataset_requires_scaler,
)
from datapipeline.artifacts.validation import (
    nested_schedule_dependencies,
    stream_schedule_artifacts,
    validate_artifact_plan,
)
from datapipeline.build.state import ArtifactFileFingerprint, BuildState
from datapipeline.config.dataset.dataset import DatasetConfig, SampleConfig
from datapipeline.config.dataset.series import SeriesConfig
from datapipeline.config.preview import PreviewStage
from datapipeline.config.streams import StreamsConfig
from datapipeline.config.tasks.base import ArtifactTask, RuntimeTask
from datapipeline.config.tasks.coverage import CoverageTask
from datapipeline.config.tasks.coverage_stats import CoverageStatsTask
from datapipeline.config.tasks.dataset import DatasetTask
from datapipeline.config.tasks.matrix import MatrixTask
from datapipeline.config.tasks.metadata import MetadataTask
from datapipeline.config.tasks.scaler import ScalerTask
from datapipeline.config.tasks.series import SeriesTask
from datapipeline.config.tasks.schedule import ScheduleTask
from datapipeline.plugins import BUILD_OPERATIONS_EP
from datapipeline.services.definitions import ArtifactHashes


def _current_hashes(graph: ArtifactGraph) -> ArtifactHashes:
    return ArtifactHashes(
        {definition.key: "current" for definition in graph.definitions}
    )


def _declared_entrypoints(group: str) -> dict[str, str]:
    repo_root = Path(__file__).resolve().parents[3]
    data = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    project = data.get("project", {})
    entrypoints = project.get("entry-points", {})
    group_values = entrypoints.get(group, {})
    if not isinstance(group_values, dict):
        return {}
    return {
        str(key): str(value)
        for key, value in group_values.items()
        if isinstance(key, str) and isinstance(value, str)
    }


def test_artifact_graph_uses_dependency_order_not_declaration_order():
    graph = ArtifactGraph(
        (
            ArtifactDefinition(key="result", dependencies=("input",)),
            ArtifactDefinition(key="input"),
        ),
        {},
    )

    assert graph.topological_order({"result", "input"}) == ("input", "result")


def test_artifact_keys_match_task_ids():
    graph = build_artifact_graph(
        [
            MetadataTask(id="metadata"),
            ScalerTask(id="scaler"),
            CoverageStatsTask(id="coverage_stats", stage="assembled"),
            SeriesTask(id="series"),
        ]
    )

    assert graph.declared_artifact_keys() == {
        VECTOR_METADATA,
        SCALER_STATISTICS,
        COVERAGE_STATS,
        SERIES,
    }


@pytest.mark.parametrize("stage", ["assembled", "postprocessed"])
def test_coverage_stats_build_selects_metadata_dependency_chain(stage):
    graph = build_artifact_graph(
        [
            CoverageStatsTask(id="coverage_stats", stage=stage),
            MetadataTask(id="metadata"),
            SeriesTask(id="series"),
            ScalerTask(id="scaler"),
        ]
    )

    keys = set(graph.dependency_closure({"coverage_stats"}))

    assert keys == {
        COVERAGE_STATS,
        VECTOR_METADATA,
        SERIES,
    }


def test_schedule_task_uses_task_id_as_artifact_key():
    graph = build_artifact_graph(
        [
            ScheduleTask(
                id="schedule",
                entrypoint="core.artifact.schedule",
                stream="reference.stream",
                partition_by=[],
                output="build/schedule.jsonl",
            )
        ]
    )

    assert graph.declared_artifact_keys() == {"schedule"}


def test_schedule_artifacts_feed_scaler_and_series() -> None:
    schedule_task = ScheduleTask(
        id="schedule",
        stream="reference.stream",
        partition_by=[],
        output="build/schedule.jsonl",
    )
    dataset = DatasetConfig(
        sample=SampleConfig(cadence="1h"),
        features=[
            SeriesConfig(
                id="price",
                stream="feature.stream",
                field="close",
                scale=True,
            )
        ],
    )
    streams = StreamsConfig.model_validate(
        {
            "streams": {
                "feature.stream": {
                    "id": "feature.stream",
                    "from": {"source": "raw"},
                    "map": {"entrypoint": "identity"},
                    "transforms": [
                        {
                            "operation": "ensure_schedule",
                            "schedule": "schedule",
                        }
                    ],
                }
            }
        }
    )
    graph = build_artifact_graph(
        [
            schedule_task,
            ScalerTask(id="scaler"),
            SeriesTask(id="series"),
        ],
        dataset,
        streams,
    )

    assert graph.definition(SCALER_STATISTICS).dependencies == ("schedule",)
    assert graph.definition(SERIES).dependencies == ("schedule",)
    assert graph.dependents_of({"schedule"}) == {
        SCALER_STATISTICS,
        SERIES,
        VECTOR_METADATA,
        COVERAGE_STATS,
    }


def test_schedule_artifact_rejects_nested_schedule_in_upstream_stream() -> None:
    schedule_task = ScheduleTask(
        id="derived_schedule",
        stream="derived",
        partition_by=[],
        output="build/derived-schedule.jsonl",
    )
    graph = build_artifact_graph([schedule_task])
    streams = StreamsConfig.model_validate(
        {
            "streams": {
                "base": {
                    "id": "base",
                    "from": {"source": "raw"},
                    "map": {"entrypoint": "identity"},
                    "transforms": [
                        {
                            "operation": "ensure_schedule",
                            "schedule": "base_schedule",
                        }
                    ],
                },
                "derived": {
                    "id": "derived",
                    "from": {"align": ["base", "duration"]},
                    "combine": {"entrypoint": "combine"},
                },
                "duration": {
                    "id": "duration",
                    "from": {"source": "raw"},
                    "map": {"entrypoint": "identity"},
                    "transforms": [
                        {
                            "operation": "ensure_cadence",
                            "cadence": "1d",
                        }
                    ],
                },
            }
        }
    )
    assert stream_schedule_artifacts("derived", streams) == {"base_schedule"}

    dependencies = nested_schedule_dependencies(
        streams,
        graph,
        {"derived_schedule"},
    )
    assert len(dependencies) == 1
    assert dependencies[0].task == schedule_task
    assert dependencies[0].task is not schedule_task
    assert dependencies[0].schedule_artifacts == {"base_schedule"}

    with pytest.raises(ValueError, match="Nested schedule artifact dependencies"):
        validate_artifact_plan(streams, graph, {"derived_schedule"})


def test_schedule_artifact_allows_duration_cadence() -> None:
    schedule_task = ScheduleTask(
        id="hourly_schedule",
        stream="hourly",
        partition_by=[],
        output="build/hourly-schedule.jsonl",
    )
    graph = build_artifact_graph([schedule_task])
    streams = StreamsConfig.model_validate(
        {
            "streams": {
                "hourly": {
                    "id": "hourly",
                    "from": {"source": "raw"},
                    "map": {"entrypoint": "identity"},
                    "transforms": [
                        {
                            "operation": "ensure_cadence",
                            "cadence": "1h",
                        }
                    ],
                }
            }
        }
    )
    validate_artifact_plan(streams, graph, {"hourly_schedule"})


def test_inactive_scaler_prunes_its_schedule_dependency() -> None:
    schedule_task = ScheduleTask(
        id="schedule",
        stream="reference.stream",
        partition_by=[],
        output="build/schedule.jsonl",
    )
    graph = build_artifact_graph([schedule_task, ScalerTask(id="scaler")])
    dataset = DatasetConfig(sample=SampleConfig(cadence="1h"), features=[], targets=[])

    assert (
        graph.active_dependency_closure(
            {SCALER_STATISTICS},
            dataset,
        )
        == ()
    )
    assert graph.active_dependency_closure(
        {SCALER_STATISTICS, "schedule"},
        dataset,
    ) == ("schedule",)


def test_generic_artifact_task_is_a_dependency_free_leaf():
    graph = build_artifact_graph(
        [
            ArtifactTask(
                id="custom_snapshot",
                entrypoint="plugin.snapshot",
                output="build/custom.json",
            )
        ]
    )

    assert graph.dependency_closure({"custom_snapshot"}) == ("custom_snapshot",)
    assert graph.definition("custom_snapshot").dependencies == ()


@pytest.mark.parametrize(
    "outputs",
    [
        ("build/shared.json", "build/shared.json"),
        ("build/Shared.json", "build/shared.json"),
    ],
)
def test_artifact_graph_rejects_duplicate_output_paths(outputs):
    with pytest.raises(ValueError, match="write the same output"):
        build_artifact_graph(
            [
                ArtifactTask(
                    id="first",
                    entrypoint="plugin.first",
                    output=outputs[0],
                ),
                ArtifactTask(
                    id="second",
                    entrypoint="plugin.second",
                    output=outputs[1],
                ),
            ]
        )


def test_artifact_graph_rejects_unknown_dependencies():
    with pytest.raises(ValueError, match="unknown dependency 'missing'"):
        ArtifactGraph(
            (
                ArtifactDefinition(
                    key="result",
                    dependencies=("missing",),
                ),
            ),
            {},
        )


def test_artifact_graph_rejects_task_without_matching_definition():
    task = ArtifactTask(
        id="snapshot",
        entrypoint="plugin.snapshot",
        output="snapshot.json",
    )

    with pytest.raises(ValueError, match="has no matching artifact definition"):
        ArtifactGraph((), {task.id: task})


def test_artifact_graph_rejects_task_mapping_key_that_differs_from_id():
    task = ArtifactTask(
        id="snapshot",
        entrypoint="plugin.snapshot",
        output="snapshot.json",
    )

    with pytest.raises(ValueError, match="does not match operation id 'snapshot'"):
        ArtifactGraph((ArtifactDefinition(key="wrong"),), {"wrong": task})


def test_artifact_graph_rejects_cycles_with_path():
    with pytest.raises(ValueError, match="first -> second -> first"):
        ArtifactGraph(
            (
                ArtifactDefinition(
                    key="first",
                    dependencies=("second",),
                ),
                ArtifactDefinition(
                    key="second",
                    dependencies=("first",),
                ),
            ),
            {},
        )


def test_artifact_graph_rejects_unknown_requested_artifact():
    graph = build_artifact_graph([])

    with pytest.raises(ValueError, match="Unknown artifact 'missing'"):
        graph.dependency_closure({"missing"})


def test_stale_dependency_makes_current_dependent_outdated(tmp_path):
    graph = ArtifactGraph(
        (
            ArtifactDefinition(key="input"),
            ArtifactDefinition(
                key="result",
                dependencies=("input",),
            ),
        ),
        {},
    )
    (tmp_path / "input.json").write_text("{}", encoding="utf-8")
    (tmp_path / "result.json").write_text("{}", encoding="utf-8")
    state = BuildState()
    state.register(
        "input",
        artifact_hash="old",
        files=(
            ArtifactFileFingerprint.from_path(
                "input.json",
                tmp_path / "input.json",
            ),
        ),
    )
    state.register(
        "result",
        artifact_hash="current",
        files=(
            ArtifactFileFingerprint.from_path(
                "result.json",
                tmp_path / "result.json",
            ),
        ),
    )

    freshness = graph.freshness(
        keys={"input", "result"},
        state=state,
        artifact_hashes=_current_hashes(graph),
        artifacts_root=tmp_path,
    )

    assert freshness.stale == {"input"}
    assert freshness.outdated == {"input", "result"}


def test_freshness_compares_each_artifacts_semantic_hash(tmp_path) -> None:
    graph = ArtifactGraph(
        (ArtifactDefinition(key="left"), ArtifactDefinition(key="right")),
        {},
    )
    state = BuildState()
    for key in ("left", "right"):
        path = tmp_path / f"{key}.json"
        path.write_text("{}", encoding="utf-8")
        state.register(
            key,
            artifact_hash="same" if key == "left" else "old",
            files=(ArtifactFileFingerprint.from_path(path.name, path),),
        )

    freshness = graph.freshness(
        keys={"left", "right"},
        state=state,
        artifact_hashes=ArtifactHashes({"left": "same", "right": "new"}),
        artifacts_root=tmp_path,
    )

    assert freshness.stale == {"right"}
    assert freshness.outdated == {"right"}


def test_artifact_with_missing_file_is_not_current(tmp_path):
    graph = ArtifactGraph(
        (ArtifactDefinition(key="result"),),
        {},
    )
    state = BuildState()
    state.register(
        "result",
        artifact_hash="current",
        files=(
            ArtifactFileFingerprint(
                relative_path="missing.json",
                size=0,
                mtime_ns=0,
                ctime_ns=0,
            ),
        ),
    )

    freshness = graph.freshness(
        keys={"result"},
        state=state,
        artifact_hashes=_current_hashes(graph),
        artifacts_root=tmp_path,
    )

    assert freshness.missing == {"result"}
    assert freshness.outdated == {"result"}


def test_artifact_at_path_other_than_declared_output_is_stale(tmp_path):
    task = ArtifactTask(
        id="snapshot",
        entrypoint="plugin.snapshot",
        output="declared.json",
    )
    graph = build_artifact_graph([task])
    (tmp_path / "legacy.json").write_text("{}", encoding="utf-8")
    state = BuildState()
    state.register(
        "snapshot",
        artifact_hash="current",
        files=(
            ArtifactFileFingerprint.from_path(
                "legacy.json",
                tmp_path / "legacy.json",
            ),
        ),
    )

    freshness = graph.freshness(
        keys={"snapshot"},
        state=state,
        artifact_hashes=_current_hashes(graph),
        artifacts_root=tmp_path,
    )

    assert freshness.stale == {"snapshot"}
    assert freshness.outdated == {"snapshot"}


@pytest.mark.parametrize("change", ["remove", "alter"])
def test_artifact_companion_changes_affect_freshness(tmp_path, change):
    graph = ArtifactGraph((ArtifactDefinition(key="bundle"),), {})
    primary = tmp_path / "manifest.json"
    companion = tmp_path / "manifest.shards/000000.jsonl.gz"
    primary.write_text("{}", encoding="utf-8")
    companion.parent.mkdir()
    companion.write_bytes(b"original")
    files = (
        ArtifactFileFingerprint.from_path("manifest.json", primary),
        ArtifactFileFingerprint.from_path(
            "manifest.shards/000000.jsonl.gz",
            companion,
        ),
    )
    state = BuildState()
    state.register(
        "bundle",
        artifact_hash="current",
        files=files,
    )

    if change == "remove":
        companion.unlink()
    else:
        companion.write_bytes(b"altered")

    freshness = graph.freshness(
        keys={"bundle"},
        state=state,
        artifact_hashes=_current_hashes(graph),
        artifacts_root=tmp_path,
    )

    expected = freshness.missing if change == "remove" else freshness.stale
    assert expected == {"bundle"}
    assert freshness.outdated == {"bundle"}


def test_same_size_artifact_replacement_with_preserved_mtime_is_stale(tmp_path):
    graph = ArtifactGraph((ArtifactDefinition(key="result"),), {})
    output = tmp_path / "result.json"
    output.write_bytes(b"first")
    fingerprint = ArtifactFileFingerprint.from_path("result.json", output)
    state = BuildState()
    state.register(
        "result",
        artifact_hash="current",
        files=(fingerprint,),
    )

    previous = output.stat()
    output.write_bytes(b"other")
    os.utime(output, ns=(previous.st_atime_ns, previous.st_mtime_ns))

    freshness = graph.freshness(
        keys={"result"},
        state=state,
        artifact_hashes=_current_hashes(graph),
        artifacts_root=tmp_path,
    )

    assert freshness.stale == {"result"}
    assert freshness.outdated == {"result"}


@pytest.mark.parametrize(
    ("preview", "expected"),
    [
        (None, {VECTOR_METADATA}),
        ("input", set()),
        ("canonical", set()),
        ("records", set()),
        ("series", set()),
        ("samples", {VECTOR_METADATA}),
        ("postprocess", {VECTOR_METADATA}),
    ],
)
def test_dataset_runtime_requirements_follow_preview_stage(
    preview,
    expected,
):
    graph = build_artifact_graph([])
    task = DatasetTask(id="dataset")

    assert (
        graph.runtime_requirements(
            task,
            preview=preview,
        )
        == expected
    )


@pytest.mark.parametrize(
    "entrypoint",
    ["core.runtime.dataset", "core.runtime.coverage"],
)
def test_plugin_task_cannot_claim_core_requirements_by_entrypoint(
    entrypoint: str,
) -> None:
    graph = build_artifact_graph([])
    task = RuntimeTask(id="plugin", entrypoint=entrypoint, requires=("declared",))

    assert graph.runtime_requirements(task, preview=None) == {"declared"}


@pytest.mark.parametrize("preview", ["input", "canonical", "records", "series"])
def test_record_and_series_previews_require_declared_schedule(
    preview: PreviewStage,
) -> None:
    schedule_task = ScheduleTask(
        id="schedule",
        stream="reference.stream",
        partition_by=[],
        output="build/schedule.jsonl",
    )
    unused_schedule = ScheduleTask(
        id="unused_schedule",
        stream="unused.stream",
        partition_by=[],
        output="build/unused_schedule.jsonl",
    )
    dataset = DatasetConfig(
        sample=SampleConfig(cadence="1h"),
        features=[SeriesConfig(id="price", stream="feature.stream", field="close")],
    )
    streams = StreamsConfig.model_validate(
        {
            "streams": {
                "feature.stream": {
                    "id": "feature.stream",
                    "from": {"source": "raw"},
                    "map": {"entrypoint": "identity"},
                    "transforms": [
                        {
                            "operation": "ensure_schedule",
                            "schedule": "schedule",
                        }
                    ],
                }
            }
        }
    )
    graph = build_artifact_graph(
        [schedule_task, unused_schedule],
        dataset,
        streams,
    )
    task = DatasetTask(id="dataset")

    assert "schedule" in graph.runtime_requirements(
        task,
        preview=preview,
    )
    assert "unused_schedule" not in graph.runtime_requirements(
        task,
        preview=preview,
    )


def test_invalid_dataset_preview_is_rejected_for_empty_dataset() -> None:
    graph = build_artifact_graph([])
    task = DatasetTask(id="dataset")
    dataset = DatasetConfig(sample=SampleConfig(cadence="1h"))

    with pytest.raises(ValueError, match="preview must be one of"):
        graph.runtime_dependency_closure(
            task,
            preview="unknown",  # type: ignore[arg-type]
            dataset=dataset,
        )


def test_runtime_dependency_closure_uses_coverage_stats_task_stage():
    graph = build_artifact_graph(
        [
            SeriesTask(id="series"),
            MetadataTask(id="metadata"),
            CoverageStatsTask(id="coverage_stats", stage="assembled"),
        ]
    )
    task = CoverageTask(id="coverage")
    dataset = DatasetConfig(sample=SampleConfig(cadence="1h"), features=[], targets=[])

    assert graph.runtime_dependency_closure(
        task,
        preview=None,
        dataset=dataset,
    ) == (SERIES, VECTOR_METADATA, COVERAGE_STATS)


@pytest.mark.parametrize(
    ("stage", "expected"),
    [
        ("assembled", (SERIES, VECTOR_METADATA)),
        ("postprocessed", (SERIES, VECTOR_METADATA)),
    ],
)
def test_matrix_uses_vector_artifacts_without_coverage_stats(stage, expected) -> None:
    graph = build_artifact_graph(
        [
            SeriesTask(id="series"),
            MetadataTask(id="metadata"),
        ]
    )
    task = MatrixTask(id="matrix", options={"stage": stage})

    assert (
        graph.runtime_dependency_closure(
            task,
            preview=None,
            dataset=DatasetConfig(sample=SampleConfig(cadence="1h")),
        )
        == expected
    )


@pytest.mark.parametrize(
    ("scale", "expected"),
    [
        (False, False),
        (True, True),
    ],
)
def test_dataset_scaler_requirement_matches_feature_config(scale, expected):
    dataset = DatasetConfig(
        sample=SampleConfig(cadence="1h"),
        features=[
            SeriesConfig(
                id="price",
                stream="prices",
                field="close",
                scale=scale,
            )
        ],
    )

    assert dataset_requires_scaler(dataset) is expected


def test_scaled_dataset_runtime_requires_scaler_beside_vector_artifacts() -> None:
    graph = build_artifact_graph(
        [
            ScalerTask(),
            SeriesTask(),
            MetadataTask(),
        ]
    )
    dataset = DatasetConfig(
        sample=SampleConfig(cadence="1h"),
        features=[
            SeriesConfig(
                id="price",
                stream="prices",
                field="close",
                scale=True,
            )
        ],
    )

    assert graph.runtime_dependency_closure(
        DatasetTask(id="dataset"),
        preview=None,
        dataset=dataset,
    ) == (
        SCALER_STATISTICS,
        SERIES,
        VECTOR_METADATA,
    )


def test_empty_dataset_has_no_runtime_artifact_requirements():
    graph = build_artifact_graph([])
    task = DatasetTask(id="dataset")
    dataset = DatasetConfig(sample=SampleConfig(cadence="1h"))

    assert (
        graph.runtime_dependency_closure(
            task,
            preview=None,
            dataset=dataset,
        )
        == ()
    )
    assert (
        graph.runtime_dependency_closure(
            task,
            preview="series",
            dataset=dataset,
        )
        == ()
    )


def test_custom_runtime_task_has_no_inferred_artifact_dependencies():
    graph = build_artifact_graph([])
    task = RuntimeTask(id="pipeline", entrypoint="plugin.runtime.pipeline")

    assert graph.runtime_requirements(task, preview=None) == set()


def test_custom_runtime_task_uses_declared_artifact_dependencies():
    snapshot = ArtifactTask(
        id="custom_snapshot",
        entrypoint="plugin.snapshot",
        output="build/custom.json",
    )
    graph = build_artifact_graph([snapshot])
    task = RuntimeTask(
        id="report",
        entrypoint="plugin.runtime.report",
        requires=("custom_snapshot",),
    )

    assert graph.runtime_dependency_closure(
        task,
        preview=None,
        dataset=None,
    ) == ("custom_snapshot",)


def test_empty_dataset_keeps_explicit_artifact_dependencies():
    snapshot = ArtifactTask(
        id="custom_snapshot",
        entrypoint="plugin.snapshot",
        output="build/custom.json",
    )
    graph = build_artifact_graph([snapshot])
    task = DatasetTask(id="dataset", requires=("custom_snapshot",))

    assert graph.runtime_dependency_closure(
        task,
        preview=None,
        dataset=DatasetConfig(sample=SampleConfig(cadence="1h")),
    ) == ("custom_snapshot",)


def test_runtime_task_rejects_unknown_declared_artifact_dependency():
    graph = build_artifact_graph([])
    task = RuntimeTask(
        id="report",
        entrypoint="plugin.runtime.report",
        requires=("missing",),
    )

    with pytest.raises(ValueError, match="Unknown artifact 'missing'"):
        graph.runtime_dependency_closure(
            task,
            preview=None,
            dataset=None,
        )


def test_runtime_task_rejects_inactive_declared_artifact_dependency():
    graph = build_artifact_graph([ScalerTask(id="scaler")])
    task = RuntimeTask(
        id="report",
        entrypoint="plugin.runtime.report",
        requires=("scaler",),
    )

    with pytest.raises(ValueError, match="inactive for this dataset: scaler"):
        graph.runtime_dependency_closure(
            task,
            preview=None,
            dataset=DatasetConfig(sample=SampleConfig(cadence="1h")),
        )


def test_artifact_definitions_have_runner_bound_entrypoints():
    declared = _declared_entrypoints(BUILD_OPERATIONS_EP)
    task_by_id = {
        "metadata": MetadataTask(id="metadata"),
        "scaler": ScalerTask(id="scaler"),
        "coverage_stats": CoverageStatsTask(
            id="coverage_stats",
            stage="postprocessed",
        ),
        "series": SeriesTask(id="series"),
    }
    for definition in ARTIFACT_DEFINITIONS:
        task = task_by_id[definition.key]
        assert task.entrypoint in declared


def test_schedule_entrypoint_is_declared():
    declared = _declared_entrypoints(BUILD_OPERATIONS_EP)
    assert "core.artifact.schedule" in declared
