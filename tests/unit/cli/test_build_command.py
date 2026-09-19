import json
import logging
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from jerrythomas.artifacts import executor as build_exec
from jerrythomas.artifacts.errors import ArtifactResolutionError
from jerrythomas.artifacts.planning import build_artifact_graph
from jerrythomas.artifacts.output import ArtifactOutput
from jerrythomas.artifacts.registry import ArtifactRegistry
from jerrythomas.artifacts.settings import BuildSettings
from jerrythomas.artifacts.specs import (
    SCALER_STATISTICS,
    SERIES,
    VECTOR_METADATA,
    COVERAGE_STATS,
)
from jerrythomas.artifacts.state import (
    ArtifactFileFingerprint,
    BuildState,
    load_build_state,
    save_build_state,
)
from jerrythomas.config.dataset.dataset import DatasetConfig, SampleConfig
from jerrythomas.config.dataset.series import SeriesConfig
from jerrythomas.config.execution import ExecutionConfig
from jerrythomas.config.tasks.base import ArtifactTask
from jerrythomas.config.tasks.coverage_stats import CoverageStatsTask
from jerrythomas.config.tasks.metadata import MetadataTask
from jerrythomas.config.tasks.scaler import ScalerTask
from jerrythomas.config.tasks.series import SeriesTask
from jerrythomas.execution.settings import (
    DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
    LogLevelDecision,
    LogOutputSettings,
    LogOutputTarget,
    ObservabilitySettings,
)
from jerrythomas.services.definitions import ArtifactHashes, ProjectDefinition
from jerrythomas.services.project_definition import load_project_definition


def _dataset_with_feature(*, scale: bool) -> DatasetConfig:
    return DatasetConfig(
        sample=SampleConfig(rounding="ceil", cadence="1h"),
        features=[
            SeriesConfig(
                id="x",
                stream="stream",
                field="value",
                scale=scale,
            )
        ],
    )


def _runtime(artifacts_root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        artifacts_root=artifacts_root,
        artifacts=ArtifactRegistry(artifacts_root),
        observe_node_events=True,
        heartbeat_interval_seconds=None,
        execution=ExecutionConfig(),
    )


def _write_project(tmp_path: Path) -> Path:
    for name in ("streams", "sources", "operations", "profiles"):
        (tmp_path / name).mkdir(parents=True, exist_ok=True)
    (tmp_path / "dataset.yaml").write_text(
        "sample:\n  rounding: ceil\n  cadence: 1h\nfeatures: []\ntargets: []\n",
        encoding="utf-8",
    )
    project_path = tmp_path / "project.yaml"
    project_path.write_text(
        "\n".join(
            [
                "schema_version: 6",
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
    return project_path


def _definition(
    tmp_path: Path,
    dataset: DatasetConfig | None = None,
) -> ProjectDefinition:
    definition = load_project_definition(_write_project(tmp_path))
    if dataset is None:
        return definition
    return replace(definition, dataset=dataset)


def _definition_with_local_source(
    tmp_path: Path,
) -> tuple[ProjectDefinition, Path]:
    project = _write_project(tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    source_path = data / "a.jsonl"
    source_path.write_text("{}\n", encoding="utf-8")
    (tmp_path / "sources" / "prices.yaml").write_text(
        "id: prices\n"
        "parser: {entrypoint: identity}\n"
        "loader: {transport: fs, path: 'data/*.jsonl', reader: {format: jsonl}}\n",
        encoding="utf-8",
    )
    (tmp_path / "streams" / "prices.yaml").write_text(
        "id: prices\nfrom: {source: prices}\nmap: {entrypoint: identity}\n",
        encoding="utf-8",
    )
    (tmp_path / "dataset.yaml").write_text(
        "sample: {rounding: ceil, cadence: 1h}\n"
        "features:\n"
        "  - {id: price, stream: prices, field: value}\n"
        "targets: []\n",
        encoding="utf-8",
    )
    return load_project_definition(project), source_path


def _edit_source(path: Path) -> None:
    path.write_text('{"changed": true}\n', encoding="utf-8")


def _add_source(path: Path) -> None:
    path.with_name("b.jsonl").write_text("{}\n", encoding="utf-8")


def _remove_source(path: Path) -> None:
    path.unlink()


def _build_state_path(definition: ProjectDefinition) -> Path:
    return (
        definition.project.artifacts_root / "_system" / "build" / "state.json"
    ).resolve()


def _write_artifact(root: Path, task: ArtifactTask) -> None:
    destination = root / task.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("{}", encoding="utf-8")


def _register_artifact(
    state: BuildState,
    root: Path,
    task: ArtifactTask,
    artifact_hash: str,
) -> None:
    _write_artifact(root, task)
    state.register(
        task.id,
        artifact_hash=artifact_hash,
        files=(ArtifactFileFingerprint.from_path(task.output, root / task.output),),
    )


def _build_artifact(runtime, task: ArtifactTask) -> ArtifactOutput:
    _write_artifact(runtime.artifacts_root, task)
    return ArtifactOutput()


def _patch_artifact_build(monkeypatch, build) -> None:
    def load_entrypoint(operation_group, entrypoint):
        assert operation_group == "jerrythomas.operations.build"

        def run(*, runtime, task_cfg):
            assert task_cfg.entrypoint == entrypoint
            return build(runtime, task_cfg)

        return run

    monkeypatch.setattr(build_exec, "load_entrypoint", load_entrypoint)


def _patch_stable_artifact_inputs(monkeypatch) -> None:
    monkeypatch.setattr(
        build_exec,
        "_require_stable_artifact_inputs",
        lambda definition, artifact_id, expected_hash: None,
    )


def _build_settings(mode: str = "auto") -> BuildSettings:
    return BuildSettings(
        mode=mode,
        observability=ObservabilitySettings(
            visuals=True,
            heartbeat_interval_seconds=DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
            log_decision=LogLevelDecision(name="INFO", value=logging.INFO),
            log_output=LogOutputSettings(
                outputs=(LogOutputTarget(transport="stderr"),)
            ),
        ),
    )


@pytest.mark.parametrize("version", [1, 2, 3, 4, 5, 6, 7, 8])
def test_load_build_state_invalidates_previous_cache_version(
    tmp_path: Path,
    version: int,
) -> None:
    artifacts_root = tmp_path / "artifacts"
    state_path = artifacts_root / "_system/build/state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        f'{{"version": {version}, "artifacts": {{}}}}',
        encoding="utf-8",
    )

    assert load_build_state(artifacts_root) is None


def test_report_artifact_plan_logs_current_roots(monkeypatch) -> None:
    captured: list[tuple[str, int]] = []
    monkeypatch.setattr(
        build_exec,
        "emit_execution_message",
        lambda message, level=logging.INFO: captured.append((message, level)),
    )

    build_exec._report_artifact_plan(
        build_exec.SkippedBuild(
            reason="up_to_date",
            artifacts=(SERIES, VECTOR_METADATA),
        ),
        mode="auto",
        requested_artifacts={VECTOR_METADATA},
    )

    assert len(captured) == 1
    message, level = captured[0]
    assert level == logging.DEBUG
    assert message.startswith("Artifact plan:\n")
    assert json.loads(message.removeprefix("Artifact plan:\n")) == {
        "action": "skip",
        "reason": "up_to_date",
        "mode": "auto",
        "requested": [VECTOR_METADATA],
        "required": [SERIES, VECTOR_METADATA],
        "jobs": [],
        "current": [SERIES, VECTOR_METADATA],
    }


def test_report_artifact_plan_logs_not_required(monkeypatch) -> None:
    captured: list[tuple[str, int]] = []
    monkeypatch.setattr(
        build_exec,
        "emit_execution_message",
        lambda message, level=logging.INFO: captured.append((message, level)),
    )

    build_exec._report_artifact_plan(
        build_exec.SkippedBuild(reason="not_required", artifacts=()),
        mode="auto",
        requested_artifacts={SCALER_STATISTICS},
    )

    assert len(captured) == 1
    message, level = captured[0]
    assert level == logging.DEBUG
    assert message.startswith("Artifact plan:\n")
    assert json.loads(message.removeprefix("Artifact plan:\n")) == {
        "action": "skip",
        "reason": "not_required",
        "mode": "auto",
        "requested": [SCALER_STATISTICS],
        "required": [],
        "jobs": [],
        "current": [],
    }


def test_report_artifact_plan_keeps_run_details_at_debug(monkeypatch) -> None:
    task = MetadataTask(id="metadata")
    captured: list[tuple[str, int]] = []
    monkeypatch.setattr(
        build_exec,
        "emit_execution_message",
        lambda message, level=logging.INFO: captured.append((message, level)),
    )

    build_exec._report_artifact_plan(
        build_exec.BuildPlan(
            reason="force",
            artifacts=(SERIES, VECTOR_METADATA),
            jobs=(build_exec.ArtifactBuildJob(task, (VECTOR_METADATA,)),),
            previous_state=None,
        ),
        mode="rebuild",
        requested_artifacts={VECTOR_METADATA},
    )

    assert len(captured) == 1
    message, level = captured[0]
    assert level == logging.DEBUG
    assert message.startswith("Artifact plan:\n")
    assert json.loads(message.removeprefix("Artifact plan:\n")) == {
        "action": "run",
        "reason": "force",
        "mode": "rebuild",
        "requested": [VECTOR_METADATA],
        "required": [SERIES, VECTOR_METADATA],
        "jobs": [VECTOR_METADATA],
        "current": [SERIES],
    }


def test_plan_skips_scaler_when_dataset_has_no_scaled_features(
    tmp_path: Path,
) -> None:
    definition = _definition(tmp_path, _dataset_with_feature(scale=False))
    graph = build_artifact_graph([ScalerTask(id="scaler")])
    definition = replace(definition, artifact_graph=graph)

    plan = build_exec._plan_build(
        definition=definition,
        required_artifacts={SCALER_STATISTICS},
        mode="auto",
    )

    assert plan == build_exec.SkippedBuild(reason="not_required", artifacts=())


def test_plan_builds_only_requested_generic_artifact(
    tmp_path: Path,
) -> None:
    definition = _definition(tmp_path)
    task = ArtifactTask(
        id="custom_snapshot",
        entrypoint="plugin.snapshot",
        output="build/custom.json",
    )
    graph = build_artifact_graph([task, ScalerTask(id="scaler")])
    definition = replace(
        definition,
        artifact_graph=graph,
        artifact_hashes=ArtifactHashes(
            {
                task.id: "artifact-hash",
                SCALER_STATISTICS: "scaler-hash",
            }
        ),
    )

    plan = build_exec._plan_build(
        definition=definition,
        required_artifacts={task.id},
        mode="rebuild",
    )

    assert isinstance(plan, build_exec.BuildPlan)
    assert plan.artifacts == (task.id,)
    assert tuple(job.task.id for job in plan.jobs) == (task.id,)


def test_plan_expands_metadata_dependencies(tmp_path: Path) -> None:
    definition = _definition(tmp_path, _dataset_with_feature(scale=False))
    series = SeriesTask(id="series")
    metadata = MetadataTask(id="metadata")
    graph = build_artifact_graph([ScalerTask(id="scaler"), series, metadata])
    definition = replace(definition, artifact_graph=graph)
    plan = build_exec._plan_build(
        definition=definition,
        required_artifacts={VECTOR_METADATA},
        mode="rebuild",
    )

    assert isinstance(plan, build_exec.BuildPlan)
    assert plan.artifacts == (SERIES, VECTOR_METADATA)
    assert tuple(job.task.id for job in plan.jobs) == (
        SERIES,
        VECTOR_METADATA,
    )


def test_v5_vector_inputs_state_is_not_reused(tmp_path: Path) -> None:
    definition = _definition(tmp_path, _dataset_with_feature(scale=False))
    series = SeriesTask()
    metadata = MetadataTask()
    graph = build_artifact_graph([series, metadata])
    definition = replace(definition, artifact_graph=graph)
    previous_state = BuildState()
    _register_artifact(
        previous_state,
        definition.project.artifacts_root,
        ArtifactTask(
            id="vector_inputs",
            entrypoint="core.artifact.vector_inputs",
            output="build/vector_inputs/manifest.json",
        ),
        "current",
    )
    save_build_state(previous_state, definition.project.artifacts_root)

    plan = build_exec._plan_build(
        definition=definition,
        required_artifacts={VECTOR_METADATA},
        mode="auto",
    )

    assert isinstance(plan, build_exec.BuildPlan)
    assert plan.reason == "missing"
    assert tuple(job.task.id for job in plan.jobs) == (
        SERIES,
        VECTOR_METADATA,
    )


def test_plan_skips_current_dependency(tmp_path: Path) -> None:
    definition = _definition(tmp_path, _dataset_with_feature(scale=False))
    series = SeriesTask(id="series")
    metadata = MetadataTask(id="metadata")
    graph = build_artifact_graph([series, metadata])
    definition = replace(definition, artifact_graph=graph)
    state = BuildState()
    _register_artifact(
        state,
        definition.project.artifacts_root,
        series,
        definition.artifact_hashes.for_artifact(series.id),
    )
    save_build_state(state, definition.project.artifacts_root)

    plan = build_exec._plan_build(
        definition=definition,
        required_artifacts={VECTOR_METADATA},
        mode="auto",
    )

    assert isinstance(plan, build_exec.BuildPlan)
    assert tuple(job.task.id for job in plan.jobs) == (VECTOR_METADATA,)


def test_runtime_operation_change_keeps_artifact_plan_current(
    tmp_path: Path,
) -> None:
    project = _write_project(tmp_path)
    (tmp_path / "operations" / "custom_snapshot.yaml").write_text(
        "kind: artifact\nentrypoint: plugin.snapshot\noutput: build/custom.json\n",
        encoding="utf-8",
    )
    runtime_operation = tmp_path / "operations" / "custom_report.yaml"
    runtime_operation.write_text(
        "kind: runtime\nentrypoint: plugin.report\noptions: {threshold: 1}\n",
        encoding="utf-8",
    )
    first = load_project_definition(project)
    task = next(
        task
        for task in first.artifact_graph.tasks_by_id.values()
        if task.id == "custom_snapshot"
    )
    state = BuildState()
    _register_artifact(
        state,
        first.project.artifacts_root,
        task,
        first.artifact_hashes.for_artifact(task.id),
    )
    save_build_state(state, first.project.artifacts_root)

    runtime_operation.write_text(
        "kind: runtime\nentrypoint: plugin.report\noptions: {threshold: 2}\n",
        encoding="utf-8",
    )
    second = load_project_definition(project)
    plan = build_exec._plan_build(
        definition=second,
        required_artifacts={task.id},
        mode="auto",
    )

    assert second.runtime_operations != first.runtime_operations
    assert second.artifact_hashes == first.artifact_hashes
    assert plan == build_exec.SkippedBuild(
        reason="up_to_date",
        artifacts=(task.id,),
    )


def test_plan_rejects_resolved_artifact_that_became_stale(
    tmp_path: Path,
) -> None:
    definition = _definition(tmp_path)
    first = ArtifactTask(
        id="first_snapshot",
        entrypoint="plugin.first",
        output="build/first.json",
    )
    second = ArtifactTask(
        id="second_snapshot",
        entrypoint="plugin.second",
        output="build/second.json",
    )
    graph = build_artifact_graph([first, second])
    definition = replace(
        definition,
        artifact_graph=graph,
        artifact_hashes=ArtifactHashes({first.id: "current", second.id: "current"}),
    )
    state = BuildState()
    _register_artifact(
        state,
        definition.project.artifacts_root,
        first,
        "stale-artifact-hash",
    )
    save_build_state(state, definition.project.artifacts_root)

    with pytest.raises(RuntimeError, match="earlier build profile became stale"):
        build_exec._plan_build(
            definition=definition,
            required_artifacts={second.id},
            mode="rebuild",
            resolved_artifacts={first.id},
        )


def test_plan_rejects_missing_dependency_producer(
    tmp_path: Path,
) -> None:
    definition = _definition(tmp_path, _dataset_with_feature(scale=False))
    graph = build_artifact_graph([MetadataTask(id="metadata")])
    definition = replace(definition, artifact_graph=graph)

    with pytest.raises(
        ArtifactResolutionError,
        match="Required artifact operation 'series' is not declared",
    ):
        build_exec._plan_build(
            definition=definition,
            required_artifacts={VECTOR_METADATA},
            mode="auto",
        )


def test_plan_rejects_unknown_artifact(tmp_path: Path) -> None:
    definition = _definition(tmp_path)
    definition = replace(
        definition,
        artifact_graph=build_artifact_graph([]),
        artifact_hashes=ArtifactHashes({}),
    )

    with pytest.raises(
        ArtifactResolutionError,
        match="Unknown artifact 'unknown'",
    ):
        build_exec._plan_build(
            definition=definition,
            required_artifacts={"unknown"},
            mode="auto",
        )


def test_stale_dependency_rebuilds_current_dependent(
    tmp_path: Path,
) -> None:
    definition = _definition(tmp_path, _dataset_with_feature(scale=False))
    series = SeriesTask(id="series")
    metadata = MetadataTask(id="metadata")
    graph = build_artifact_graph([series, metadata])
    definition = replace(definition, artifact_graph=graph)
    state = BuildState()
    _register_artifact(
        state,
        definition.project.artifacts_root,
        series,
        "stale-artifact-hash",
    )
    _register_artifact(
        state,
        definition.project.artifacts_root,
        metadata,
        definition.artifact_hashes.for_artifact(metadata.id),
    )
    save_build_state(state, definition.project.artifacts_root)

    plan = build_exec._plan_build(
        definition=definition,
        required_artifacts={VECTOR_METADATA},
        mode="auto",
    )

    assert isinstance(plan, build_exec.BuildPlan)
    assert tuple(job.task.id for job in plan.jobs) == (
        SERIES,
        VECTOR_METADATA,
    )
    assert plan.jobs[0].invalidated_artifacts == (
        SERIES,
        VECTOR_METADATA,
        COVERAGE_STATS,
    )


def test_mode_off_rejects_missing_artifact(tmp_path: Path) -> None:
    definition = _definition(tmp_path)
    task = ArtifactTask(
        id="snapshot",
        entrypoint="plugin.snapshot",
        output="build/snapshot.json",
    )
    graph = build_artifact_graph([task])
    definition = replace(
        definition,
        artifact_graph=graph,
        artifact_hashes=ArtifactHashes({task.id: "artifact-hash"}),
    )

    with pytest.raises(
        ArtifactResolutionError,
        match=(
            "Artifact mode is require_current, but required artifacts are missing or stale: "
            "snapshot"
        ),
    ):
        build_exec._plan_build(
            definition=definition,
            required_artifacts={task.id},
            mode="require_current",
        )


def test_execute_build_rejects_invalid_operation_result(
    monkeypatch,
    tmp_path: Path,
) -> None:
    definition = _definition(tmp_path)
    _patch_stable_artifact_inputs(monkeypatch)
    task = ArtifactTask(
        id="snapshot",
        entrypoint="plugin.snapshot",
        output="build/snapshot.json",
    )
    graph = build_artifact_graph([task])
    definition = replace(
        definition,
        artifact_graph=graph,
        artifact_hashes=ArtifactHashes({task.id: "artifact-hash"}),
    )
    plan = build_exec.BuildPlan(
        reason="missing",
        artifacts=(task.id,),
        jobs=(build_exec.ArtifactBuildJob(task, (task.id,)),),
        previous_state=None,
    )
    _patch_artifact_build(monkeypatch, lambda _runtime, _task: None)

    with pytest.raises(TypeError, match="must return ArtifactOutput"):
        build_exec._execute_build_jobs(
            definition,
            runtime=_runtime(tmp_path / "artifacts"),
            plan=plan,
            settings=_build_settings(),
        )


def test_execute_build_jobs_persists_completed_job_before_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    definition = _definition(tmp_path)
    _patch_stable_artifact_inputs(monkeypatch)
    series = SeriesTask(id="series")
    metadata = MetadataTask(id="metadata")
    graph = build_artifact_graph([series, metadata])
    definition = replace(
        definition,
        artifact_graph=graph,
        artifact_hashes=ArtifactHashes(
            {
                SERIES: "artifact-hash-1",
                VECTOR_METADATA: "artifact-hash-1",
            }
        ),
    )
    previous_state = BuildState()
    previous_state.register(
        VECTOR_METADATA,
        artifact_hash="artifact-hash-1",
        files=(
            ArtifactFileFingerprint(
                relative_path=metadata.output,
                size=0,
                mtime_ns=0,
                ctime_ns=0,
                sha256="0" * 64,
            ),
        ),
    )
    save_build_state(previous_state, definition.project.artifacts_root)

    def build(runtime, task):
        if task.id == VECTOR_METADATA:
            raise RuntimeError("metadata failed")
        return _build_artifact(runtime, task)

    outputs: list[tuple[str, Path]] = []
    _patch_artifact_build(monkeypatch, build)
    monkeypatch.setattr(
        build_exec,
        "emit_file_result",
        lambda label, path: outputs.append((label, path)),
    )
    plan = build_exec.BuildPlan(
        reason="stale",
        artifacts=(SERIES, VECTOR_METADATA),
        jobs=(
            build_exec.ArtifactBuildJob(
                series,
                (SERIES, VECTOR_METADATA),
            ),
            build_exec.ArtifactBuildJob(metadata, (VECTOR_METADATA,)),
        ),
        previous_state=previous_state,
    )

    runtime = _runtime(tmp_path / "artifacts")
    with pytest.raises(RuntimeError, match="metadata failed"):
        build_exec._execute_build_jobs(
            definition,
            runtime=runtime,
            plan=plan,
            settings=_build_settings(),
        )

    state = load_build_state(definition.project.artifacts_root)
    assert state is not None
    assert list(state.artifacts) == [SERIES]
    assert outputs == [
        (
            "Series",
            (runtime.artifacts_root / series.output).resolve(),
        )
    ]


def test_execute_build_failure_preserves_previous_persisted_state(
    monkeypatch,
    tmp_path: Path,
) -> None:
    definition = _definition(tmp_path)
    _patch_stable_artifact_inputs(monkeypatch)
    series = SeriesTask(id="series")
    metadata = MetadataTask(id="metadata")
    graph = build_artifact_graph([series, metadata])
    definition = replace(
        definition,
        artifact_graph=graph,
        artifact_hashes=ArtifactHashes(
            {
                SERIES: "artifact-hash-1",
                VECTOR_METADATA: "artifact-hash-1",
            }
        ),
    )
    previous_state = BuildState()
    previous_state.register(
        VECTOR_METADATA,
        artifact_hash="artifact-hash-1",
        files=(
            ArtifactFileFingerprint(
                relative_path=metadata.output,
                size=0,
                mtime_ns=0,
                ctime_ns=0,
                sha256="0" * 64,
            ),
        ),
    )
    save_build_state(previous_state, definition.project.artifacts_root)

    def fail_build(runtime, task):
        raise RuntimeError("series failed")

    _patch_artifact_build(monkeypatch, fail_build)
    plan = build_exec.BuildPlan(
        reason="stale",
        artifacts=(SERIES, VECTOR_METADATA),
        jobs=(
            build_exec.ArtifactBuildJob(
                series,
                (SERIES, VECTOR_METADATA),
            ),
        ),
        previous_state=previous_state,
    )

    with pytest.raises(RuntimeError, match="series failed"):
        build_exec._execute_build_jobs(
            definition,
            runtime=_runtime(tmp_path / "artifacts"),
            plan=plan,
            settings=_build_settings(),
        )

    persisted = load_build_state(definition.project.artifacts_root)
    assert persisted is not None
    assert list(persisted.artifacts) == [VECTOR_METADATA]


@pytest.mark.parametrize(
    "task",
    [
        MetadataTask(output="linked/metadata.json"),
        ArtifactTask(
            id="snapshot",
            entrypoint="plugin.snapshot",
            output="linked/snapshot.json",
        ),
    ],
    ids=("core", "plugin"),
)
def test_execute_build_rejects_symlink_escape_before_calling_runner(
    monkeypatch,
    tmp_path: Path,
    task: ArtifactTask,
) -> None:
    definition = _definition(tmp_path)
    _patch_stable_artifact_inputs(monkeypatch)
    artifacts_root = tmp_path / "artifacts"
    artifacts_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    victim = outside / Path(task.output).name
    victim.write_text("keep", encoding="utf-8")
    (artifacts_root / "linked").symlink_to(outside, target_is_directory=True)
    runner_calls: list[str] = []

    def mutate_outside(runtime, task_cfg):
        runner_calls.append(task_cfg.id)
        victim.write_text("mutated", encoding="utf-8")
        return ArtifactOutput()

    _patch_artifact_build(monkeypatch, mutate_outside)
    graph = build_artifact_graph([task])
    definition = replace(
        definition,
        artifact_graph=graph,
        artifact_hashes=ArtifactHashes({task.id: "artifact-hash-1"}),
    )
    plan = build_exec.BuildPlan(
        reason="force",
        artifacts=(task.id,),
        jobs=(build_exec.ArtifactBuildJob(task, (task.id,)),),
        previous_state=None,
    )

    with pytest.raises(ValueError, match="must stay under artifacts root"):
        build_exec._execute_build_jobs(
            definition,
            runtime=_runtime(artifacts_root),
            plan=plan,
            settings=_build_settings("rebuild"),
        )

    assert runner_calls == []
    assert victim.read_text(encoding="utf-8") == "keep"


def test_execute_build_preflights_every_output_before_running_any_job(
    monkeypatch,
    tmp_path: Path,
) -> None:
    definition = _definition(tmp_path)
    _patch_stable_artifact_inputs(monkeypatch)
    artifacts_root = tmp_path / "artifacts"
    artifacts_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (artifacts_root / "linked").symlink_to(outside, target_is_directory=True)
    first = ArtifactTask(
        id="first",
        entrypoint="plugin.first",
        output="build/first.json",
    )
    second = ArtifactTask(
        id="second",
        entrypoint="plugin.second",
        output="linked/second.json",
    )
    runner_calls: list[str] = []

    def build(runtime, task_cfg):
        runner_calls.append(task_cfg.id)
        _write_artifact(runtime.artifacts_root, task_cfg)
        return ArtifactOutput()

    _patch_artifact_build(monkeypatch, build)
    graph = build_artifact_graph([first, second])
    definition = replace(
        definition,
        artifact_graph=graph,
        artifact_hashes=ArtifactHashes(
            {first.id: "artifact-hash-1", second.id: "artifact-hash-2"}
        ),
    )
    plan = build_exec.BuildPlan(
        reason="force",
        artifacts=(first.id, second.id),
        jobs=(
            build_exec.ArtifactBuildJob(first, (first.id,)),
            build_exec.ArtifactBuildJob(second, (second.id,)),
        ),
        previous_state=None,
    )

    with pytest.raises(ValueError, match="must stay under artifacts root"):
        build_exec._execute_build_jobs(
            definition,
            runtime=_runtime(artifacts_root),
            plan=plan,
            settings=_build_settings("rebuild"),
        )

    assert runner_calls == []
    assert not (artifacts_root / first.output).exists()


def test_execute_build_job_invalidates_only_graph_descendants(
    monkeypatch,
    tmp_path: Path,
) -> None:
    definition = _definition(tmp_path)
    _patch_stable_artifact_inputs(monkeypatch)
    scaler = ScalerTask(id="scaler")
    custom = ArtifactTask(
        id="custom_snapshot",
        entrypoint="plugin.snapshot",
        output="build/custom.json",
    )
    series = SeriesTask(id="series")
    graph = build_artifact_graph(
        [
            scaler,
            series,
            MetadataTask(id="metadata"),
            CoverageStatsTask(id="coverage_stats", stage="postprocessed"),
            custom,
        ]
    )
    definition = replace(
        definition,
        artifact_graph=graph,
        artifact_hashes=ArtifactHashes(
            {key: "artifact-hash-1" for key in graph.tasks_by_id}
        ),
    )
    runtime = _runtime(tmp_path / "artifacts")
    messages: list[tuple[str, int]] = []
    monkeypatch.setattr(
        build_exec,
        "emit_execution_message",
        lambda message, level: messages.append((message, level)),
    )
    previous_state = BuildState()
    for task in graph.tasks_by_id.values():
        _register_artifact(
            previous_state,
            runtime.artifacts_root,
            task,
            "artifact-hash-1",
        )
    _patch_artifact_build(monkeypatch, _build_artifact)
    plan = build_exec.BuildPlan(
        reason="force",
        artifacts=(SERIES,),
        jobs=(
            build_exec.ArtifactBuildJob(
                series,
                (
                    SERIES,
                    VECTOR_METADATA,
                    COVERAGE_STATS,
                ),
            ),
        ),
        previous_state=previous_state,
    )

    state = build_exec._execute_build_jobs(
        definition,
        runtime=runtime,
        plan=plan,
        settings=_build_settings("rebuild"),
    )

    assert set(state.artifacts) == {SCALER_STATISTICS, SERIES, custom.id}
    assert runtime.artifacts.has(SERIES)
    assert not runtime.artifacts.has(VECTOR_METADATA)
    assert len(messages) == 1
    message, level = messages[0]
    assert level == logging.DEBUG
    assert message.startswith("Config:\n")
    config = json.loads(message[8:])
    assert config["operation"]["entrypoint"] == "core.artifact.series"
    assert config["mode"] == "rebuild"
    assert config["execution"] == {"sort_buffer_mb": 128, "workers": 1}
    assert config["observability"]["visuals"] is True


@pytest.mark.parametrize(
    "mutate_source",
    [_edit_source, _add_source, _remove_source],
    ids=("edit", "glob-add", "glob-remove"),
)
def test_build_rejects_source_drift_before_starting_runner(
    monkeypatch,
    tmp_path: Path,
    mutate_source,
) -> None:
    definition, source_path = _definition_with_local_source(tmp_path)
    runner_calls: list[str] = []

    def build(runtime, task):
        runner_calls.append(task.id)
        return _build_artifact(runtime, task)

    _patch_artifact_build(monkeypatch, build)
    mutate_source(source_path)

    with pytest.raises(RuntimeError, match="Source files changed while building"):
        build_exec.run_build_if_needed(
            definition,
            required_artifacts={SERIES},
            settings=_build_settings("rebuild"),
            runtime=_runtime(definition.project.artifacts_root),
        )

    assert runner_calls == []
    assert not _build_state_path(definition).exists()


def test_build_rejects_source_drift_before_registering_result(
    monkeypatch,
    tmp_path: Path,
) -> None:
    definition, source_path = _definition_with_local_source(tmp_path)
    task = definition.artifact_graph.tasks_by_id[SERIES]
    previous_state = BuildState()
    _register_artifact(
        previous_state,
        definition.project.artifacts_root,
        task,
        definition.artifact_hashes.for_artifact(task.id),
    )
    state_path = _build_state_path(definition)
    save_build_state(previous_state, definition.project.artifacts_root)
    persisted_state = state_path.read_bytes()
    runtime = _runtime(definition.project.artifacts_root)

    def build(runtime, task):
        _edit_source(source_path)
        return _build_artifact(runtime, task)

    _patch_artifact_build(monkeypatch, build)

    with pytest.raises(RuntimeError, match="Source files changed while building"):
        build_exec.run_build_if_needed(
            definition,
            required_artifacts={SERIES},
            settings=_build_settings("rebuild"),
            runtime=runtime,
        )

    assert state_path.read_bytes() == persisted_state
    assert not runtime.artifacts.has(SERIES)


def test_build_accepts_unchanged_local_sources(monkeypatch, tmp_path: Path) -> None:
    definition, _ = _definition_with_local_source(tmp_path)
    _patch_artifact_build(monkeypatch, _build_artifact)
    runtime = _runtime(definition.project.artifacts_root)

    assert build_exec.run_build_if_needed(
        definition,
        required_artifacts={SERIES},
        settings=_build_settings("rebuild"),
        runtime=runtime,
    )

    state = load_build_state(definition.project.artifacts_root)
    assert state is not None
    assert state.artifacts[SERIES].artifact_hash == (
        definition.artifact_hashes.for_artifact(SERIES)
    )
    assert runtime.artifacts.has(SERIES)


def test_run_build_hydrates_current_dependencies_before_job(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _patch_stable_artifact_inputs(monkeypatch)
    definition = _definition(tmp_path, _dataset_with_feature(scale=False))
    series = SeriesTask(id="series")
    metadata = MetadataTask(id="metadata")
    coverage_stats = CoverageStatsTask(id="coverage_stats", stage="assembled")
    graph = build_artifact_graph([series, metadata, coverage_stats])
    definition = replace(definition, artifact_graph=graph)
    state = BuildState()
    for task in (series, metadata):
        _register_artifact(
            state,
            definition.project.artifacts_root,
            task,
            definition.artifact_hashes.for_artifact(task.id),
        )
    save_build_state(state, definition.project.artifacts_root)
    runtime = _runtime(definition.project.artifacts_root)

    def build(runtime, task):
        assert task == coverage_stats
        assert task is not coverage_stats
        assert runtime.artifacts.has(VECTOR_METADATA)
        return _build_artifact(runtime, task)

    _patch_artifact_build(monkeypatch, build)

    did_build = build_exec.run_build_if_needed(
        definition,
        required_artifacts={COVERAGE_STATS},
        settings=_build_settings(),
        runtime=runtime,
    )

    assert did_build is True


def test_force_build_preserves_artifacts_resolved_by_previous_profile(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _patch_stable_artifact_inputs(monkeypatch)
    definition = _definition(tmp_path, _dataset_with_feature(scale=False))
    series = SeriesTask(id="series")
    metadata = MetadataTask(id="metadata")
    coverage_stats = CoverageStatsTask(id="coverage_stats", stage="postprocessed")
    graph = build_artifact_graph([series, metadata, coverage_stats])
    definition = replace(definition, artifact_graph=graph)
    state = BuildState()
    for task in (series, metadata):
        _register_artifact(
            state,
            definition.project.artifacts_root,
            task,
            definition.artifact_hashes.for_artifact(task.id),
        )
    save_build_state(state, definition.project.artifacts_root)
    built: list[str] = []

    def build(runtime, task):
        built.append(task.id)
        return _build_artifact(runtime, task)

    _patch_artifact_build(monkeypatch, build)
    runtime = _runtime(definition.project.artifacts_root)
    resolved: set[str] = set()

    assert not build_exec.run_build_if_needed(
        definition,
        required_artifacts={VECTOR_METADATA},
        settings=_build_settings(),
        runtime=runtime,
        resolved_artifacts=resolved,
    )
    assert build_exec.run_build_if_needed(
        definition,
        required_artifacts={COVERAGE_STATS},
        settings=_build_settings("rebuild"),
        runtime=runtime,
        resolved_artifacts=resolved,
    )

    assert built == [COVERAGE_STATS]
    assert resolved == {SERIES, VECTOR_METADATA, COVERAGE_STATS}


def test_run_build_keeps_loaded_definition_when_config_changes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    definition = _definition(tmp_path)
    task = ArtifactTask(
        id="snapshot",
        entrypoint="plugin.snapshot",
        output="build/snapshot.json",
    )
    graph = build_artifact_graph([task])
    definition = replace(
        definition,
        artifact_graph=graph,
        artifact_hashes=ArtifactHashes({task.id: "current"}),
    )
    plan = build_exec.BuildPlan(
        reason="missing",
        artifacts=(task.id,),
        jobs=(),
        previous_state=None,
    )
    monkeypatch.setattr(build_exec, "_plan_build", lambda **_kwargs: plan)
    monkeypatch.setattr(
        build_exec, "_report_artifact_plan", lambda *_args, **_kwargs: None
    )

    def change_project(_definition, **_kwargs) -> None:
        definition.project.path.write_text(
            definition.project.path.read_text(encoding="utf-8") + "\n# changed\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(build_exec, "_execute_build_jobs", change_project)
    resolved: set[str] = set()

    did_build = build_exec.run_build_if_needed(
        definition,
        required_artifacts={task.id},
        settings=_build_settings(),
        runtime=_runtime(tmp_path / "artifacts"),
        resolved_artifacts=resolved,
    )

    assert did_build is True
    assert resolved == {task.id}


def test_run_build_rejects_unknown_mode(tmp_path: Path) -> None:
    definition = replace(
        _definition(tmp_path),
        artifact_graph=build_artifact_graph([]),
    )
    with pytest.raises(ValueError, match="Unknown artifact mode 'sometimes'"):
        build_exec.run_build_if_needed(
            definition,
            required_artifacts=set(),
            settings=_build_settings("sometimes"),
            runtime=_runtime(definition.project.artifacts_root),
        )
