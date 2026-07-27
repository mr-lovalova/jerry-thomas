import json
import logging
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from datapipeline.artifacts.errors import ArtifactResolutionError
from datapipeline.artifacts.planning import build_artifact_graph
from datapipeline.artifacts.registry import ArtifactRegistry
from datapipeline.artifacts.settings import BuildSettings
from datapipeline.artifacts.specs import (
    SERIES,
    VECTOR_METADATA,
    COVERAGE_STATS,
)
from datapipeline.build.state import (
    ArtifactFileFingerprint,
    BuildState,
    save_build_state,
)
from datapipeline.config.dataset.dataset import DatasetConfig, SampleConfig
from datapipeline.config.dataset.series import SeriesConfig
from datapipeline.config.execution import ExecutionConfig
from datapipeline.config.preview import PreviewStage
from datapipeline.config.streams import StreamsConfig
from datapipeline.config.tasks.base import ArtifactTask, RuntimeTask
from datapipeline.config.tasks.coverage import CoverageTask
from datapipeline.config.tasks.coverage_stats import CoverageStatsTask
from datapipeline.config.tasks.dataset import DatasetTask
from datapipeline.config.tasks.matrix import MatrixTask
from datapipeline.config.tasks.metadata import MetadataTask
from datapipeline.config.tasks.series import SeriesTask
from datapipeline.config.tasks.schedule import ScheduleTask
from datapipeline.execution.settings import (
    DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
    LogLevelDecision,
    LogOutputSettings,
    ObservabilitySettings,
)
from datapipeline.io.output import OutputTarget
from datapipeline.io.runs import (
    RunPaths,
    finish_run_success,
    set_latest_run,
    start_run,
)
from datapipeline.operations.persistence import (
    RoutedRuntimeOutput,
    RuntimeOutputBatch,
)
from datapipeline.profiles.execution import (
    RuntimeJobPlan,
    execute_runtime_job,
    plan_runtime_job,
    run_runtime_operation,
)
from datapipeline.profiles.models import (
    BuildJob,
    BuildRunRequest,
    MaterializeJob,
    MaterializeRunRequest,
    RuntimeJob,
    RuntimeRunRequest,
    ServeRunPlan,
)
from datapipeline.profiles.errors import ProfileCommandError
from datapipeline.profiles.orchestration import (
    _prune_series_caches,
    _validate_build_order,
    run_profiles,
)
from datapipeline.services.materialize import resolve_materialize_output
from tests.unit.profiles.helpers import project_definition

_LOG_DECISION = LogLevelDecision(name="INFO", value=logging.INFO)
_LOG_OUTPUT = LogOutputSettings(outputs=())


@contextmanager
def _passthrough_execution_scope(_runtime, _observability):
    yield


def _artifact_settings(
    mode: str = "AUTO",
    heartbeat_interval_seconds: float = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
    visuals: str = "off",
) -> BuildSettings:
    return BuildSettings(
        mode=mode,
        observability=_observability(heartbeat_interval_seconds, visuals),
    )


def _observability(
    heartbeat_interval_seconds: float = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
    visuals: str = "off",
) -> ObservabilitySettings:
    return ObservabilitySettings(
        visuals=visuals,
        heartbeat_interval_seconds=heartbeat_interval_seconds,
        log_decision=_LOG_DECISION,
        log_output=_LOG_OUTPUT,
    )


def _dataset(*, scale: bool = False) -> DatasetConfig:
    return DatasetConfig(
        sample=SampleConfig(cadence="1h"),
        features=[
            SeriesConfig(
                id="feature",
                stream="stream",
                field="value",
                scale=scale,
            )
        ],
    )


def _stream_catalog() -> StreamsConfig:
    return StreamsConfig.model_validate(
        {
            "sources": {
                "test.source": {
                    "id": "test.source",
                    "parser": {"entrypoint": "identity"},
                    "loader": {"entrypoint": "identity"},
                }
            },
            "streams": {
                stream_id: {
                    "id": stream_id,
                    "from": {"source": "test.source"},
                    "map": {"entrypoint": "identity"},
                }
                for stream_id in ("stream", "prices")
            },
        }
    )


def _runtime(tmp_path: Path, marker: str = "runtime") -> SimpleNamespace:
    return SimpleNamespace(
        marker=marker,
        artifacts=ArtifactRegistry(tmp_path / marker / "artifacts"),
        dataset=_dataset(),
        execution=ExecutionConfig(),
        heartbeat_interval_seconds=None,
        output_ids=(),
    )


def _output() -> OutputTarget:
    return OutputTarget(
        transport="stdout",
        format="jsonl",
        view="raw",
        encoding=None,
        destination=None,
    )


def _materialize_output(path: Path) -> OutputTarget:
    return resolve_materialize_output(path)


def _runtime_job(
    name: str,
    task: RuntimeTask,
    runtime,
    *,
    limit: int | None = None,
    preview: PreviewStage | None = None,
    heartbeat_interval_seconds: float = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
    output_ids: tuple[str, ...] = (),
    output: OutputTarget | None = None,
) -> RuntimeJob:
    return RuntimeJob(
        name=name,
        task=task,
        runtime=runtime,
        output=_output() if output is None else output,
        observability=_observability(heartbeat_interval_seconds),
        limit=limit,
        throttle_ms=None,
        preview=preview,
        output_ids=output_ids,
    )


def _build_request(
    tmp_path: Path,
    artifact_tasks,
    jobs,
    execution: ExecutionConfig | None = None,
) -> BuildRunRequest:
    return BuildRunRequest(
        definition=project_definition(
            tmp_path / "project.yaml",
            dataset=_dataset(),
            streams=_stream_catalog(),
            artifact_operations=artifact_tasks,
        ),
        jobs=jobs,
        execution=ExecutionConfig() if execution is None else execution,
    )


def _runtime_request(
    tmp_path: Path,
    *,
    command: str,
    artifact_tasks,
    jobs,
    artifact_settings: BuildSettings | None = None,
    serve_run_plans: tuple[ServeRunPlan, ...] = (),
    execution: ExecutionConfig | None = None,
) -> RuntimeRunRequest:
    return RuntimeRunRequest(
        command=command,
        definition=project_definition(
            tmp_path / "project.yaml",
            dataset=_dataset(),
            streams=_stream_catalog(),
            artifact_operations=artifact_tasks,
        ),
        jobs=jobs,
        execution=ExecutionConfig() if execution is None else execution,
        artifact_settings=artifact_settings or _artifact_settings(),
        serve_run_plans=serve_run_plans,
    )


def _materialize_request(
    tmp_path: Path,
    artifact_tasks,
    jobs,
    runtime,
    execution: ExecutionConfig | None = None,
) -> MaterializeRunRequest:
    return MaterializeRunRequest(
        definition=project_definition(
            tmp_path / "project.yaml",
            dataset=_dataset(),
            streams=_stream_catalog(),
            artifact_operations=artifact_tasks,
        ),
        jobs=jobs,
        execution=ExecutionConfig() if execution is None else execution,
        artifact_settings=_artifact_settings(),
        runtime=runtime,
    )


def _run_paths(tmp_path: Path, run_id: str = "r1") -> RunPaths:
    run_root = tmp_path / "runs" / run_id
    return RunPaths(
        serve_root=tmp_path,
        run_id=run_id,
        run_root=run_root,
        dataset_dir=run_root / "dataset",
        metadata_path=run_root / "run.json",
    )


def _assert_preflight_rejected(request: BuildRunRequest | RuntimeRunRequest) -> None:
    with pytest.raises(ProfileCommandError):
        run_profiles(request)


def test_unexpected_planning_runtime_error_propagates(
    monkeypatch,
    tmp_path: Path,
) -> None:
    task = MetadataTask(id="metadata")
    request = _build_request(
        tmp_path,
        [task],
        [BuildJob(task, _artifact_settings())],
    )
    error = RuntimeError("unexpected planner bug")

    def fail(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(
        "datapipeline.profiles.orchestration.build_artifact_graph",
        fail,
    )

    with pytest.raises(RuntimeError) as raised:
        run_profiles(request)

    assert raised.value is error


def test_build_order_accepts_configured_dependency_order() -> None:
    series = SeriesTask(id="series")
    metadata = MetadataTask(id="metadata")
    coverage_stats = CoverageStatsTask(id="coverage_stats", stage="postprocessed")
    graph = build_artifact_graph([series, metadata, coverage_stats])

    _validate_build_order(
        [
            BuildJob(series, _artifact_settings()),
            BuildJob(metadata, _artifact_settings()),
            BuildJob(coverage_stats, _artifact_settings()),
        ],
        graph,
    )


def test_build_order_rejects_dependency_after_dependent() -> None:
    series = SeriesTask(id="series")
    metadata = MetadataTask(id="metadata")
    graph = build_artifact_graph([series, metadata])

    with pytest.raises(ValueError, match="series.*before.*metadata"):
        _validate_build_order(
            [
                BuildJob(metadata, _artifact_settings()),
                BuildJob(series, _artifact_settings()),
            ],
            graph,
        )


def test_build_order_rejects_duplicate_operations() -> None:
    metadata = MetadataTask(id="metadata")

    with pytest.raises(ValueError, match="unique artifact operations"):
        _validate_build_order(
            [
                BuildJob(metadata, _artifact_settings()),
                BuildJob(metadata, _artifact_settings()),
            ],
            build_artifact_graph([metadata]),
        )


def test_build_jobs_keep_order_and_share_resolved_artifacts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    series = SeriesTask(id="series")
    metadata = MetadataTask(id="metadata")
    coverage_stats = CoverageStatsTask(id="coverage_stats", stage="postprocessed")
    vector_runtime = _runtime(tmp_path, "vector-runtime")
    metadata_runtime = _runtime(tmp_path, "metadata-runtime")
    coverage_stats_runtime = _runtime(tmp_path, "coverage-stats-runtime")
    execution = ExecutionConfig(sort_buffer_mb=32)
    request = _build_request(
        tmp_path,
        [series, metadata, coverage_stats],
        [
            BuildJob(series, _artifact_settings()),
            BuildJob(metadata, _artifact_settings()),
            BuildJob(coverage_stats, _artifact_settings("FORCE")),
        ],
        execution,
    )
    calls: list[dict[str, object]] = []
    execution_runtimes: list[object] = []
    runtimes = iter((vector_runtime, metadata_runtime, coverage_stats_runtime))

    def build(_project, **kwargs):
        calls.append(dict(kwargs))
        kwargs["resolved_artifacts"].update(kwargs["required_artifacts"])

    @contextmanager
    def execute(runtime, _observability):
        execution_runtimes.append(runtime)
        yield

    monkeypatch.setattr(
        "datapipeline.profiles.orchestration.compile_runtime",
        lambda _definition: next(runtimes),
    )
    monkeypatch.setattr(
        "datapipeline.profiles.orchestration.execution_scope",
        execute,
    )
    monkeypatch.setattr(
        "datapipeline.profiles.orchestration.run_build_if_needed",
        build,
    )

    run_profiles(request)

    assert [call["required_artifacts"] for call in calls] == [
        {SERIES},
        {VECTOR_METADATA},
        {COVERAGE_STATS},
    ]
    assert [call["runtime"].marker for call in calls] == [
        "vector-runtime",
        "metadata-runtime",
        "coverage-stats-runtime",
    ]
    assert [call["settings"].mode for call in calls] == ["AUTO", "AUTO", "FORCE"]
    assert calls[0]["resolved_artifacts"] is calls[2]["resolved_artifacts"]
    assert calls[2]["resolved_artifacts"] == {
        SERIES,
        VECTOR_METADATA,
        COVERAGE_STATS,
    }
    assert execution_runtimes == [
        vector_runtime,
        metadata_runtime,
        coverage_stats_runtime,
    ]
    assert vector_runtime.execution == execution
    assert metadata_runtime.execution == execution
    assert coverage_stats_runtime.execution == execution


def test_runtime_artifact_union_is_prepared_once_before_jobs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    series = SeriesTask(id="series")
    metadata = MetadataTask(id="metadata")
    dataset_task = DatasetTask(id="dataset")
    first_runtime = _runtime(tmp_path, "first-job")
    second_runtime = _runtime(tmp_path, "second-job")
    canonical_runtime = _runtime(tmp_path, "canonical-build")
    execution = ExecutionConfig(sort_buffer_mb=32)
    artifact_settings = _artifact_settings("FORCE", 0)
    request = _runtime_request(
        tmp_path,
        command="serve",
        artifact_tasks=[series, metadata],
        jobs=[
            _runtime_job(
                "records-preview",
                dataset_task,
                first_runtime,
                preview="records",
            ),
            _runtime_job(
                "postprocess-preview",
                dataset_task,
                second_runtime,
                preview="postprocess",
            ),
        ],
        artifact_settings=artifact_settings,
        execution=execution,
    )
    events: list[tuple[str, object]] = []
    build_calls: list[dict[str, object]] = []
    execution_runtimes: list[object] = []

    def build(_project, **kwargs):
        events.append(("build", kwargs["runtime"]))
        build_calls.append(dict(kwargs))

    @contextmanager
    def execute(runtime, _observability):
        execution_runtimes.append(runtime)
        events.append(("execution started", runtime))
        try:
            yield
        finally:
            events.append(("execution finished", runtime))

    monkeypatch.setattr(
        "datapipeline.profiles.orchestration.compile_runtime",
        lambda _definition: canonical_runtime,
    )
    monkeypatch.setattr(
        "datapipeline.profiles.orchestration.run_build_if_needed",
        build,
    )
    monkeypatch.setattr(
        "datapipeline.profiles.orchestration.execution_scope",
        execute,
    )
    monkeypatch.setattr(
        "datapipeline.profiles.orchestration.execute_runtime_job",
        lambda _command, _project, _graph, plan: events.append(
            ("job", plan.job.runtime)
        ),
    )

    run_profiles(request)

    assert events == [
        ("execution started", canonical_runtime),
        ("build", canonical_runtime),
        ("execution finished", canonical_runtime),
        ("execution started", first_runtime),
        ("job", first_runtime),
        ("execution finished", first_runtime),
        ("execution started", second_runtime),
        ("job", second_runtime),
        ("execution finished", second_runtime),
    ]
    assert execution_runtimes == [
        canonical_runtime,
        first_runtime,
        second_runtime,
    ]
    assert len(build_calls) == 1
    assert build_calls[0]["required_artifacts"] == {
        SERIES,
        VECTOR_METADATA,
    }
    assert build_calls[0]["settings"] is artifact_settings
    assert canonical_runtime.execution == execution
    assert first_runtime.execution == execution
    assert second_runtime.execution == execution


def test_custom_runtime_artifact_requirement_is_prepared(
    monkeypatch,
    tmp_path: Path,
) -> None:
    snapshot = ArtifactTask(
        id="snapshot",
        entrypoint="plugin.snapshot",
        output="build/snapshot.json",
    )
    report = RuntimeTask(
        id="report",
        entrypoint="plugin.runtime.report",
        requires=("snapshot",),
    )
    request = _runtime_request(
        tmp_path,
        command="inspect",
        artifact_tasks=[snapshot],
        jobs=[_runtime_job("report", report, _runtime(tmp_path))],
    )
    build_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        "datapipeline.profiles.orchestration.compile_runtime",
        lambda _definition: _runtime(tmp_path, "artifact-build"),
    )
    monkeypatch.setattr(
        "datapipeline.profiles.orchestration.run_build_if_needed",
        lambda _project, **kwargs: build_calls.append(dict(kwargs)),
    )
    monkeypatch.setattr(
        "datapipeline.profiles.orchestration.execution_scope",
        _passthrough_execution_scope,
    )
    monkeypatch.setattr(
        "datapipeline.profiles.orchestration.execute_runtime_job",
        lambda *_args: None,
    )

    run_profiles(request)

    assert len(build_calls) == 1
    assert build_calls[0]["required_artifacts"] == {"snapshot"}


def test_custom_runtime_missing_required_producer_is_rejected_before_execution(
    tmp_path: Path,
) -> None:
    report = RuntimeTask(
        id="report",
        entrypoint="plugin.runtime.report",
        requires=("custom_snapshot",),
    )
    request = _runtime_request(
        tmp_path,
        command="inspect",
        artifact_tasks=[],
        jobs=[_runtime_job("report", report, _runtime(tmp_path))],
    )

    _assert_preflight_rejected(request)


def test_missing_runtime_artifact_producer_is_rejected_before_execution(
    tmp_path: Path,
) -> None:
    metadata = MetadataTask(id="metadata")
    dataset_task = DatasetTask(id="dataset")
    request = _runtime_request(
        tmp_path,
        command="serve",
        artifact_tasks=[metadata],
        jobs=[_runtime_job("serve", dataset_task, _runtime(tmp_path))],
    )

    _assert_preflight_rejected(request)


def test_v5_features_preview_is_rejected_before_starting_run(tmp_path: Path) -> None:
    dataset_task = DatasetTask(id="dataset")
    run_paths = _run_paths(tmp_path)
    request = _runtime_request(
        tmp_path,
        command="serve",
        artifact_tasks=[],
        jobs=[
            _runtime_job(
                "preview",
                dataset_task,
                _runtime(tmp_path),
                preview="features",  # type: ignore[arg-type]
            )
        ],
        serve_run_plans=(
            ServeRunPlan(run_paths, "features"),  # type: ignore[arg-type]
        ),
    )

    _assert_preflight_rejected(request)
    assert not run_paths.run_root.exists()


def test_runtime_jobs_keep_order_and_apply_execution_settings(
    monkeypatch,
    tmp_path: Path,
) -> None:
    task = DatasetTask(id="dataset")
    execution = ExecutionConfig(sort_buffer_mb=16)
    jobs = [
        _runtime_job(
            name,
            task,
            _runtime(tmp_path, name),
            heartbeat_interval_seconds=heartbeat,
            output_ids=output_ids,
        )
        for name, heartbeat, output_ids in (
            ("first", 10, ("train",)),
            ("second", 20, ("val",)),
            ("third", DEFAULT_HEARTBEAT_INTERVAL_SECONDS, ()),
        )
    ]
    request = _runtime_request(
        tmp_path,
        command="serve",
        artifact_tasks=[
            SeriesTask(id="series"),
            MetadataTask(id="metadata"),
        ],
        jobs=jobs,
        execution=execution,
    )
    observed: list[tuple[str, object, object, object]] = []
    execution_runtimes: list[object] = []

    @contextmanager
    def execute(runtime, _observability):
        execution_runtimes.append(runtime)
        yield

    def execute_job(_command, _project, _graph, plan):
        runtime = plan.job.runtime
        observed.append(
            (
                plan.job.name,
                runtime.heartbeat_interval_seconds,
                plan.job.output_ids,
                runtime.execution,
            )
        )

    monkeypatch.setattr(
        "datapipeline.profiles.orchestration.execution_scope",
        execute,
    )
    monkeypatch.setattr(
        "datapipeline.profiles.orchestration.execute_runtime_job",
        execute_job,
    )
    monkeypatch.setattr(
        "datapipeline.profiles.orchestration._prepare_runtime_artifacts",
        lambda *_args: None,
    )

    run_profiles(request)

    assert observed == [
        ("first", 10, ("train",), execution),
        ("second", 20, ("val",), execution),
        ("third", DEFAULT_HEARTBEAT_INTERVAL_SECONDS, (), execution),
    ]
    assert execution_runtimes == [job.runtime for job in jobs]


def test_runtime_job_emits_resolved_config_at_debug(
    monkeypatch, tmp_path: Path
) -> None:
    runtime = _runtime(tmp_path)
    runtime.execution = ExecutionConfig(sort_buffer_mb=24)
    task = RuntimeTask(id="report", entrypoint="plugin.runtime.report")
    job = _runtime_job("coverage", task, runtime)
    messages: list[tuple[str, int]] = []

    monkeypatch.setattr(
        "datapipeline.profiles.execution.hydrate_runtime_artifacts_for_pipeline",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(
        "datapipeline.profiles.execution.emit_execution_message",
        lambda message, level: messages.append((message, level)),
    )
    monkeypatch.setattr(
        "datapipeline.profiles.execution.load_ep",
        lambda *_args: lambda *_runner_args: None,
    )

    execute_runtime_job(
        "inspect",
        project_definition(tmp_path / "project.yaml", dataset=runtime.dataset),
        build_artifact_graph([]),
        RuntimeJobPlan(job, ()),
    )

    assert len(messages) == 1
    message, level = messages[0]
    assert level == logging.DEBUG
    assert message.startswith("Config:\n")
    config = json.loads(message.removeprefix("Config:\n"))
    assert config["operation"]["id"] == "report"
    assert config["operation"]["entrypoint"] == "plugin.runtime.report"
    assert "dataset" not in config
    assert config["execution"]["sort_buffer_mb"] == 24
    assert config["observability"]["log_level"] == "INFO"


def test_runtime_job_does_not_hide_plugin_value_errors(
    monkeypatch, tmp_path: Path
) -> None:
    task = RuntimeTask(id="report", entrypoint="plugin.runtime.report")
    job = _runtime_job("coverage", task, _runtime(tmp_path))
    monkeypatch.setattr(
        "datapipeline.profiles.execution.hydrate_runtime_artifacts_for_pipeline",
        lambda *_args, **_kwargs: (),
    )

    def fail(*_args):
        def run(*_runner_args):
            raise ValueError("plugin bug")

        return run

    monkeypatch.setattr("datapipeline.profiles.execution.load_ep", fail)

    with pytest.raises(ValueError, match="plugin bug"):
        execute_runtime_job(
            "inspect",
            project_definition(tmp_path / "project.yaml"),
            build_artifact_graph([]),
            RuntimeJobPlan(job, ()),
        )


def test_runtime_job_reports_unavailable_artifacts(monkeypatch, tmp_path: Path) -> None:
    task = RuntimeTask(id="report", entrypoint="plugin.runtime.report")
    job = _runtime_job("report", task, _runtime(tmp_path))
    monkeypatch.setattr(
        "datapipeline.profiles.execution.hydrate_runtime_artifacts_for_pipeline",
        lambda *_args, **_kwargs: (),
    )

    with pytest.raises(
        ArtifactResolutionError,
        match=(
            "Runtime operation 'report' requires missing or stale artifacts: snapshot"
        ),
    ):
        execute_runtime_job(
            "inspect",
            project_definition(tmp_path / "project.yaml"),
            build_artifact_graph([]),
            RuntimeJobPlan(job, ("snapshot",)),
        )


def test_runtime_plugin_receives_the_documented_contract(
    monkeypatch, tmp_path: Path
) -> None:
    task = RuntimeTask(id="report", entrypoint="plugin.runtime.report")
    job = _runtime_job("report", task, _runtime(tmp_path), limit=7)
    received = None

    def load_runner(group, entrypoint):
        assert group == "datapipeline.operations.runtime"
        assert entrypoint == task.entrypoint

        def run(runtime, operation_task, limit):
            nonlocal received
            received = runtime, operation_task, limit
            return "result"

        return run

    monkeypatch.setattr(
        "datapipeline.profiles.execution.load_ep",
        load_runner,
    )

    assert run_runtime_operation(job) == "result"
    assert received == (job.runtime, task, 7)


def test_dataset_operation_uses_its_core_runner(monkeypatch, tmp_path: Path) -> None:
    task = DatasetTask(id="dataset")
    job = _runtime_job(
        "dataset",
        task,
        _runtime(tmp_path),
        limit=5,
        output_ids=("holdout.train",),
    )
    received = None

    def run_dataset(
        runtime,
        output_ids,
        limit,
        target,
        throttle_ms,
        preview,
    ):
        nonlocal received
        received = runtime, output_ids, limit, target, throttle_ms, preview
        return "dataset"

    monkeypatch.setattr(
        "datapipeline.profiles.execution.run_dataset_operation",
        run_dataset,
    )
    monkeypatch.setattr(
        "datapipeline.profiles.execution.load_ep",
        lambda *_args: pytest.fail("core operations must not load plugin entry points"),
    )

    assert run_runtime_operation(job) == "dataset"
    assert received == (job.runtime, job.output_ids, 5, job.output, None, None)


def test_matrix_operation_uses_its_core_runner(monkeypatch, tmp_path: Path) -> None:
    task = MatrixTask(id="matrix")
    job = _runtime_job("matrix", task, _runtime(tmp_path), limit=5)
    received = None

    def run_matrix(runtime, operation_task, limit):
        nonlocal received
        received = runtime, operation_task, limit
        return "matrix"

    monkeypatch.setattr(
        "datapipeline.profiles.execution.run_matrix_operation",
        run_matrix,
    )
    monkeypatch.setattr(
        "datapipeline.profiles.execution.load_ep",
        lambda *_args: pytest.fail("core operations must not load plugin entry points"),
    )

    assert run_runtime_operation(job) == "matrix"
    assert received == (job.runtime, task, 5)


def test_coverage_operation_rejects_limit_before_planning(tmp_path: Path) -> None:
    job = _runtime_job(
        "coverage",
        CoverageTask(id="coverage"),
        _runtime(tmp_path),
        limit=1,
    )

    with pytest.raises(ValueError, match="coverage operation does not support"):
        plan_runtime_job(job, SimpleNamespace(), SimpleNamespace())


def test_parquet_output_rejects_non_dataset_operation_before_planning(
    tmp_path: Path,
) -> None:
    job = _runtime_job(
        "matrix",
        MatrixTask(id="matrix"),
        _runtime(tmp_path),
        output=OutputTarget(
            transport="fs",
            format="parquet",
            view="flat",
            encoding=None,
            destination=tmp_path / "matrix.parquet",
        ),
    )

    with pytest.raises(ValueError, match="only by the dataset operation"):
        plan_runtime_job(job, SimpleNamespace(), SimpleNamespace())


def test_parquet_output_rejects_record_preview_before_planning(
    tmp_path: Path,
) -> None:
    job = _runtime_job(
        "dataset",
        DatasetTask(id="dataset"),
        _runtime(tmp_path),
        preview="records",
        output=OutputTarget(
            transport="fs",
            format="parquet",
            view="flat",
            encoding=None,
            destination=tmp_path / "records.parquet",
        ),
    )

    with pytest.raises(ValueError, match="only 'samples' and 'postprocess'"):
        plan_runtime_job(job, SimpleNamespace(), SimpleNamespace())


def test_shared_serve_run_is_finalized_once(monkeypatch, tmp_path: Path) -> None:
    task = RuntimeTask(id="pipeline", entrypoint="plugin.runtime")
    run_paths = _run_paths(tmp_path)
    request = _runtime_request(
        tmp_path,
        command="serve",
        artifact_tasks=[],
        jobs=[
            _runtime_job(name, task, _runtime(tmp_path, name))
            for name in ("train", "val")
        ],
        serve_run_plans=(ServeRunPlan(run_paths, None),),
    )
    calls = {"start": 0, "success": 0, "failed": 0, "latest": 0}
    monkeypatch.setattr(
        "datapipeline.profiles.orchestration.execution_scope",
        _passthrough_execution_scope,
    )
    monkeypatch.setattr(
        "datapipeline.profiles.orchestration.execute_runtime_job",
        lambda *_args: None,
    )
    for name in calls:
        function = "set_latest_run" if name == "latest" else f"{name}_run"
        if name == "success":
            function = "finish_run_success"
        elif name == "failed":
            function = "finish_run_failed"
        monkeypatch.setattr(
            f"datapipeline.profiles.orchestration.{function}",
            lambda *_args, key=name, **_kwargs: calls.__setitem__(key, calls[key] + 1),
        )

    run_profiles(request)

    assert calls == {"start": 1, "success": 1, "failed": 0, "latest": 1}


def test_series_cache_filesystem_failure_does_not_fail_published_run(
    monkeypatch,
    tmp_path: Path,
    caplog,
) -> None:
    series = SeriesTask(id="series")
    task = RuntimeTask(id="pipeline", entrypoint="plugin.runtime")
    run_paths = _run_paths(tmp_path)
    request = _runtime_request(
        tmp_path,
        command="serve",
        artifact_tasks=[series],
        jobs=[_runtime_job("serve", task, _runtime(tmp_path))],
        serve_run_plans=(ServeRunPlan(run_paths, None),),
    )

    monkeypatch.setattr(
        "datapipeline.profiles.orchestration.execution_scope",
        _passthrough_execution_scope,
    )
    monkeypatch.setattr(
        "datapipeline.profiles.orchestration.execute_runtime_job",
        lambda *_args: None,
    )

    def fail_pruning(*_args) -> None:
        raise PermissionError("cache is read-only")

    monkeypatch.setattr(
        "datapipeline.profiles.orchestration.prune_series_cache",
        fail_pruning,
    )

    with caplog.at_level(
        logging.WARNING,
        logger="datapipeline.profiles.orchestration",
    ):
        run_profiles(request)

    metadata = json.loads(run_paths.metadata_path.read_text(encoding="utf-8"))
    assert metadata["status"] == "success"
    assert (run_paths.serve_root / "latest").resolve() == run_paths.run_root.resolve()
    assert "Series cache cleanup skipped" in caplog.text
    assert "cache is read-only" in caplog.text


def test_series_cache_programming_error_remains_strict(
    monkeypatch,
    tmp_path: Path,
) -> None:
    series = SeriesTask(id="series")
    request = _build_request(
        tmp_path,
        [series],
        [],
    )
    error = RuntimeError("invalid cache layout")

    def fail_pruning(*_args) -> None:
        raise error

    monkeypatch.setattr(
        "datapipeline.profiles.orchestration.prune_series_cache",
        fail_pruning,
    )

    with pytest.raises(RuntimeError, match="invalid cache layout") as raised:
        _prune_series_caches(request)

    assert raised.value is error


def test_job_failure_marks_shared_run_failed(monkeypatch, tmp_path: Path) -> None:
    task = RuntimeTask(id="pipeline", entrypoint="plugin.runtime")
    run_paths = _run_paths(tmp_path)
    request = _runtime_request(
        tmp_path,
        command="serve",
        artifact_tasks=[],
        jobs=[
            _runtime_job(name, task, _runtime(tmp_path, name))
            for name in ("train", "val")
        ],
        serve_run_plans=(ServeRunPlan(run_paths, None),),
    )
    calls = 0
    failed: list[RunPaths] = []

    def execute(*_args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("boom")

    monkeypatch.setattr(
        "datapipeline.profiles.orchestration.execution_scope",
        _passthrough_execution_scope,
    )
    monkeypatch.setattr(
        "datapipeline.profiles.orchestration.execute_runtime_job",
        execute,
    )
    monkeypatch.setattr(
        "datapipeline.profiles.orchestration.start_run",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "datapipeline.profiles.orchestration.finish_run_failed",
        failed.append,
    )

    with pytest.raises(RuntimeError, match="boom"):
        run_profiles(request)

    assert failed == [run_paths]


def test_cleanup_failure_does_not_replace_job_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    task = RuntimeTask(id="pipeline", entrypoint="plugin.runtime")
    run_paths = _run_paths(tmp_path)
    request = _runtime_request(
        tmp_path,
        command="serve",
        artifact_tasks=[],
        jobs=[_runtime_job("serve", task, _runtime(tmp_path))],
        serve_run_plans=(ServeRunPlan(run_paths, None),),
    )
    job_error = RuntimeError("job failed")

    def fail_job(*_args) -> None:
        raise job_error

    def fail_cleanup(_paths: RunPaths) -> None:
        raise OSError("cleanup failed")

    monkeypatch.setattr(
        "datapipeline.profiles.orchestration.execution_scope",
        _passthrough_execution_scope,
    )
    monkeypatch.setattr(
        "datapipeline.profiles.orchestration.execute_runtime_job",
        fail_job,
    )
    monkeypatch.setattr(
        "datapipeline.profiles.orchestration.finish_run_failed",
        fail_cleanup,
    )

    with pytest.raises(RuntimeError, match="job failed") as raised:
        run_profiles(request)

    assert raised.value is job_error
    assert raised.value.__notes__ == [
        "Failed to finalize serve run 'r1': cleanup failed"
    ]


def test_latest_failure_still_finalizes_all_runs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    task = RuntimeTask(id="pipeline", entrypoint="plugin.runtime")
    first = _run_paths(tmp_path / "first", "first")
    second = _run_paths(tmp_path / "second", "second")
    request = _runtime_request(
        tmp_path,
        command="serve",
        artifact_tasks=[],
        jobs=[_runtime_job("serve", task, _runtime(tmp_path))],
        serve_run_plans=(ServeRunPlan(first, None), ServeRunPlan(second, None)),
    )
    latest: list[RunPaths] = []

    monkeypatch.setattr(
        "datapipeline.profiles.orchestration.execution_scope",
        _passthrough_execution_scope,
    )
    monkeypatch.setattr(
        "datapipeline.profiles.orchestration.execute_runtime_job",
        lambda *_args: None,
    )

    def set_latest(paths: RunPaths) -> None:
        latest.append(paths)
        if paths == first:
            raise OSError("latest failed")
        raise OSError("second latest failed")

    monkeypatch.setattr(
        "datapipeline.profiles.orchestration.set_latest_run",
        set_latest,
    )

    with pytest.raises(OSError, match="latest failed") as raised:
        run_profiles(request)

    metadata = [
        json.loads(paths.metadata_path.read_text(encoding="utf-8"))
        for paths in (first, second)
    ]
    assert [item["status"] for item in metadata] == ["success", "success"]
    assert all(item["finished_at"] is not None for item in metadata)
    assert latest == [first, second]
    assert raised.value.__notes__ == [
        "Also failed to finalize serve run 'second': second latest failed"
    ]


def test_later_output_commit_failure_marks_run_failed_and_preserves_latest(
    monkeypatch,
    tmp_path: Path,
) -> None:
    serve_root = tmp_path / "served"
    previous_paths = _run_paths(serve_root, "previous")
    current_paths = _run_paths(serve_root, "current")
    start_run(previous_paths)
    finish_run_success(previous_paths)
    set_latest_run(previous_paths)

    first_output = current_paths.dataset_dir / "first.jsonl"
    blocked_output = current_paths.dataset_dir / "blocked.jsonl"
    blocked_output.mkdir(parents=True)
    result = RuntimeOutputBatch(
        outputs=(
            RoutedRuntimeOutput(
                rows=(
                    ("first", {"output": "first", "value": 1}),
                    ("blocked", {"output": "blocked", "value": 2}),
                ),
                targets={
                    "first": OutputTarget(
                        transport="fs",
                        format="jsonl",
                        view="raw",
                        encoding="utf-8",
                        destination=first_output,
                    ),
                    "blocked": OutputTarget(
                        transport="fs",
                        format="jsonl",
                        view="raw",
                        encoding="utf-8",
                        destination=blocked_output,
                    ),
                },
            ),
        )
    )
    task = RuntimeTask(id="pipeline", entrypoint="plugin.runtime")
    request = _runtime_request(
        tmp_path,
        command="serve",
        artifact_tasks=[],
        jobs=[_runtime_job("dataset", task, _runtime(tmp_path))],
        serve_run_plans=(ServeRunPlan(current_paths, None),),
    )
    monkeypatch.setattr(
        "datapipeline.profiles.orchestration.execution_scope",
        _passthrough_execution_scope,
    )
    monkeypatch.setattr(
        "datapipeline.profiles.execution.load_ep",
        lambda *_args: lambda *_plugin_args: result,
    )

    with pytest.raises(OSError):
        run_profiles(request)

    current_run = json.loads(current_paths.metadata_path.read_text(encoding="utf-8"))
    assert current_run["status"] == "failed"
    assert current_run["finished_at"] is not None
    assert first_output.read_text(encoding="utf-8") == (
        '{"output": "first", "value": 1}\n'
    )
    assert (serve_root / "latest").resolve() == previous_paths.run_root.resolve()


def test_artifact_resolution_failure_becomes_profile_error(
    monkeypatch,
    tmp_path: Path,
) -> None:
    request = _runtime_request(
        tmp_path,
        command="inspect",
        artifact_tasks=[],
        jobs=[],
    )

    def fail(_request) -> None:
        error = ArtifactResolutionError("required artifact is unavailable")
        error.add_note("failed to finalize run")
        raise error

    monkeypatch.setattr(
        "datapipeline.profiles.orchestration._run_runtime_profiles",
        fail,
    )

    with pytest.raises(
        ProfileCommandError,
        match="required artifact is unavailable",
    ) as raised:
        run_profiles(request)

    assert isinstance(raised.value.__cause__, ArtifactResolutionError)
    assert raised.value.__notes__ == ["failed to finalize run"]


def test_preview_run_exists_at_job_boundary_and_is_not_latest(
    monkeypatch,
    tmp_path: Path,
) -> None:
    dataset_task = DatasetTask(id="dataset")
    run_paths = _run_paths(tmp_path)
    runtime = _runtime(tmp_path)
    request = _runtime_request(
        tmp_path,
        command="serve",
        artifact_tasks=[],
        jobs=[
            _runtime_job(
                "preview",
                dataset_task,
                runtime,
                preview="records",
                heartbeat_interval_seconds=15,
            )
        ],
        serve_run_plans=(ServeRunPlan(run_paths, "records"),),
    )
    observed: dict[str, object] = {}

    @contextmanager
    def execute(runtime, _observability):
        run = json.loads(run_paths.metadata_path.read_text(encoding="utf-8"))
        observed.update(
            status=run["status"],
            preview=run["preview"],
            heartbeat=runtime.heartbeat_interval_seconds,
        )
        yield

    monkeypatch.setattr(
        "datapipeline.profiles.orchestration.execution_scope",
        execute,
    )
    monkeypatch.setattr(
        "datapipeline.profiles.orchestration.execute_runtime_job",
        lambda *_args: None,
    )
    latest: list[RunPaths] = []
    monkeypatch.setattr(
        "datapipeline.profiles.orchestration.set_latest_run",
        latest.append,
    )

    run_profiles(request)

    assert observed == {
        "status": "running",
        "preview": "records",
        "heartbeat": 15,
    }
    finished = json.loads(run_paths.metadata_path.read_text(encoding="utf-8"))
    assert finished["status"] == "success"
    assert latest == []


def test_later_run_start_failure_fails_only_started_run(
    monkeypatch,
    tmp_path: Path,
) -> None:
    task = RuntimeTask(id="pipeline", entrypoint="plugin.runtime")
    first = _run_paths(tmp_path / "first")
    second = _run_paths(tmp_path / "second")
    request = _runtime_request(
        tmp_path,
        command="serve",
        artifact_tasks=[],
        jobs=[_runtime_job("serve", task, _runtime(tmp_path))],
        serve_run_plans=(ServeRunPlan(first, None), ServeRunPlan(second, None)),
    )
    starts: list[RunPaths] = []
    failed: list[RunPaths] = []

    def start(paths, *, preview):
        starts.append(paths)
        if paths == second:
            raise RuntimeError("cannot start second run")

    monkeypatch.setattr("datapipeline.profiles.orchestration.start_run", start)
    monkeypatch.setattr(
        "datapipeline.profiles.orchestration.finish_run_failed",
        failed.append,
    )

    with pytest.raises(RuntimeError, match="cannot start second run"):
        run_profiles(request)

    assert starts == [first, second]
    assert failed == [first]


def test_materialize_uses_shared_artifact_and_execution_lifecycle(
    monkeypatch,
    tmp_path: Path,
) -> None:
    schedule = ScheduleTask(
        id="market_schedule",
        stream="prices",
        partition_by=[],
        output="schedule.jsonl",
    )
    execution = ExecutionConfig(sort_buffer_mb=32)
    runtime = SimpleNamespace(
        execution=ExecutionConfig(),
        heartbeat_interval_seconds=None,
        streams={"adv.20": object(), "adv.63": object()},
        artifacts_root=tmp_path / "artifacts",
    )
    jobs = [
        MaterializeJob(
            name="adv-20",
            stream="adv.20",
            output=_materialize_output(tmp_path / "adv-20.jsonl"),
            overwrite=False,
            observability=_observability(10),
        ),
        MaterializeJob(
            name="adv-63",
            stream="adv.63",
            output=_materialize_output(tmp_path / "adv-63.jsonl"),
            overwrite=False,
            observability=_observability(20),
        ),
    ]
    request = _materialize_request(
        tmp_path,
        [schedule],
        jobs,
        runtime,
        execution,
    )
    monkeypatch.setattr(
        "datapipeline.artifacts.planning.stream_schedule_artifacts",
        lambda stream, streams: {"market_schedule"},
    )
    build_calls: list[dict] = []
    materialized: list[tuple[str, float | None]] = []

    def build(project_path, **kwargs):
        assert runtime.execution == execution
        build_calls.append(kwargs)

    monkeypatch.setattr(
        "datapipeline.profiles.orchestration.run_build_if_needed",
        build,
    )
    monkeypatch.setattr(
        "datapipeline.profiles.orchestration.execution_scope",
        _passthrough_execution_scope,
    )
    monkeypatch.setattr(
        "datapipeline.profiles.orchestration.execute_materialize_job",
        lambda job, active_runtime: materialized.append(
            (job.name, active_runtime.heartbeat_interval_seconds)
        ),
    )

    run_profiles(request)

    assert len(build_calls) == 1
    assert build_calls[0]["required_artifacts"] == {"market_schedule"}
    assert materialized == [("adv-20", 10), ("adv-63", 20)]


@pytest.mark.parametrize("mode", ["AUTO", "OFF"])
def test_materialize_hydrates_current_schedule_when_build_skips(
    monkeypatch,
    tmp_path: Path,
    mode: str,
) -> None:
    schedule = ScheduleTask(
        id="market_schedule",
        stream="prices",
        partition_by=[],
        output="schedule.jsonl",
    )
    job = MaterializeJob(
        name="adv-20",
        stream="adv.20",
        output=_materialize_output(tmp_path / "adv-20.jsonl"),
        overwrite=False,
        observability=_observability(),
    )
    definition = project_definition(
        tmp_path / "project.yaml",
        dataset=_dataset(),
        streams=_stream_catalog(),
        artifact_operations=[schedule],
    )
    artifacts_root = definition.project.artifacts_root
    artifact_path = artifacts_root / schedule.output
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text("{}\n", encoding="utf-8")
    state = BuildState()
    state.register(
        schedule.id,
        artifact_hash=definition.artifact_hashes.for_artifact(schedule.id),
        files=(ArtifactFileFingerprint.from_path(schedule.output, artifact_path),),
    )
    save_build_state(
        state,
        artifacts_root,
    )
    runtime = SimpleNamespace(
        execution=ExecutionConfig(),
        heartbeat_interval_seconds=None,
        streams={"adv.20": object()},
        artifacts=ArtifactRegistry(artifacts_root),
        artifacts_root=artifacts_root,
    )
    request = MaterializeRunRequest(
        definition=definition,
        jobs=[job],
        execution=ExecutionConfig(),
        artifact_settings=_artifact_settings(mode),
        runtime=runtime,
    )
    monkeypatch.setattr(
        "datapipeline.artifacts.planning.stream_schedule_artifacts",
        lambda stream, streams: {"market_schedule"},
    )
    monkeypatch.setattr(
        "datapipeline.profiles.orchestration.execution_scope",
        _passthrough_execution_scope,
    )
    monkeypatch.setattr(
        "datapipeline.profiles.orchestration.execute_materialize_job",
        lambda job, active_runtime: active_runtime.artifacts.require("market_schedule"),
    )

    run_profiles(request)

    assert runtime.artifacts.has("market_schedule")


@pytest.mark.parametrize(
    ("artifact_tasks", "message"),
    [
        ([], "requires a declared schedule operation"),
        (
            [
                ArtifactTask(
                    id="market_schedule",
                    entrypoint="plugin.snapshot",
                    output="snapshot.json",
                )
            ],
            "not a schedule operation",
        ),
    ],
)
def test_materialize_rejects_invalid_schedule_artifact_producer(
    monkeypatch,
    tmp_path: Path,
    artifact_tasks,
    message,
) -> None:
    runtime = SimpleNamespace(
        execution=ExecutionConfig(),
        heartbeat_interval_seconds=None,
        streams={"adv.20": object()},
        artifacts_root=tmp_path / "artifacts",
    )
    job = MaterializeJob(
        name="adv-20",
        stream="adv.20",
        output=_materialize_output(tmp_path / "adv-20.jsonl"),
        overwrite=False,
        observability=_observability(),
    )
    request = _materialize_request(tmp_path, artifact_tasks, [job], runtime)
    monkeypatch.setattr(
        "datapipeline.artifacts.planning.stream_schedule_artifacts",
        lambda stream, streams: {"market_schedule"},
    )

    with pytest.raises(ProfileCommandError, match=message):
        run_profiles(request)
