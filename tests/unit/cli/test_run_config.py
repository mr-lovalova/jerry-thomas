from pathlib import Path

import pytest

from jerrythomas.execution.settings import CommandObservability
from jerrythomas.profiles.errors import ProfileCommandError
from jerrythomas.profiles.request_builder import (
    build_build_run_request,
    build_runtime_run_request,
)


def _write_project(tmp_path: Path) -> Path:
    project_yaml = tmp_path / "project.yaml"
    project_yaml.write_text(
        "\n".join(
            [
                "schema_version: 6",
                "artifact_revision: 1",
                "paths:",
                "  streams: streams",
                "  sources: sources",
                "  dataset: dataset.yaml",
                "  artifacts: artifacts",
                "  profiles: profiles",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "dataset.yaml").write_text(
        "sample:\n  rounding: ceil\n  cadence: 1h\n",
        encoding="utf-8",
    )
    for directory in ("streams", "sources"):
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    return project_yaml


def test_serve_request_resolves_named_profile(tmp_path: Path):
    project_yaml = _write_project(tmp_path)
    profiles = tmp_path / "profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    (profiles / "serve.coverage.yaml").write_text(
        "operation: coverage\n",
        encoding="utf-8",
    )

    request = build_runtime_run_request(
        command="serve",
        project=str(project_yaml),
        profile_name="coverage",
        limit=5,
    )
    assert request is not None
    assert request.command == "serve"
    assert len(request.jobs) == 1
    job = request.jobs[0]
    assert job.name == "coverage"
    assert job.task.id == "coverage"
    assert job.limit == 5
    assert request.artifact_settings is not None
    assert request.artifact_settings.mode == "auto"


def test_inspect_request_defaults_to_enabled_profiles(tmp_path: Path):
    project_yaml = _write_project(tmp_path)
    profiles = tmp_path / "profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    (profiles / "inspect.coverage.yaml").write_text(
        "operation: coverage\nenabled: false\n",
        encoding="utf-8",
    )
    (profiles / "inspect.matrix.yaml").write_text(
        "operation: matrix\nenabled: true\n",
        encoding="utf-8",
    )

    request = build_runtime_run_request(
        command="inspect",
        project=str(project_yaml),
        limit=7,
    )
    assert request is not None
    assert request.command == "inspect"
    assert len(request.jobs) == 1
    job = request.jobs[0]
    assert job.name == "matrix"
    assert job.task.id == "matrix"
    assert job.limit == 7


def test_inspect_request_rejects_preview(tmp_path: Path) -> None:
    project_yaml = _write_project(tmp_path)
    profiles = tmp_path / "profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    (profiles / "inspect.coverage.yaml").write_text(
        "operation: coverage\n",
        encoding="utf-8",
    )

    with pytest.raises(ProfileCommandError) as exc:
        build_runtime_run_request(
            command="inspect",
            project=str(project_yaml),
            preview="input",
        )

    assert "Inspect profiles do not support previews" in str(exc.value)


def test_runtime_request_rejects_stdout_data_and_logging(tmp_path: Path) -> None:
    project_yaml = _write_project(tmp_path)
    profiles = tmp_path / "profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    (profiles / "serve.coverage.yaml").write_text(
        (
            "operation: coverage\n"
            "observability:\n"
            "  logging:\n"
            "    outputs:\n"
            "      - transport: stdout\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ProfileCommandError, match="cannot both use stdout"):
        build_runtime_run_request(
            command="serve",
            project=str(project_yaml),
        )


def test_runtime_request_rejects_cross_profile_output_log_collision(
    tmp_path: Path,
) -> None:
    project_yaml = _write_project(tmp_path)
    profiles = tmp_path / "profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    (profiles / "inspect.first.yaml").write_text(
        (
            "operation: coverage\n"
            "output:\n"
            "  transport: fs\n"
            "  format: jsonl\n"
            "  directory: out\n"
        ),
        encoding="utf-8",
    )
    (profiles / "inspect.second.yaml").write_text(
        (
            "operation: matrix\n"
            "output:\n"
            "  transport: fs\n"
            "  format: jsonl\n"
            "  directory: out\n"
            "observability:\n"
            "  logging:\n"
            "    outputs:\n"
            "      - transport: fs\n"
            "        path: out/first/first.jsonl\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ProfileCommandError, match="same path"):
        build_runtime_run_request(
            command="inspect",
            project=str(project_yaml),
        )


def test_runtime_request_rejects_cross_profile_output_collision(
    tmp_path: Path,
) -> None:
    project_yaml = _write_project(tmp_path)
    profiles = tmp_path / "profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    profile = (
        "operation: coverage\n"
        "output:\n"
        "  transport: fs\n"
        "  format: jsonl\n"
        "  directory: out\n"
        "  filename: shared\n"
    )
    (profiles / "serve.first.yaml").write_text(profile, encoding="utf-8")
    (profiles / "serve.second.yaml").write_text(profile, encoding="utf-8")

    with pytest.raises(ProfileCommandError, match="same path"):
        build_runtime_run_request(
            command="serve",
            project=str(project_yaml),
        )


def test_runtime_request_rejects_log_at_latest_run_pointer(tmp_path: Path) -> None:
    project_yaml = _write_project(tmp_path)
    profiles = tmp_path / "profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    (profiles / "serve.coverage.yaml").write_text(
        (
            "operation: coverage\n"
            "output:\n"
            "  transport: fs\n"
            "  format: jsonl\n"
            "  directory: served\n"
            "observability:\n"
            "  logging:\n"
            "    outputs:\n"
            "      - transport: fs\n"
            "        path: served/latest\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ProfileCommandError, match="managed serve run paths"):
        build_runtime_run_request(
            command="serve",
            project=str(project_yaml),
        )


def test_runtime_request_rejects_colliding_routed_output_ids(
    tmp_path: Path,
) -> None:
    project_yaml = _write_project(tmp_path)
    (tmp_path / "dataset.yaml").write_text(
        (
            "sample: {rounding: ceil, cadence: 1h}\n"
            "split:\n"
            "  mode: hash\n"
            "  ratios: {train: 1.0}\n"
            "  folds:\n"
            "    - {id: north/west, train: [train]}\n"
            "    - {id: north_west, train: [train]}\n"
        ),
        encoding="utf-8",
    )
    profiles = tmp_path / "profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    (profiles / "serve.dataset.yaml").write_text(
        (
            "operation: dataset\n"
            "include_outputs: [north/west.train, north_west.train]\n"
            "output:\n"
            "  transport: fs\n"
            "  format: jsonl\n"
            "  directory: output\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ProfileCommandError, match="same path"):
        build_runtime_run_request(
            command="serve",
            project=str(project_yaml),
        )


def test_runtime_request_rejects_colliding_preview_output_ids(
    tmp_path: Path,
) -> None:
    project_yaml = _write_project(tmp_path)
    (tmp_path / "sources" / "source.yaml").write_text(
        (
            "id: source\n"
            "parser: {entrypoint: core.temporal_record}\n"
            "loader:\n"
            "  entrypoint: core.synthetic.ticks\n"
            "  args:\n"
            '    start: "2024-01-01T00:00:00Z"\n'
            '    end: "2024-01-02T00:00:00Z"\n'
            "    frequency: 1h\n"
            "freshness: opaque\n"
        ),
        encoding="utf-8",
    )
    for stream_id in ("first", "second"):
        (tmp_path / "streams" / f"{stream_id}.yaml").write_text(
            (
                f"id: {stream_id}\n"
                "from: {source: source}\n"
                "map: {entrypoint: identity}\n"
            ),
            encoding="utf-8",
        )
    (tmp_path / "dataset.yaml").write_text(
        (
            "sample: {rounding: ceil, cadence: 1h}\n"
            "features:\n"
            '  - {id: "a/b", stream: first, field: value}\n'
            '  - {id: "a?b", stream: second, field: value}\n'
        ),
        encoding="utf-8",
    )
    profiles = tmp_path / "profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    (profiles / "serve.dataset.yaml").write_text(
        (
            "operation: dataset\n"
            "preview: series\n"
            "output:\n"
            "  transport: fs\n"
            "  format: jsonl\n"
            "  directory: output\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ProfileCommandError, match="same path"):
        build_runtime_run_request(
            command="serve",
            project=str(project_yaml),
        )


@pytest.mark.parametrize(
    "log_path",
    (
        "artifacts/build/metadata.json",
        "artifacts/build",
        "artifacts/_system/build/state.json",
        "ARTIFACTS/logs/build.log",
    ),
)
def test_build_request_rejects_logs_under_artifacts_root(
    tmp_path: Path,
    log_path: str,
) -> None:
    project_yaml = _write_project(tmp_path)
    profiles = tmp_path / "profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    (profiles / "build.metadata.yaml").write_text(
        (
            "operation: metadata\n"
            "observability:\n"
            "  logging:\n"
            "    outputs:\n"
            "      - transport: fs\n"
            f"        path: {log_path}\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ProfileCommandError, match="outside the artifacts root"):
        build_build_run_request(project=str(project_yaml))


def test_runtime_request_allows_profiles_to_share_a_log(tmp_path: Path) -> None:
    project_yaml = _write_project(tmp_path)
    profiles = tmp_path / "profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    for name, operation in (("first", "coverage"), ("second", "matrix")):
        (profiles / f"inspect.{name}.yaml").write_text(
            (
                f"operation: {operation}\n"
                "output:\n"
                "  transport: fs\n"
                "  format: jsonl\n"
                "  directory: output\n"
                "observability:\n"
                "  logging:\n"
                "    outputs:\n"
                "      - transport: fs\n"
                "        path: logs/shared.log\n"
            ),
            encoding="utf-8",
        )

    request = build_runtime_run_request(
        command="inspect",
        project=str(project_yaml),
    )

    assert request is not None
    assert len(request.jobs) == 2
    assert {
        job.observability.log_output.outputs[0].destination for job in request.jobs
    } == {tmp_path / "logs" / "shared.log"}


def test_serve_profile_rejects_artifact_operation(tmp_path: Path):
    project_yaml = _write_project(tmp_path)
    profiles = tmp_path / "profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    (profiles / "serve.metadata.yaml").write_text(
        "operation: metadata\n",
        encoding="utf-8",
    )

    with pytest.raises(ProfileCommandError) as exc:
        build_runtime_run_request(
            command="serve",
            project=str(project_yaml),
            profile_name="metadata",
        )

    assert (
        "must reference a runtime operation; 'metadata' is an artifact operation"
        in str(exc.value)
    )


def test_inspect_profile_rejects_artifact_operation(tmp_path: Path):
    project_yaml = _write_project(tmp_path)
    profiles = tmp_path / "profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    (profiles / "inspect.coverage_stats.yaml").write_text(
        "operation: coverage_stats\n",
        encoding="utf-8",
    )

    with pytest.raises(ProfileCommandError) as exc:
        build_runtime_run_request(command="inspect", project=str(project_yaml))

    assert (
        "must reference a runtime operation; 'coverage_stats' is an artifact operation"
        in str(exc.value)
    )


def test_build_profile_rejects_runtime_operation(tmp_path: Path):
    project_yaml = _write_project(tmp_path)
    profiles = tmp_path / "profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    (profiles / "build.dataset.yaml").write_text(
        "operation: dataset\n",
        encoding="utf-8",
    )

    with pytest.raises(ProfileCommandError) as exc:
        build_build_run_request(project=str(project_yaml))

    assert (
        "must reference an artifact operation; 'dataset' is a runtime operation"
        in str(exc.value)
    )


def test_build_profile_rejects_unknown_operation(tmp_path: Path):
    project_yaml = _write_project(tmp_path)
    profiles = tmp_path / "profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    (profiles / "build.typo.yaml").write_text(
        "operation: scheam\n",
        encoding="utf-8",
    )

    with pytest.raises(ProfileCommandError) as exc:
        build_build_run_request(project=str(project_yaml))

    assert "references unknown operation 'scheam'" in str(exc.value)


def test_serve_profile_rejects_removed_pipeline_operation_id(tmp_path: Path):
    project_yaml = _write_project(tmp_path)
    profiles = tmp_path / "profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    (profiles / "serve.legacy.yaml").write_text(
        "operation: pipeline\n",
        encoding="utf-8",
    )

    with pytest.raises(ProfileCommandError) as exc:
        build_runtime_run_request(command="serve", project=str(project_yaml))

    assert "references unknown operation 'pipeline'" in str(exc.value)


def test_serve_request_orders_enabled_profiles_and_run_selects_named_profile(
    tmp_path: Path,
):
    project_yaml = _write_project(tmp_path)
    profiles = tmp_path / "profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    (profiles / "serve.early.yaml").write_text(
        "order: 10\noperation: dataset\nenabled: true\n",
        encoding="utf-8",
    )
    (profiles / "serve.train.yaml").write_text(
        "order: 40\noperation: dataset\nenabled: true\n",
        encoding="utf-8",
    )

    request_all = build_runtime_run_request(
        command="serve",
        project=str(project_yaml),
    )
    assert request_all is not None
    assert [job.name for job in request_all.jobs] == ["early", "train"]
    assert [job.task.id for job in request_all.jobs] == [
        "dataset",
        "dataset",
    ]
    assert request_all.jobs[0].runtime is not request_all.jobs[1].runtime

    request_train = build_runtime_run_request(
        command="serve",
        project=str(project_yaml),
        profile_name="train",
    )
    assert request_train is not None
    assert [job.name for job in request_train.jobs] == ["train"]
    assert [job.task.id for job in request_train.jobs] == ["dataset"]


def test_cli_artifact_mode_overrides_serve_defaults(tmp_path: Path):
    project_yaml = _write_project(tmp_path)
    profiles = tmp_path / "profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    (profiles / "serve.defaults.yaml").write_text(
        (
            'artifact_mode: "require_current"\n'
            "observability:\n"
            "  visuals: false\n"
            "  heartbeat_interval_seconds: 30\n"
            "  logging:\n"
            "    level: warning\n"
            "    outputs:\n"
            "      - transport: stderr\n"
            "      - transport: fs\n"
            "        scope: execution\n"
        ),
        encoding="utf-8",
    )
    (profiles / "serve.first.yaml").write_text(
        "operation: dataset\n",
        encoding="utf-8",
    )
    (profiles / "serve.second.yaml").write_text(
        "operation: dataset\n",
        encoding="utf-8",
    )

    request = build_runtime_run_request(
        command="serve",
        project=str(project_yaml),
        artifact_mode="rebuild",
        command_observability=CommandObservability(
            visuals=True,
            heartbeat_interval_seconds=0,
            log_level="debug",
        ),
    )

    assert request is not None
    settings = request.artifact_settings
    assert settings is not None
    assert settings.mode == "rebuild"
    assert settings.observability.heartbeat_interval_seconds == 0
    assert settings.observability.visuals is True
    assert settings.observability.log_decision.name == "DEBUG"
    assert [
        (
            output.transport,
            output.destination.name if output.destination is not None else None,
        )
        for output in settings.observability.log_output.outputs
    ] == [
        ("stderr", None),
        ("fs", "serve.artifacts.log"),
    ]
    assert {
        job.name: [
            (
                output.transport,
                output.destination.name if output.destination is not None else None,
            )
            for output in job.observability.log_output.outputs
        ]
        for job in request.jobs
    } == {
        "first": [("stderr", None), ("fs", "serve.first.log")],
        "second": [("stderr", None), ("fs", "serve.second.log")],
    }


def test_serve_defaults_control_artifact_mode_for_all_profiles(tmp_path: Path):
    project_yaml = _write_project(tmp_path)
    profiles = tmp_path / "profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    (profiles / "serve.defaults.yaml").write_text(
        'artifact_mode: "require_current"\n',
        encoding="utf-8",
    )
    (profiles / "serve.first.yaml").write_text(
        "operation: dataset\n",
        encoding="utf-8",
    )
    (profiles / "serve.second.yaml").write_text(
        "operation: dataset\n",
        encoding="utf-8",
    )

    request = build_runtime_run_request(
        command="serve",
        project=str(project_yaml),
    )
    assert request is not None
    assert request.artifact_settings.mode == "require_current"


def test_serve_defaults_apply_when_profile_omits_fields(tmp_path: Path):
    project_yaml = _write_project(tmp_path)
    profiles = tmp_path / "profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    (profiles / "serve.defaults.yaml").write_text(
        (
            "output:\n"
            "  transport: fs\n"
            "  format: jsonl\n"
            "  directory: ./artifacts/serve\n"
            "observability:\n"
            "  heartbeat_interval_seconds: 30\n"
            "  logging:\n"
            "    outputs:\n"
            "      - transport: stdout\n"
        ),
        encoding="utf-8",
    )
    (profiles / "serve.train.yaml").write_text(
        "operation: dataset\n",
        encoding="utf-8",
    )

    request = build_runtime_run_request(
        command="serve",
        project=str(project_yaml),
        profile_name="train",
    )
    assert request is not None
    job = request.jobs[0]
    assert job.output.transport == "fs"
    assert job.output.run is not None
    assert job.observability.log_output.outputs[0].transport == "stdout"
    assert job.observability.heartbeat_interval_seconds == 30
    assert request.artifact_settings is not None
    assert request.artifact_settings.observability.heartbeat_interval_seconds == 30
    assert len(request.serve_run_plans) == 1
    assert request.serve_run_plans[0].paths == job.output.run
    assert not (tmp_path / "artifacts" / "_system" / "executions").exists()
    assert not job.output.run.dataset_dir.exists()
    assert not job.output.run.metadata_path.exists()


def test_serve_profile_fields_override_serve_defaults(tmp_path: Path):
    project_yaml = _write_project(tmp_path)
    profiles = tmp_path / "profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    (profiles / "serve.defaults.yaml").write_text(
        ("output:\n  transport: fs\n  format: jsonl\n  directory: ./artifacts/serve\n"),
        encoding="utf-8",
    )
    (profiles / "serve.train.yaml").write_text(
        ("operation: dataset\noutput:\n  transport: stdout\n  format: jsonl\n"),
        encoding="utf-8",
    )

    request = build_runtime_run_request(
        command="serve",
        project=str(project_yaml),
        profile_name="train",
    )
    assert request is not None
    job = request.jobs[0]
    assert job.output.transport == "stdout"
    assert job.output.run is None


def test_cli_directory_override_inherits_profile_transport_and_format(
    tmp_path: Path,
):
    project_yaml = _write_project(tmp_path)
    profiles = tmp_path / "profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    (profiles / "serve.defaults.yaml").write_text(
        ("output:\n  transport: fs\n  format: jsonl\n  directory: ./artifacts/serve\n"),
        encoding="utf-8",
    )
    (profiles / "serve.train.yaml").write_text("operation: dataset\n")

    request = build_runtime_run_request(
        command="serve",
        project=str(project_yaml),
        profile_name="train",
        cli_output={"directory": tmp_path / "elsewhere"},
    )
    assert request is not None
    job = request.jobs[0]
    assert job.output.transport == "fs"
    assert job.output.format == "jsonl"
    assert job.output.run is not None
    assert job.output.run.dataset_dir == (
        (tmp_path / "elsewhere").resolve() / "runs" / job.output.run.run_id / "dataset"
    )
    assert job.output.destination == job.output.run.dataset_dir / "train.jsonl"


def test_serve_profile_nested_observability_deep_merges_defaults(
    tmp_path: Path,
):
    project_yaml = _write_project(tmp_path)
    profiles = tmp_path / "profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    (profiles / "serve.defaults.yaml").write_text(
        (
            "output:\n"
            "  transport: fs\n"
            "  format: jsonl\n"
            "  directory: output\n"
            "observability:\n"
            "  logging:\n"
            "    outputs:\n"
            "      - transport: stdout\n"
        ),
        encoding="utf-8",
    )
    (profiles / "serve.train.yaml").write_text(
        ("operation: dataset\nobservability:\n  logging:\n    level: debug\n"),
        encoding="utf-8",
    )

    request = build_runtime_run_request(
        command="serve",
        project=str(project_yaml),
        profile_name="train",
    )
    assert request is not None
    job = request.jobs[0]
    assert job.observability.log_decision.name == "DEBUG"
    assert job.observability.log_output.outputs[0].transport == "stdout"


def test_serve_heartbeat_preserves_prerequisite_and_profile_precedence(
    tmp_path: Path,
):
    project_yaml = _write_project(tmp_path)
    profiles = tmp_path / "profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    (profiles / "serve.defaults.yaml").write_text(
        "observability:\n  heartbeat_interval_seconds: 30\n",
        encoding="utf-8",
    )
    (profiles / "serve.train.yaml").write_text(
        ("operation: dataset\nobservability:\n  heartbeat_interval_seconds: 180\n"),
        encoding="utf-8",
    )

    request = build_runtime_run_request(
        command="serve",
        project=str(project_yaml),
        profile_name="train",
    )
    cli_override = build_runtime_run_request(
        command="serve",
        project=str(project_yaml),
        profile_name="train",
        command_observability=CommandObservability(
            heartbeat_interval_seconds=0,
        ),
    )

    assert request is not None
    assert request.artifact_settings.observability.heartbeat_interval_seconds == 30
    assert request.jobs[0].observability.heartbeat_interval_seconds == 180
    assert cli_override is not None
    assert cli_override.artifact_settings.observability.heartbeat_interval_seconds == 0
    assert cli_override.jobs[0].observability.heartbeat_interval_seconds == 0


def test_build_defaults_apply_to_build_profiles(tmp_path: Path):
    project_yaml = _write_project(tmp_path)
    profiles = tmp_path / "profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    (profiles / "build.defaults.yaml").write_text(
        (
            "artifact_mode: rebuild\n"
            "execution:\n"
            "  sort_buffer_mb: 256\n"
            "observability:\n"
            "  visuals: false\n"
            "  logging:\n"
            "    level: debug\n"
        ),
        encoding="utf-8",
    )
    (profiles / "build.metadata.yaml").write_text(
        "operation: metadata\n",
        encoding="utf-8",
    )

    request = build_build_run_request(
        project=str(project_yaml),
        profile_name="metadata",
    )
    assert request is not None
    assert request.definition.artifact_hashes.values
    assert request.execution.sort_buffer_mb == 256
    job = request.jobs[0]
    assert job.settings.mode == "rebuild"
    assert job.settings.observability.visuals is False
    assert job.settings.observability.log_decision.name == "DEBUG"
