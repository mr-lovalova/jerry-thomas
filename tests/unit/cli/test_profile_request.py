from pathlib import Path
from types import SimpleNamespace

import pytest

from jerrythomas.cli.commands.profile_runner import handle_serve
from jerrythomas.cli.parser_builder import build_parser
from jerrythomas.config.execution import ExecutionConfig
from jerrythomas.execution.settings import CommandObservability, LogOutputTarget
from jerrythomas.profiles.errors import ProfileCommandError
from jerrythomas.profiles.request_builder import (
    build_build_run_request,
    build_materialize_run_request,
    build_runtime_run_request,
)


def _write_project(tmp_path: Path) -> Path:
    project_yaml = tmp_path / "project.yaml"
    project_yaml.write_text(
        "\n".join(
            [
                "schema_version: 7",
                "artifact_revision: 1",
                "paths:",
                "  streams: streams",
                "  sources: sources",
                "  datasets: datasets",
                "  artifacts: artifacts",
                "  operations: operations",
                "  profiles: profiles",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "datasets").mkdir(parents=True, exist_ok=True)
    (tmp_path / "datasets/default.yaml").write_text(
        "version: v1\nsample:\n  rounding: ceil\n  cadence: 1h\n",
        encoding="utf-8",
    )
    for directory in ("streams", "sources", "operations"):
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    for operation, entrypoint in (
        ("dataset", "core.dataset"),
        ("coverage", "core.coverage_report"),
        ("matrix", "core.availability_matrix"),
    ):
        (tmp_path / "operations" / f"{operation}.yaml").write_text(
            f"kind: output\nentrypoint: {entrypoint}\ndataset: default\n",
            encoding="utf-8",
        )
    return project_yaml


def _write_stream(tmp_path: Path, stream: str) -> None:
    (tmp_path / "sources" / "source.yaml").write_text(
        "id: source\nparser: {entrypoint: identity}\n"
        "loader: {entrypoint: plugin.source}\nfreshness: opaque\n",
        encoding="utf-8",
    )
    (tmp_path / "streams" / f"{stream}.yaml").write_text(
        f"id: {stream}\nfrom: {{source: source}}\nmap: {{entrypoint: identity}}\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize("output_file", ["serve.dataset.yaml", "serve.defaults.yaml"])
def test_cli_format_override_resolves_real_request_without_text_encoding(
    tmp_path: Path, monkeypatch, output_file: str
) -> None:
    project_yaml = _write_project(tmp_path)
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    (profiles / "serve.dataset.yaml").write_text(
        "operation: dataset\n", encoding="utf-8"
    )
    with (profiles / output_file).open("a", encoding="utf-8") as profile_file:
        profile_file.write(
            "output:\n  transport: fs\n  format: jsonl\n  directory: out\n"
        )
    requests = []
    monkeypatch.setattr(
        "jerrythomas.cli.commands.profile_runner.execute_profile_request",
        requests.append,
    )
    monkeypatch.setattr(
        "jerrythomas.cli.commands.profile_runner.configure_profile_logging",
        lambda *_args: None,
    )
    args = build_parser().parse_args(
        ["serve", "--project", str(project_yaml), "--output-format", "parquet"]
    )

    handle_serve(args, workspace=None, cli_log_level=None, cli_log_outputs=[])

    assert len(requests) == 1
    output = requests[0].jobs[0].output
    assert output.format == "parquet"
    assert output.encoding is None
    assert output.destination.name == "dataset.parquet"
    assert output.destination.is_relative_to(tmp_path / "out")


def test_build_request_requires_declared_build_profiles(tmp_path: Path):
    project_yaml = _write_project(tmp_path)
    (tmp_path / "operations").mkdir(parents=True, exist_ok=True)
    (tmp_path / "profiles").mkdir(parents=True, exist_ok=True)

    with pytest.raises(ProfileCommandError) as exc:
        build_build_run_request(
            project=str(project_yaml),
        )
    assert "Project does not define build profiles" in str(exc.value)


def test_inspect_request_requires_declared_inspect_profiles(tmp_path: Path):
    project_yaml = _write_project(tmp_path)
    (tmp_path / "operations").mkdir(parents=True, exist_ok=True)
    (tmp_path / "profiles").mkdir(parents=True, exist_ok=True)

    with pytest.raises(ProfileCommandError) as exc:
        build_runtime_run_request(
            command="inspect",
            project=str(project_yaml),
        )
    assert "Project does not define inspect profiles" in str(exc.value)


def test_project_definition_unexpected_runtime_error_propagates(monkeypatch) -> None:
    error = RuntimeError("unexpected loader bug")

    def fail(_path: Path):
        raise error

    monkeypatch.setattr(
        "jerrythomas.profiles.request_builder.load_project_definition",
        fail,
    )

    with pytest.raises(RuntimeError) as raised:
        build_build_run_request(project="project.yaml")

    assert raised.value is error


def test_project_definition_validation_does_not_log_secret_inputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_yaml = _write_project(tmp_path)
    secret = "do-not-log-this-secret"
    monkeypatch.setenv("JERRY_TEST_SECRET", secret)
    project_yaml.write_text(
        project_yaml.read_text(encoding="utf-8").replace(
            "artifact_revision: 1",
            "artifact_revision: ${env:JERRY_TEST_SECRET}",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ProfileCommandError) as raised:
        build_runtime_run_request(
            command="serve",
            project=str(project_yaml),
        )

    message = str(raised.value)
    assert secret not in message
    assert "input_value" not in message
    assert "input_type" not in message
    assert "artifact_revision" in message
    assert "Input should be a valid integer" in message


def test_profile_validation_does_not_log_secret_inputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_yaml = _write_project(tmp_path)
    secret = "do-not-log-this-secret"
    monkeypatch.setenv("JERRY_TEST_SECRET", secret)
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    (profiles / "serve.dataset.yaml").write_text(
        """\
operation: dataset
observability:
  logging:
    level: ${env:JERRY_TEST_SECRET}
""",
        encoding="utf-8",
    )

    with pytest.raises(ProfileCommandError) as raised:
        build_runtime_run_request(
            command="serve",
            project=str(project_yaml),
        )

    message = str(raised.value)
    assert secret not in message
    assert "input_value" not in message
    assert "input_type" not in message
    assert "logging.level" in message
    assert "Invalid configuration value" in message


def test_custom_validation_message_does_not_log_resolved_secret(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_yaml = _write_project(tmp_path)
    secret = "do-not-log-this-output-id"
    monkeypatch.setenv("JERRY_TEST_SECRET", secret)
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    (profiles / "serve.dataset.yaml").write_text(
        "operation: dataset\n"
        "include_outputs:\n"
        "  - ${env:JERRY_TEST_SECRET}\n"
        "  - ${env:JERRY_TEST_SECRET}\n",
        encoding="utf-8",
    )

    with pytest.raises(ProfileCommandError) as raised:
        build_runtime_run_request(
            command="serve",
            project=str(project_yaml),
        )

    message = str(raised.value)
    assert secret not in message
    assert "include_outputs" in message
    assert "Invalid configuration value" in message


def test_inspect_request_materializes_execution_scoped_log_output(
    tmp_path: Path, monkeypatch
):
    execution_dir = tmp_path / "execution"
    monkeypatch.setattr(
        "jerrythomas.profiles.request_builder._execution_root",
        lambda _project: execution_dir,
    )
    project_yaml = _write_project(tmp_path)
    ops = tmp_path / "operations"
    profiles = tmp_path / "profiles"
    ops.mkdir(parents=True, exist_ok=True)
    profiles.mkdir(parents=True, exist_ok=True)
    (ops / "coverage.yaml").write_text(
        "kind: output\nentrypoint: core.coverage_report\ndataset: default\n",
        encoding="utf-8",
    )
    (profiles / "inspect.coverage.yaml").write_text(
        ("operation: coverage\nenabled: true\n"),
        encoding="utf-8",
    )
    (tmp_path / "sources").mkdir(parents=True, exist_ok=True)

    request = build_runtime_run_request(
        command="inspect",
        project=str(project_yaml),
        command_observability=CommandObservability(
            log_outputs=(LogOutputTarget(transport="fs", scope="execution"),),
        ),
    )
    assert request is not None
    job = request.jobs[0]
    assert job.observability.log_output.outputs[0].scope == "execution"
    assert job.observability.log_output.outputs[0].destination == (
        execution_dir / "logs" / "inspect.coverage.log"
    )
    assert not execution_dir.exists()


def test_disabled_profiles_do_not_create_execution_directory(tmp_path: Path):
    project_yaml = _write_project(tmp_path)
    ops = tmp_path / "operations"
    profiles = tmp_path / "profiles"
    ops.mkdir(parents=True, exist_ok=True)
    profiles.mkdir(parents=True, exist_ok=True)
    (ops / "coverage.yaml").write_text(
        "kind: output\nentrypoint: core.coverage_report\ndataset: default\n",
        encoding="utf-8",
    )
    (profiles / "inspect.coverage.yaml").write_text(
        ("operation: coverage\nenabled: false\n"),
        encoding="utf-8",
    )

    request = build_runtime_run_request(
        command="inspect",
        project=str(project_yaml),
    )

    assert request is None
    assert not (tmp_path / "artifacts" / "_system" / "executions").exists()


def test_serve_request_uses_dataset_output_ids_by_default(tmp_path: Path):
    project_yaml = _write_project(tmp_path)
    (tmp_path / "datasets/default.yaml").write_text(
        """\
version: v1
sample: {rounding: ceil, cadence: 1h}
split:
  mode: hash
  ratios: {train: 0.8, test: 0.2}
  folds:
    - id: default
      train: [train]
      test: [test]
""",
        encoding="utf-8",
    )
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    (profiles / "serve.defaults.yaml").write_text(
        "output: {transport: fs, format: jsonl, directory: output}\n",
        encoding="utf-8",
    )
    (profiles / "serve.dataset.yaml").write_text(
        "operation: dataset\n",
        encoding="utf-8",
    )

    request = build_runtime_run_request(
        command="serve",
        project=str(project_yaml),
    )

    assert request is not None
    assert request.jobs[0].output_ids == ("default.train", "default.test")


def test_materialize_request_uses_shared_resolution_snapshot(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_yaml = _write_project(tmp_path)
    (tmp_path / "operations").mkdir(parents=True, exist_ok=True)
    profiles = tmp_path / "profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    (profiles / "materialize.defaults.yaml").write_text(
        "artifact_mode: rebuild\nexecution:\n  sort_buffer_mb: 32\n",
        encoding="utf-8",
    )
    (profiles / "materialize.adv-20.yaml").write_text(
        "operation: adv-20\noutput: outputs/adv-20.jsonl.gz\n",
        encoding="utf-8",
    )
    _write_stream(tmp_path, "adv.20")
    (tmp_path / "operations" / "adv-20.yaml").write_text(
        "kind: output\nentrypoint: core.records\nstream: adv.20\n",
        encoding="utf-8",
    )
    runtime = SimpleNamespace(execution=ExecutionConfig())
    compiled_definitions = []

    def compile_runtime(definition):
        compiled_definitions.append(definition)
        return runtime

    monkeypatch.setattr(
        "jerrythomas.profiles.request_builder.compile_runtime",
        compile_runtime,
    )
    execution_dir = tmp_path / "execution"
    execution_root_calls: list[Path] = []

    def shared_execution_root(path: Path) -> Path:
        execution_root_calls.append(path)
        return execution_dir

    monkeypatch.setattr(
        "jerrythomas.profiles.request_builder._execution_root",
        shared_execution_root,
    )
    request = build_materialize_run_request(
        project=str(project_yaml),
        profile_name=None,
        overwrite=None,
        output=None,
        artifact_mode=None,
        command_observability=CommandObservability(
            log_outputs=(LogOutputTarget(transport="fs", scope="execution"),),
        ),
    )

    assert request is not None
    assert request.definition.artifact_hashes.values
    assert compiled_definitions == [request.definition]
    assert request.execution.sort_buffer_mb == 32
    assert request.artifact_settings.mode == "rebuild"
    assert request.runtime is runtime
    assert request.jobs[0].name == "adv-20"
    assert request.jobs[0].stream == "adv.20"
    assert request.jobs[0].output.destination == (
        tmp_path / "outputs" / "adv-20.jsonl.gz"
    )
    assert request.jobs[0].output.compression == "gzip"
    assert request.jobs[0].observability.log_output.outputs[0].destination == (
        execution_dir / "logs" / "materialize.adv-20.log"
    )
    assert (
        request.artifact_settings.observability.log_output.outputs[0].destination
        == execution_dir / "logs" / "materialize.artifacts.log"
    )
    assert execution_root_calls == [tmp_path / "artifacts"]


def test_materialize_request_rejects_output_log_collision(tmp_path: Path) -> None:
    project_yaml = _write_project(tmp_path)
    profiles = tmp_path / "profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    (profiles / "materialize.adv.yaml").write_text(
        (
            "operation: adv\n"
            "output: output/adv.jsonl\n"
            "observability:\n"
            "  logging:\n"
            "    outputs:\n"
            "      - transport: fs\n"
            "        path: output/adv.jsonl\n"
        ),
        encoding="utf-8",
    )

    _write_stream(tmp_path, "adv")
    (tmp_path / "operations" / "adv.yaml").write_text(
        "kind: output\nentrypoint: core.records\nstream: adv\n",
        encoding="utf-8",
    )

    with pytest.raises(ProfileCommandError, match="same path"):
        build_materialize_run_request(
            project=str(project_yaml),
            profile_name=None,
            overwrite=None,
            output=None,
            artifact_mode=None,
        )


@pytest.mark.parametrize("mode", [None, "auto", "rebuild", "require_current"])
def test_build_artifact_mode_override_applies_to_every_selected_profile(
    tmp_path: Path, mode
) -> None:
    project_yaml = _write_project(tmp_path)
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    (profiles / "build.defaults.yaml").write_text("artifact_mode: require_current\n")
    (profiles / "build.series.yaml").write_text("operation: dataset.default.series\n")
    (profiles / "build.metadata.yaml").write_text(
        "operation: dataset.default.metadata\nartifact_mode: rebuild\n"
    )

    request = build_build_run_request(project=str(project_yaml), artifact_mode=mode)

    assert request is not None
    settings = {job.task.id: job.settings.mode for job in request.jobs}
    assert settings == {
        "dataset.default.series": mode or "require_current",
        "dataset.default.metadata": mode or "rebuild",
    }


def test_build_request_rejects_invalid_artifact_mode(tmp_path: Path) -> None:
    project_yaml = _write_project(tmp_path)
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    (profiles / "build.series.yaml").write_text("operation: dataset.default.series\n")

    with pytest.raises(ProfileCommandError, match="Invalid artifact mode"):
        build_build_run_request(project=str(project_yaml), artifact_mode="force")
