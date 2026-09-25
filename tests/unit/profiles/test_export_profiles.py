import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from jerrythomas.config.execution import ExecutionConfig
from jerrythomas.config.tasks.stream import StreamTask
from jerrythomas.config.profiles.export import ExportProfile
from jerrythomas.execution.settings import (
    CommandObservability,
    DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
    LogLevelDecision,
    LogOutputSettings,
    ObservabilitySettings,
)
from jerrythomas.operations.persistence import WrittenOutput
from jerrythomas.profiles import export
from jerrythomas.profiles.models import ExportJob
from jerrythomas.services.export import resolve_export_output
from tests.run_helpers import empty_recipe


def _observability() -> ObservabilitySettings:
    return ObservabilitySettings(
        visuals=False,
        heartbeat_interval_seconds=DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
        log_decision=LogLevelDecision(name="INFO", value=logging.INFO),
        log_output=LogOutputSettings(outputs=()),
    )


def _profile(name: str, stream: str, output: str) -> ExportProfile:
    return ExportProfile(
        cmd="export",
        name=name,
        operation=stream,
        output=output,
    )


def _job(
    name: str,
    stream: str,
    output: Path,
    overwrite: bool = False,
):
    return ExportJob(
        name=name,
        task=StreamTask(id=stream, stream=stream),
        output=resolve_export_output(output),
        overwrite=overwrite,
        observability=_observability(),
    )


def test_resolve_export_jobs_applies_command_overrides(
    tmp_path,
) -> None:
    profiles = [
        _profile("adv-20", "adv.20", "adv-20.jsonl"),
        _profile("adv-63", "adv.63", "adv-63.jsonl"),
    ]

    jobs = export.resolve_export_jobs(
        profiles=profiles,
        definition=SimpleNamespace(
            project=SimpleNamespace(path=tmp_path / "project.yaml"),
            runtime_operations=tuple(
                StreamTask(id=stream, stream=stream) for stream in ("adv.20", "adv.63")
            ),
        ),
        execution_dir=tmp_path / "execution",
        overwrite=True,
        cli_output=None,
        command_observability=CommandObservability(),
    )

    assert [job.name for job in jobs] == ["adv-20", "adv-63"]
    assert [job.stream for job in jobs] == ["adv.20", "adv.63"]
    assert [job.output.destination for job in jobs] == [
        tmp_path / "adv-20.jsonl",
        tmp_path / "adv-63.jsonl",
    ]
    assert all(job.output.compression is None for job in jobs)
    assert all(job.overwrite for job in jobs)


def test_resolve_export_jobs_derives_gzip_from_profile_output(tmp_path) -> None:
    jobs = export.resolve_export_jobs(
        profiles=[_profile("adv-20", "adv.20", "adv-20.jsonl.gz")],
        definition=SimpleNamespace(
            project=SimpleNamespace(path=tmp_path / "project.yaml"),
            runtime_operations=tuple(
                StreamTask(id=stream, stream=stream) for stream in ("adv.20", "adv.63")
            ),
        ),
        execution_dir=tmp_path / "execution",
        overwrite=None,
        cli_output=None,
        command_observability=CommandObservability(),
    )

    assert jobs[0].output.destination == tmp_path / "adv-20.jsonl.gz"
    assert jobs[0].output.compression == "gzip"


def test_resolve_export_jobs_derives_gzip_from_output_override(tmp_path) -> None:
    jobs = export.resolve_export_jobs(
        profiles=[_profile("adv-20", "adv.20", "profile.jsonl")],
        definition=SimpleNamespace(
            project=SimpleNamespace(path=tmp_path / "project.yaml"),
            runtime_operations=tuple(
                StreamTask(id=stream, stream=stream) for stream in ("adv.20", "adv.63")
            ),
        ),
        execution_dir=tmp_path / "execution",
        overwrite=None,
        cli_output=tmp_path / "override.jsonl.gz",
        command_observability=CommandObservability(),
    )

    assert jobs[0].output.destination == tmp_path / "override.jsonl.gz"
    assert jobs[0].output.compression == "gzip"


def test_output_override_requires_one_selected_profile(tmp_path) -> None:
    profiles = [
        _profile("adv-20", "adv.20", "adv-20.jsonl"),
        _profile("adv-63", "adv.63", "adv-63.jsonl"),
    ]

    with pytest.raises(ValueError, match="one selected profile"):
        export.resolve_export_jobs(
            profiles=profiles,
            definition=SimpleNamespace(
                project=SimpleNamespace(path=tmp_path / "project.yaml"),
                runtime_operations=tuple(
                    StreamTask(id=stream, stream=stream)
                    for stream in ("adv.20", "adv.63")
                ),
            ),
            execution_dir=tmp_path / "execution",
            overwrite=None,
            cli_output=tmp_path / "override.jsonl",
            command_observability=CommandObservability(),
        )


def test_preflight_rejects_unknown_stream(tmp_path) -> None:
    runtime = SimpleNamespace(streams={}, artifacts_root=tmp_path / "artifacts")

    with pytest.raises(ValueError, match="unknown stream 'adv.20'"):
        export.preflight_export_jobs(
            runtime,
            [_job("adv-20", "adv.20", tmp_path / "adv-20.jsonl")],
        )


@pytest.mark.parametrize(
    "filenames",
    [("adv.jsonl", "adv.jsonl"), ("ADV.jsonl", "adv.jsonl")],
)
def test_preflight_rejects_duplicate_destinations(tmp_path, filenames) -> None:
    runtime = SimpleNamespace(
        streams={"adv.20": object(), "adv.63": object()},
        artifacts_root=tmp_path / "artifacts",
    )
    with pytest.raises(ValueError, match="resolve to the same path"):
        export.preflight_export_jobs(
            runtime,
            [
                _job("adv-20", "adv.20", tmp_path / filenames[0]),
                _job("adv-63", "adv.63", tmp_path / filenames[1]),
            ],
        )


def test_preflight_checks_every_destination_before_execution(tmp_path) -> None:
    existing = tmp_path / "second.jsonl"
    existing.write_text("existing\n", encoding="utf-8")
    runtime = SimpleNamespace(
        streams={"adv.20": object(), "adv.63": object()},
        artifacts_root=tmp_path / "artifacts",
    )

    with pytest.raises(FileExistsError, match="--overwrite"):
        export.preflight_export_jobs(
            runtime,
            [
                _job("first", "adv.20", tmp_path / "first.jsonl"),
                _job("second", "adv.63", existing),
            ],
        )


@pytest.mark.parametrize(
    "reserved", ["first.jsonl.run.json", "first.jsonl.recipe.json", ".first.jsonl.lock"]
)
def test_preflight_rejects_output_nested_under_receipt_or_lock(tmp_path, reserved):
    runtime = SimpleNamespace(
        streams={"adv.20": object(), "adv.63": object()},
        artifacts_root=tmp_path / "artifacts",
    )
    with pytest.raises(ValueError, match="overlaps data output"):
        export.preflight_export_jobs(
            runtime,
            [
                _job("first", "adv.20", tmp_path / "first.jsonl"),
                _job("second", "adv.63", tmp_path / reserved / "second.jsonl"),
            ],
        )


def test_preflight_rejects_managed_artifact_destination(tmp_path) -> None:
    artifacts = tmp_path / "artifacts"
    runtime = SimpleNamespace(
        streams={"adv.20": object()},
        artifacts_root=artifacts,
    )

    with pytest.raises(ValueError, match="inside the managed artifacts root"):
        export.preflight_export_jobs(
            runtime,
            [_job("adv-20", "adv.20", artifacts / "adv-20.jsonl", True)],
        )


def test_execute_export_job_emits_config_and_returns_output(
    monkeypatch,
    tmp_path,
) -> None:
    runtime = SimpleNamespace(execution=ExecutionConfig())
    job = _job("adv-20", "adv.20", tmp_path / "adv-20.jsonl")
    messages: list[tuple[str, int]] = []
    calls: list[dict] = []
    monkeypatch.setattr(
        export,
        "emit_execution_message",
        lambda message, level: messages.append((message, level)),
    )

    def export_stream(**kwargs):
        calls.append(kwargs)
        job.output.destination.write_text("{}\n")
        return WrittenOutput(job.output.destination, None, 3)

    monkeypatch.setattr(
        export,
        "export_stream",
        export_stream,
    )

    result = export.execute_export_job(
        job, runtime, recipe=empty_recipe("export"), definition=SimpleNamespace()
    )

    assert calls == [
        {
            "runtime": runtime,
            "task": job.task,
            "output": job.output,
            "overwrite": False,
        }
    ]
    config = json.loads(messages[0][0].removeprefix("Config:\n"))
    assert messages[0][1] == logging.DEBUG
    assert config["stream"] == "adv.20"
    output = result.output("adv-20")
    assert output.profile == "adv-20"
    assert output.stream == "adv.20"
    assert result.outputs == (job.output.destination,)
    assert output.row_count == 3
    assert output.format == "jsonl"
    assert output.view == "raw"
