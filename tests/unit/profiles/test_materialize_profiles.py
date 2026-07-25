import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from datapipeline.config.execution import ExecutionConfig
from datapipeline.config.profiles import MaterializeProfile
from datapipeline.execution.settings import (
    DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
    LogLevelDecision,
    LogOutputSettings,
    ObservabilitySettings,
)
from datapipeline.profiles import materialize
from datapipeline.profiles.models import MaterializeJob
from datapipeline.services.materialize import resolve_materialize_output


def _observability() -> ObservabilitySettings:
    return ObservabilitySettings(
        visuals="off",
        heartbeat_interval_seconds=DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
        log_decision=LogLevelDecision(name="INFO", value=logging.INFO),
        log_output=LogOutputSettings(outputs=()),
    )


def _profile(name: str, stream: str, output: str) -> MaterializeProfile:
    return MaterializeProfile(
        cmd="materialize",
        name=name,
        stream=stream,
        output=output,
    )


def _job(
    name: str,
    stream: str,
    output: Path,
    overwrite: bool = False,
):
    return MaterializeJob(
        name=name,
        stream=stream,
        output=resolve_materialize_output(output),
        overwrite=overwrite,
        observability=_observability(),
    )


def test_resolve_materialize_jobs_applies_command_overrides(
    tmp_path,
) -> None:
    profiles = [
        _profile("adv-20", "adv.20", "adv-20.jsonl"),
        _profile("adv-63", "adv.63", "adv-63.jsonl"),
    ]

    jobs = materialize.resolve_materialize_jobs(
        profiles=profiles,
        project_path=tmp_path / "project.yaml",
        execution_dir=tmp_path / "execution",
        overwrite=True,
        cli_output=None,
        cli_visuals=None,
        cli_heartbeat_interval_seconds=None,
        cli_log_level=None,
        cli_log_outputs=[],
        base_log_level="INFO",
    )

    assert [job.name for job in jobs] == ["adv-20", "adv-63"]
    assert [job.stream for job in jobs] == ["adv.20", "adv.63"]
    assert [job.output.destination for job in jobs] == [
        tmp_path / "adv-20.jsonl",
        tmp_path / "adv-63.jsonl",
    ]
    assert all(job.output.compression is None for job in jobs)
    assert all(job.overwrite for job in jobs)


def test_resolve_materialize_jobs_derives_gzip_from_profile_output(tmp_path) -> None:
    jobs = materialize.resolve_materialize_jobs(
        profiles=[_profile("adv-20", "adv.20", "adv-20.jsonl.gz")],
        project_path=tmp_path / "project.yaml",
        execution_dir=tmp_path / "execution",
        overwrite=None,
        cli_output=None,
        cli_visuals=None,
        cli_heartbeat_interval_seconds=None,
        cli_log_level=None,
        cli_log_outputs=[],
        base_log_level="INFO",
    )

    assert jobs[0].output.destination == tmp_path / "adv-20.jsonl.gz"
    assert jobs[0].output.compression == "gzip"


def test_resolve_materialize_jobs_derives_gzip_from_output_override(tmp_path) -> None:
    jobs = materialize.resolve_materialize_jobs(
        profiles=[_profile("adv-20", "adv.20", "profile.jsonl")],
        project_path=tmp_path / "project.yaml",
        execution_dir=tmp_path / "execution",
        overwrite=None,
        cli_output=tmp_path / "override.jsonl.gz",
        cli_visuals=None,
        cli_heartbeat_interval_seconds=None,
        cli_log_level=None,
        cli_log_outputs=[],
        base_log_level="INFO",
    )

    assert jobs[0].output.destination == tmp_path / "override.jsonl.gz"
    assert jobs[0].output.compression == "gzip"


def test_output_override_requires_one_selected_profile(tmp_path) -> None:
    profiles = [
        _profile("adv-20", "adv.20", "adv-20.jsonl"),
        _profile("adv-63", "adv.63", "adv-63.jsonl"),
    ]

    with pytest.raises(ValueError, match="one selected profile"):
        materialize.resolve_materialize_jobs(
            profiles=profiles,
            project_path=tmp_path / "project.yaml",
            execution_dir=tmp_path / "execution",
            overwrite=None,
            cli_output=tmp_path / "override.jsonl",
            cli_visuals=None,
            cli_heartbeat_interval_seconds=None,
            cli_log_level=None,
            cli_log_outputs=[],
            base_log_level="INFO",
        )


def test_preflight_rejects_unknown_stream(tmp_path) -> None:
    runtime = SimpleNamespace(streams={}, artifacts_root=tmp_path / "artifacts")

    with pytest.raises(ValueError, match="unknown stream 'adv.20'"):
        materialize.preflight_materialize_jobs(
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
    with pytest.raises(ValueError, match="write the same path"):
        materialize.preflight_materialize_jobs(
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
        materialize.preflight_materialize_jobs(
            runtime,
            [
                _job("first", "adv.20", tmp_path / "first.jsonl"),
                _job("second", "adv.63", existing),
            ],
        )


def test_preflight_rejects_managed_artifact_destination(tmp_path) -> None:
    artifacts = tmp_path / "artifacts"
    runtime = SimpleNamespace(
        streams={"adv.20": object()},
        artifacts_root=artifacts,
    )

    with pytest.raises(ValueError, match="inside the managed artifacts root"):
        materialize.preflight_materialize_jobs(
            runtime,
            [_job("adv-20", "adv.20", artifacts / "adv-20.jsonl", True)],
        )


def test_execute_materialize_job_emits_config_and_files(
    monkeypatch,
    tmp_path,
) -> None:
    runtime = SimpleNamespace(execution=ExecutionConfig())
    job = _job("adv-20", "adv.20", tmp_path / "adv-20.jsonl")
    messages: list[tuple[str, int]] = []
    files: list[tuple[str, Path]] = []
    calls: list[dict] = []
    monkeypatch.setattr(
        materialize,
        "emit_execution_message",
        lambda message, level: messages.append((message, level)),
    )
    monkeypatch.setattr(
        materialize,
        "emit_file_result",
        lambda label, path: files.append((label, path)),
    )

    def materialize_stream(**kwargs):
        calls.append(kwargs)
        return job.output.destination

    monkeypatch.setattr(
        materialize,
        "materialize_stream",
        materialize_stream,
    )

    materialize.execute_materialize_job(job, runtime)

    assert calls == [
        {
            "runtime": runtime,
            "stream_id": "adv.20",
            "output": job.output,
            "overwrite": False,
        }
    ]
    config = json.loads(messages[0][0].removeprefix("Config:\n"))
    assert messages[0][1] == logging.DEBUG
    assert config["stream"] == "adv.20"
    assert files == [("Output", job.output.destination)]
