from pathlib import Path

import pytest

from jerrythomas.execution.settings import LogOutputSettings, LogOutputTarget
from jerrythomas.io.output import OutputTarget, validate_output_destinations
from jerrythomas.io.runs import get_run_paths
from jerrythomas.profiles.destinations import validate_command_destinations


def _output(path: Path) -> OutputTarget:
    return OutputTarget(
        transport="fs",
        format="jsonl",
        view="raw",
        encoding="utf-8",
        destination=path,
    )


def _logs(*paths: Path) -> LogOutputSettings:
    return LogOutputSettings(
        outputs=tuple(
            LogOutputTarget(transport="fs", destination=path) for path in paths
        )
    )


@pytest.mark.parametrize("reverse", (False, True))
def test_data_outputs_reject_nested_paths(tmp_path: Path, reverse: bool) -> None:
    parent = _output(tmp_path / "data.jsonl")
    child = _output(tmp_path / "data.jsonl" / "child.jsonl")
    outputs = [child, parent] if reverse else [parent, child]

    with pytest.raises(ValueError, match="Data output paths overlap"):
        validate_output_destinations(outputs)


@pytest.mark.parametrize("data_is_parent", (False, True))
def test_command_rejects_nested_data_and_log_paths(
    tmp_path: Path,
    data_is_parent: bool,
) -> None:
    parent = tmp_path / "output"
    child = parent / "events.log"
    data_path, log_path = (parent, child) if data_is_parent else (child, parent)

    with pytest.raises(ValueError, match="Data output and logging paths overlap"):
        validate_command_destinations(
            tmp_path / "artifacts",
            [_output(data_path)],
            [_logs(log_path)],
            [],
        )


def test_command_rejects_nested_logs_but_allows_shared_log(tmp_path: Path) -> None:
    shared = tmp_path / "logs" / "shared.log"
    validate_command_destinations(
        tmp_path / "artifacts",
        [],
        [_logs(shared), _logs(shared)],
        [],
    )

    with pytest.raises(ValueError, match="Filesystem log output paths overlap"):
        validate_command_destinations(
            tmp_path / "artifacts",
            [],
            [_logs(shared), _logs(shared / "nested.log")],
            [],
        )


@pytest.mark.parametrize(
    "relative_log",
    (
        ".",
        "latest/events.log",
        ".latest-run-1",
        ".latest-run-1/events.log",
    ),
)
def test_command_rejects_logs_at_managed_serve_paths(
    tmp_path: Path,
    relative_log: str,
) -> None:
    serve_root = tmp_path / "served"
    run = get_run_paths(serve_root, run_id="run-1")

    with pytest.raises(ValueError, match="managed serve run paths"):
        validate_command_destinations(
            tmp_path / "artifacts",
            [],
            [_logs(serve_root / relative_log)],
            [run],
        )


def test_command_allows_sibling_data_and_log_paths(tmp_path: Path) -> None:
    validate_command_destinations(
        tmp_path / "artifacts",
        [_output(tmp_path / "output" / "data.jsonl")],
        [_logs(tmp_path / "output" / "events.log")],
        [],
    )
