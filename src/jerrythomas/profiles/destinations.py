from collections.abc import Sequence
from pathlib import Path

from jerrythomas.execution.settings import LogOutputSettings, LogOutputTarget
from jerrythomas.io.output import (
    OutputTarget,
    output_destination_key,
    output_paths_overlap,
    validate_output_destinations,
)
from jerrythomas.io.runs import RunPaths


def _filesystem_logs(
    targets: Sequence[LogOutputTarget],
) -> tuple[tuple[LogOutputTarget, Path], ...]:
    logs: list[tuple[LogOutputTarget, Path]] = []
    for target in targets:
        if target.transport != "fs":
            continue
        if target.destination is None:
            raise ValueError("Filesystem log output has no destination.")
        logs.append((target, target.destination.resolve()))
    return tuple(logs)


def _managed_serve_paths(serve_runs: Sequence[RunPaths]) -> tuple[Path, ...]:
    return tuple(
        path.resolve()
        for run in serve_runs
        for path in (
            run.serve_root / "runs",
            run.serve_root / "latest",
            run.serve_root / f".latest-{run.run_id}",
        )
    )


def _validate_global_logs(
    logs: Sequence[tuple[LogOutputTarget, Path]],
    artifacts_root: Path,
    managed_serve_paths: Sequence[Path],
) -> None:
    for target, path in logs:
        if target.scope != "global":
            continue
        if output_paths_overlap(path, artifacts_root):
            raise ValueError(
                f"Global log output must stay outside the artifacts root: {path}"
            )
        if any(output_paths_overlap(path, managed) for managed in managed_serve_paths):
            raise ValueError(
                f"Global log output conflicts with managed serve run paths: {path}"
            )


def _validate_data_log_paths(
    data_paths: Sequence[Path],
    logs: Sequence[tuple[LogOutputTarget, Path]],
) -> None:
    for _target, log_path in logs:
        log_key = output_destination_key(log_path)
        for data_path in data_paths:
            if log_key == output_destination_key(data_path):
                raise ValueError(
                    f"Data output and logging resolve to the same path '{log_path}'."
                )
            if output_paths_overlap(log_path, data_path):
                raise ValueError(
                    "Data output and logging paths overlap: "
                    f"'{data_path}' and '{log_path}'."
                )


def _validate_log_paths(
    logs: Sequence[tuple[LogOutputTarget, Path]],
) -> None:
    for position, (_target, path) in enumerate(logs):
        path_key = output_destination_key(path)
        for _previous_target, previous in logs[:position]:
            if path_key == output_destination_key(previous):
                continue
            if output_paths_overlap(path, previous):
                raise ValueError(
                    f"Filesystem log output paths overlap: '{previous}' and '{path}'."
                )


def validate_command_destinations(
    artifacts_root: Path,
    outputs: Sequence[OutputTarget],
    log_outputs: Sequence[LogOutputSettings],
    serve_runs: Sequence[RunPaths],
) -> None:
    validate_output_destinations(outputs)

    targets = tuple(target for output in log_outputs for target in output.outputs)
    data_uses_stdout = any(output.transport == "stdout" for output in outputs)
    logs_use_stdout = any(target.transport == "stdout" for target in targets)
    if data_uses_stdout and logs_use_stdout:
        raise ValueError("Data output and logging cannot both use stdout.")

    data_paths = tuple(
        output.destination.resolve()
        for output in outputs
        if output.transport == "fs" and output.destination is not None
    )
    logs = _filesystem_logs(targets)
    _validate_global_logs(
        logs,
        artifacts_root.resolve(),
        _managed_serve_paths(serve_runs),
    )
    _validate_data_log_paths(data_paths, logs)
    _validate_log_paths(logs)
