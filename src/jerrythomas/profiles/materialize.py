import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from jerrythomas.config.profiles.materialize import MaterializeProfile
from jerrythomas.execution.settings import (
    CommandObservability,
    resolve_execution_log_outputs,
    resolve_observability_settings,
)
from jerrythomas.execution.observability import (
    emit_execution_message,
    operation_scope,
)
from jerrythomas.profiles.destinations import validate_command_destinations
from jerrythomas.profiles.models import MaterializeJob
from jerrythomas.io.runs import (
    RunOutput,
    SavedRun,
    start_run,
    finish_run_success,
    finish_run_failed,
    load_run,
    materialize_receipt_path,
)
from jerrythomas.services.execution_lock import output_execution_lock, output_lock_path
from jerrythomas.runtime import Runtime
from jerrythomas.services.materialize import (
    check_materialize_destination,
    materialize_stream,
    resolve_materialize_output,
)
from jerrythomas.services.path_policy import sanitize_path_segment


def resolve_materialize_jobs(
    profiles: Sequence[MaterializeProfile],
    project_path: Path,
    execution_dir: Path,
    overwrite: bool | None,
    cli_output: Path | None,
    command_observability: CommandObservability,
) -> list[MaterializeJob]:
    if cli_output is not None and len(profiles) != 1:
        raise ValueError("A materialize output override requires one selected profile.")

    jobs: list[MaterializeJob] = []
    for profile in profiles:
        observability = resolve_observability_settings(
            project_path,
            profile.observability,
            command_observability,
        )
        observability = replace(
            observability,
            log_output=resolve_execution_log_outputs(
                observability.log_output,
                execution_dir,
                default_path=(
                    Path("logs")
                    / f"materialize.{sanitize_path_segment(profile.name)}.log"
                ),
            ),
        )
        output = cli_output if cli_output is not None else profile.output
        if not output.is_absolute():
            output = project_path.parent / output
        jobs.append(
            MaterializeJob(
                name=profile.name,
                stream=profile.stream,
                output=resolve_materialize_output(output),
                overwrite=profile.overwrite if overwrite is None else overwrite,
                observability=observability,
            )
        )
    return jobs


def materialize_reserved_paths(jobs: Sequence[MaterializeJob]) -> tuple[Path, ...]:
    return tuple(
        path
        for job in jobs
        if job.output.destination is not None
        for path in (
            materialize_receipt_path(job.output.destination),
            output_lock_path(job.output.destination),
        )
    )


def preflight_materialize_jobs(
    runtime: Runtime,
    jobs: Sequence[MaterializeJob],
) -> None:
    destinations: list[tuple[MaterializeJob, Path]] = []
    available_streams = set(runtime.streams)
    artifacts_root = runtime.artifacts_root.resolve()
    for job in jobs:
        if job.stream not in available_streams:
            raise ValueError(
                f"Materialize profile '{job.name}' references unknown "
                f"stream '{job.stream}'."
            )
        path = job.output.destination
        if job.output.transport != "fs" or path is None:
            raise ValueError(
                f"Materialize profile '{job.name}' requires a filesystem output."
            )
        destinations.append((job, path))
        if path.is_relative_to(artifacts_root):
            raise ValueError(
                f"Materialize profile '{job.name}' writes inside the managed "
                f"artifacts root: {path}"
            )
    validate_command_destinations(
        artifacts_root,
        [job.output for job in jobs],
        [job.observability.log_output for job in jobs],
        (),
        reserved_paths=materialize_reserved_paths(jobs),
    )
    for job, path in destinations:
        check_materialize_destination(path, job.overwrite)


def execute_materialize_job(
    job: MaterializeJob,
    runtime: Runtime,
    *,
    run_id: str | None = None,
) -> SavedRun:
    with operation_scope(f"materialize:{job.name}"):
        emit_execution_message(
            "Config:\n"
            + json.dumps(
                {
                    "stream": job.stream,
                    "output": str(job.output.destination),
                    "overwrite": job.overwrite,
                    "execution": runtime.execution.model_dump(mode="json"),
                    "observability": job.observability.effective_config(),
                },
                indent=2,
            ),
            level=logging.DEBUG,
        )
        destination = job.output.destination
        assert destination is not None
        receipt = materialize_receipt_path(destination)
        with output_execution_lock(destination):
            check_materialize_destination(destination, job.overwrite)
            start_run(
                receipt,
                run_id=run_id,
                command="materialize",
                overwrite=job.overwrite,
            )
            try:
                output = materialize_stream(
                    runtime=runtime,
                    stream_id=job.stream,
                    output=job.output,
                    overwrite=job.overwrite,
                )
                completed = RunOutput(
                    profile=job.name,
                    operation="materialize",
                    stream=job.stream,
                    output_id=None,
                    path=output.path.name,
                    format=job.output.format,
                    view=job.output.view,
                    encoding=job.output.encoding,
                    compression=job.output.compression,
                    row_count=output.row_count,
                    fold=None,
                )
                finish_run_success(receipt, outputs=(completed,))
            except BaseException as exc:
                try:
                    finish_run_failed(receipt)
                except Exception as failure:
                    exc.add_note(f"Failed to finalize receipt '{receipt}': {failure}")
                raise
            return load_run(receipt)
