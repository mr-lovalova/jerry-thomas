import argparse
import json
import logging
import time

from jerrythomas.cli.logging_setup import configure_profile_logging
from jerrythomas.cli.output_options import build_cli_output_config
from jerrythomas.cli.visuals.execution import route_execution_event
from jerrythomas.cli.visuals.rich.progress import visual_summary
from jerrythomas.cli.workspace import WorkspaceContext
from jerrythomas.execution.events import RunStatus
from jerrythomas.execution.observability import CommandFinished
from jerrythomas.execution.settings import CommandObservability, LogOutputTarget
from jerrythomas.profiles.errors import ProfileCommandError
from jerrythomas.profiles.models import (
    BuildRunRequest,
    ProfileRunRequest,
    ServeRunResult,
)
from jerrythomas.profiles.orchestration import run_profiles
from jerrythomas.profiles.request_builder import (
    build_build_run_request,
    build_runtime_run_request,
)

logger = logging.getLogger(__name__)


def _command_uses_visuals(request: ProfileRunRequest) -> bool:
    if isinstance(request, BuildRunRequest):
        return any(job.settings.observability.visuals is True for job in request.jobs)
    return request.artifact_settings.observability.visuals is True or any(
        job.observability.visuals is True for job in request.jobs
    )


def execute_profile_request(request: ProfileRunRequest) -> tuple[ServeRunResult, ...]:
    started_at = time.perf_counter()
    status: RunStatus = "error"
    command_error: BaseException | None = None
    try:
        results = run_profiles(request)
    except ProfileCommandError as exc:
        command_error = exc
        logger.error("%s", exc)
        for note in getattr(exc, "__notes__", ()):
            logger.error("%s", note)
        raise SystemExit(2) from exc
    except (KeyboardInterrupt, SystemExit) as exc:
        command_error = exc
        for note in getattr(exc, "__notes__", ()):
            logger.error("%s", note)
        raise
    except BaseException as exc:
        command_error = exc
        raise
    else:
        status = "success"
    finally:
        try:
            with visual_summary(
                logger.getEffectiveLevel(),
                _command_uses_visuals(request),
            ):
                route_execution_event(
                    CommandFinished(
                        command=request.command,
                        status=status,
                        elapsed_seconds=time.perf_counter() - started_at,
                    ),
                    logger,
                )
        except BaseException as exc:
            if command_error is None:
                raise
            message = f"Reporting command completion also failed: {exc}"
            command_error.add_note(message)
            if isinstance(
                command_error,
                (ProfileCommandError, KeyboardInterrupt, SystemExit),
            ):
                logger.error("%s", message)
    return results


def handle_build(
    args: argparse.Namespace,
    cli_log_level: str | None,
    cli_log_outputs: list[LogOutputTarget],
) -> None:
    request = build_build_run_request(
        project=args.project,
        profile_name=args.profile,
        force=args.force,
        command_observability=CommandObservability(
            visuals=args.visuals,
            heartbeat_interval_seconds=args.heartbeat_interval_seconds,
            log_level=cli_log_level,
            log_outputs=tuple(cli_log_outputs),
        ),
    )
    if request is None:
        logger.info("No enabled build profiles; skipping build.")
        return
    configure_profile_logging(cli_log_level, cli_log_outputs)
    execute_profile_request(request)


def handle_serve(
    args: argparse.Namespace,
    workspace: WorkspaceContext | None,
    cli_log_level: str | None,
    cli_log_outputs: list[LogOutputTarget],
) -> None:
    workspace_root = workspace.root if workspace is not None else None
    output = build_cli_output_config(
        transport=args.output_transport,
        fmt=args.output_format,
        directory=args.output_directory,
        output_encoding=args.output_encoding,
        output_compression=args.output_compression,
        workspace_root=workspace_root,
        view=args.output_view,
    )
    request = build_runtime_run_request(
        command="serve",
        project=args.project,
        profile_name=args.profile,
        artifact_mode=args.artifact_mode,
        limit=args.limit,
        preview=args.preview,
        cli_output=output,
        command_observability=CommandObservability(
            visuals=args.visuals,
            heartbeat_interval_seconds=args.heartbeat_interval_seconds,
            log_level=cli_log_level,
            log_outputs=tuple(cli_log_outputs),
        ),
    )
    if request is None:
        if args.result_json:
            print(json.dumps({"schema_version": 1, "runs": []}))
        else:
            logger.info("No enabled serve profiles; skipping serve.")
        return
    if args.result_json:
        if any(job.output.transport == "stdout" for job in request.jobs):
            raise ProfileCommandError(
                "--result-json requires filesystem data outputs; stdout is reserved for run results."
            )
        log_outputs = (
            request.artifact_settings.observability.log_output,
            *(job.observability.log_output for job in request.jobs),
        )
        if any(
            target.transport == "stdout"
            for output in log_outputs
            for target in output.outputs
        ):
            raise ProfileCommandError(
                "--result-json cannot use stdout logging; select stderr or filesystem logs."
            )
    configure_profile_logging(cli_log_level, cli_log_outputs)
    results = execute_profile_request(request)
    if args.result_json:
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "runs": [
                        {
                            "run_id": result.paths.run_id,
                            "directory": str(result.paths.run_root),
                            "preview": result.preview,
                            "outputs": [str(path) for path in result.outputs],
                        }
                        for result in results
                    ],
                }
            )
        )


def handle_inspect(
    args: argparse.Namespace,
    workspace: WorkspaceContext | None,
    cli_log_level: str | None,
    cli_log_outputs: list[LogOutputTarget],
) -> None:
    workspace_root = workspace.root if workspace is not None else None
    output = build_cli_output_config(
        transport=args.output_transport,
        fmt=args.output_format,
        directory=args.output_directory,
        output_encoding=args.output_encoding,
        output_compression=args.output_compression,
        workspace_root=workspace_root,
        view=args.output_view,
    )
    request = build_runtime_run_request(
        command="inspect",
        project=args.project,
        profile_name=args.profile,
        artifact_mode=args.artifact_mode,
        limit=args.limit,
        cli_output=output,
        command_observability=CommandObservability(
            visuals=args.visuals,
            heartbeat_interval_seconds=args.heartbeat_interval_seconds,
            log_level=cli_log_level,
            log_outputs=tuple(cli_log_outputs),
        ),
    )
    if request is None:
        logger.info("No enabled inspect profiles; skipping inspect.")
        return
    configure_profile_logging(cli_log_level, cli_log_outputs)
    execute_profile_request(request)
