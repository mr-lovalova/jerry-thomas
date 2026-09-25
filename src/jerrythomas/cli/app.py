import argparse
import logging
import sys

from jerrythomas.cli.command_router import execute_command
from jerrythomas.cli.workspace import (
    WorkspaceContext,
    load_workspace_context,
    resolve_project_from_args,
)
from jerrythomas.cli.logging_setup import (
    configure_cli_logging,
    parse_log_output_specs,
)
from jerrythomas.cli.parser_builder import build_parser
from jerrythomas.execution.settings import LogOutputTarget
from jerrythomas.profiles.errors import ProfileCommandError
from jerrythomas.services.path_policy import resolve_workspace_path, workspace_cwd


logger = logging.getLogger(__name__)
_PROFILE_COMMANDS = {"build", "inspect", "export", "serve"}


def _resolve_project_arguments(
    args: argparse.Namespace,
    workspace_context: WorkspaceContext | None,
) -> None:
    is_project_listing = args.cmd == "list" and args.list_cmd in {
        "datasets",
        "streams",
        "profiles",
    }
    explicit_source_project = (
        args.cmd == "list" and args.list_cmd == "sources" and args.project is not None
    )
    if args.cmd not in _PROFILE_COMMANDS and not (
        is_project_listing or explicit_source_project
    ):
        return
    args.project = resolve_project_from_args(args.project, workspace_context)


def _configure_cli_logging(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    workspace_context: WorkspaceContext | None,
) -> tuple[str | None, list[LogOutputTarget]]:
    cli_level_arg = args.log_level
    cli_log_output_specs = args.log_output

    try:
        cli_log_outputs = parse_log_output_specs(
            cli_log_output_specs,
            resolve_global_path=lambda value: resolve_workspace_path(
                value,
                workspace_context.root if workspace_context is not None else None,
            ),
        )
        initial_log_outputs = cli_log_outputs
        if args.cmd in _PROFILE_COMMANDS:
            initial_log_outputs = [
                output for output in cli_log_outputs if output.transport != "fs"
            ]
        configure_cli_logging(
            cli_level_arg,
            initial_log_outputs,
        )
    except ValueError as exc:
        parser.error(str(exc))

    return cli_level_arg, cli_log_outputs


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    workspace_context = None
    if args.cmd not in {"version", "env", "clean"}:
        try:
            workspace_context = load_workspace_context(workspace_cwd())
        except (OSError, TypeError, ValueError) as exc:
            parser.error(f"Failed to load workspace: {exc}")
    _resolve_project_arguments(args=args, workspace_context=workspace_context)
    cli_level_arg, cli_log_outputs = _configure_cli_logging(
        parser=parser,
        args=args,
        workspace_context=workspace_context,
    )
    plugin_root = workspace_context.resolve_plugin_root() if workspace_context else None

    try:
        execute_command(
            args=args,
            plugin_root=plugin_root,
            workspace_context=workspace_context,
            cli_level_arg=cli_level_arg,
            cli_log_outputs=cli_log_outputs,
        )
    except ProfileCommandError as exc:
        logger.error("%s", exc)
        for note in getattr(exc, "__notes__", ()):
            logger.error("%s", note)
        raise SystemExit(2) from exc
    except KeyboardInterrupt:
        message = (
            "Serve interrupted by user"
            if args.cmd == "serve"
            else "Interrupted by user"
        )
        print(message, file=sys.stderr)
        raise SystemExit(130) from None
