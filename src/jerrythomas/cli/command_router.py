import argparse
from pathlib import Path

from jerrythomas.cli.commands.clean import handle as handle_clean
from jerrythomas.cli.commands.demo import handle as handle_demo
from jerrythomas.cli.commands.domain import handle as handle_domain
from jerrythomas.cli.commands.dto import handle as handle_dto
from jerrythomas.cli.commands.inflow import handle as handle_inflow
from jerrythomas.cli.commands.list_ import handle as handle_list
from jerrythomas.cli.commands.loader import handle as handle_loader
from jerrythomas.cli.commands.mapper import handle as handle_mapper
from jerrythomas.cli.commands.materialize import handle as handle_materialize
from jerrythomas.cli.commands.parser import handle as handle_parser
from jerrythomas.cli.commands.plugin import handle as handle_plugin
from jerrythomas.cli.commands.profile_runner import (
    handle_build,
    handle_inspect,
    handle_serve,
)
from jerrythomas.cli.commands.source import handle as handle_source
from jerrythomas.cli.commands.stream import handle as handle_stream_create
from jerrythomas.cli.commands.version import handle as handle_version
from jerrythomas.cli.commands.version import handle_env
from jerrythomas.cli.workspace import WorkspaceContext
from jerrythomas.execution.settings import LogOutputTarget


def execute_command(
    args: argparse.Namespace,
    plugin_root: Path | None,
    workspace_context: WorkspaceContext | None,
    cli_level_arg: str | None,
    cli_log_outputs: list[LogOutputTarget],
) -> None:
    match args.cmd:
        case "version":
            handle_version()
        case "env":
            handle_env()
        case "build":
            handle_build(
                args=args,
                cli_log_level=cli_level_arg,
                cli_log_outputs=cli_log_outputs,
            )
        case "serve":
            handle_serve(
                args=args,
                workspace=workspace_context,
                cli_log_level=cli_level_arg,
                cli_log_outputs=cli_log_outputs,
            )
        case "inspect":
            handle_inspect(
                args=args,
                workspace=workspace_context,
                cli_log_level=cli_level_arg,
                cli_log_outputs=cli_log_outputs,
            )
        case "clean":
            handle_clean(yes=args.yes, older_than=args.older_than)
        case "materialize":
            handle_materialize(
                project=args.project,
                profile_name=args.profile,
                output=args.output,
                overwrite=args.overwrite,
                artifact_mode=args.artifact_mode,
                visuals=args.visuals,
                heartbeat_interval_seconds=args.heartbeat_interval_seconds,
                cli_log_level=cli_level_arg,
                cli_log_outputs=cli_log_outputs,
                workspace=workspace_context,
            )
        case "source":
            handle_source(
                source_id=args.source_id,
                transport=args.transport,
                format=args.format,
                loader=args.loader,
                parser=args.parser,
                plugin_root=plugin_root,
                workspace=workspace_context,
            )
        case "list":
            handle_list(
                subcmd=args.list_cmd,
                plugin_root=plugin_root,
                workspace=workspace_context,
            )
        case "domain":
            handle_domain(
                domain=args.domain_name,
                plugin_root=plugin_root,
            )
        case "dto":
            handle_dto(name=args.name, plugin_root=plugin_root)
        case "parser":
            handle_parser(name=args.name, plugin_root=plugin_root)
        case "mapper":
            handle_mapper(name=args.name, plugin_root=plugin_root)
        case "loader":
            handle_loader(name=args.name, plugin_root=plugin_root)
        case "inflow":
            handle_inflow(plugin_root=plugin_root, workspace=workspace_context)
        case "stream":
            handle_stream_create(
                plugin_root=plugin_root,
                use_identity=args.identity,
                workspace=workspace_context,
            )
        case "plugin":
            handle_plugin(
                name=args.plugin_name,
                out=args.out,
                workspace=workspace_context,
            )
        case "demo":
            handle_demo(
                subcmd=args.demo_cmd,
                out=args.out,
                workspace=workspace_context,
            )
        case _:
            raise ValueError(f"Unsupported command: {args.cmd!r}")
