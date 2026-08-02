import logging

from jerrythomas.cli.workspace import WorkspaceContext
from jerrythomas.services.path_policy import resolve_workspace_path
from jerrythomas.services.scaffold.plugin import scaffold_plugin

logger = logging.getLogger(__name__)


def handle(
    name: str,
    out: str,
    workspace: WorkspaceContext | None = None,
) -> None:
    outdir = resolve_workspace_path(
        out,
        workspace.root if workspace is not None else None,
    )
    try:
        target = scaffold_plugin(name, outdir)
    except (FileExistsError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
    logger.info("Plugin: %s", target)
