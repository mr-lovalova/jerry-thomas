import logging
from pathlib import Path

from jerrythomas.cli.prompts import choose_name
from jerrythomas.services.scaffold.loader import create_loader

logger = logging.getLogger(__name__)


def handle(name: str | None, *, plugin_root: Path | None = None) -> None:
    if not name:
        name = choose_name("Loader name", default="custom_loader")
    try:
        entrypoint = create_loader(name=name, root=plugin_root)
    except (FileExistsError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
    logger.info("Loader entry point: %s", entrypoint)
