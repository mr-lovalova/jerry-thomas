import logging
from pathlib import Path

from jerrythomas.cli.prompts import prompt_required
from jerrythomas.services.scaffold.dto import create_dto

logger = logging.getLogger(__name__)


def handle(name: str | None, *, plugin_root: Path | None = None) -> None:
    if not name:
        name = prompt_required("DTO class name")
    try:
        path = create_dto(name, plugin_root)
    except (FileExistsError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
    logger.info("DTO: %s", path)
