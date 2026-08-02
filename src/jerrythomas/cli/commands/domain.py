import logging
from pathlib import Path

from jerrythomas.services.scaffold.domain import create_domain

logger = logging.getLogger(__name__)


def handle(domain: str, plugin_root: Path | None = None) -> None:
    try:
        path = create_domain(domain, plugin_root)
    except (FileExistsError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
    logger.info("Domain: %s", path)
