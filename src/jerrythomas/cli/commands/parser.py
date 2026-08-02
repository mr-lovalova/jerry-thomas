import logging
from pathlib import Path

from jerrythomas.cli.prompts import choose_dto, choose_name
from jerrythomas.services.scaffold.discovery import list_dtos
from jerrythomas.services.scaffold.dto import (
    create_dto,
    dto_scaffold_paths,
    validate_dto_creation,
)
from jerrythomas.services.scaffold.layout import (
    default_parser_name,
    dto_module_path,
)
from jerrythomas.services.scaffold.locking import acquire_scaffold_lock
from jerrythomas.services.scaffold.parser import (
    create_parser,
    parser_scaffold_paths,
    validate_parser_creation,
)
from jerrythomas.services.scaffold.paths import pkg_root
from jerrythomas.services.scaffold.utils import rollback_new_scaffold_paths

logger = logging.getLogger(__name__)


def handle(
    name: str | None,
    *,
    plugin_root: Path | None = None,
) -> str:
    dto_map = list_dtos(root=plugin_root)
    dto_class, should_create_dto = choose_dto(
        sorted(dto_map),
        f"{name}DTO" if name else None,
    )

    _, package_name, pyproject = pkg_root(plugin_root)
    if should_create_dto:
        dto_module = dto_module_path(package_name, dto_class)
    else:
        dto_module = dto_map[dto_class]

    if not name:
        name = choose_name("Parser class name", default=default_parser_name(dto_class))

    paths = list(parser_scaffold_paths(name, plugin_root))
    if should_create_dto:
        paths.extend(dto_scaffold_paths(dto_class, plugin_root))
    try:
        with acquire_scaffold_lock(pyproject.parent) as scaffold_lock:
            if should_create_dto:
                validate_dto_creation(dto_class, plugin_root)
            validate_parser_creation(name, plugin_root)
            with rollback_new_scaffold_paths(paths):
                if should_create_dto:
                    create_dto(
                        dto_class,
                        plugin_root,
                        scaffold_lock=scaffold_lock,
                    )
                entrypoint = create_parser(
                    name=name,
                    dto_class=dto_class,
                    dto_module=dto_module,
                    root=plugin_root,
                    scaffold_lock=scaffold_lock,
                )
    except (FileExistsError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
    logger.info("Parser entry point: %s", entrypoint)
    return entrypoint
