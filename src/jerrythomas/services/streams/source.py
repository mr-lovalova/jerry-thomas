from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

from jerrythomas.config.sources import (
    EntryPointConfig,
    FsLoaderConfig,
    HttpLoaderConfig,
    SourceConfig,
)
from jerrythomas.config.interpolation import normalize_interpolated_args
from jerrythomas.plugins import (
    LOADERS_EP,
    MAPPERS_EP,
    PARSERS_EP,
    load_entrypoint,
)
from jerrythomas.services.path_policy import resolve_relative_fs_loader_path
from jerrythomas.sources.factory import build_builtin_loader
from jerrythomas.sources.source import Source


def build_source(config: SourceConfig, project_yaml: Path) -> Source:
    parser_factory = load_entrypoint(PARSERS_EP, config.parser.entrypoint)
    if isinstance(config.loader, (FsLoaderConfig, HttpLoaderConfig)):
        loader_config = config.loader
        if isinstance(loader_config, FsLoaderConfig):
            loader_config = loader_config.model_copy(
                update={
                    "path": resolve_relative_fs_loader_path(
                        loader_config.path,
                        project_yaml.parent.resolve(),
                    )
                }
            )
        loader = build_builtin_loader(loader_config)
    else:
        loader_factory = load_entrypoint(LOADERS_EP, config.loader.entrypoint)
        loader_args = normalize_interpolated_args(config.loader.args)
        loader = loader_factory(**loader_args)
    parser_args = normalize_interpolated_args(config.parser.args)
    return Source(
        loader=loader,
        parser=parser_factory(**parser_args),
    )


def build_mapper(
    config: EntryPointConfig,
) -> Callable[[Iterator[Any]], Iterable[Any]]:
    mapper = load_entrypoint(MAPPERS_EP, config.entrypoint)
    args = normalize_interpolated_args(config.args)
    if args:
        return lambda records: mapper(records, **args)
    return mapper
