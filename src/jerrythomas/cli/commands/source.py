import logging
import sys
from pathlib import Path

from jerrythomas.cli.prompts import pick_from_menu, prompt_required
from jerrythomas.cli.source_options import SOURCE_TRANSPORTS, source_formats_for
from jerrythomas.cli.workspace import WorkspaceContext, resolve_default_project_yaml
from jerrythomas.services.scaffold.discovery import list_loaders, list_parsers
from jerrythomas.services.scaffold.source_yaml import (
    DEFAULT_TEMPORAL_RECORD_PARSER_EP,
    create_source_yaml,
    default_loader_config,
    validate_source_id,
)

logger = logging.getLogger(__name__)


def _resolve_source_id(source_id: str | None) -> str:
    if source_id is not None:
        return source_id
    return prompt_required("Source id (provider.dataset)")


def _choose_loader_transport_or_entrypoint(
    plugin_root: Path | None,
) -> tuple[str | None, str | None]:
    known_loaders = list_loaders(root=plugin_root)
    options = [
        ("fs", "Built-in fs"),
        ("http", "Built-in http"),
        ("synthetic", "Built-in synthetic"),
    ]
    if known_loaders:
        options.append(("existing", "Select existing loader"))
    options.append(("custom", "Custom loader"))

    choice = pick_from_menu("Loader:", options)
    if choice in SOURCE_TRANSPORTS:
        return choice, None
    if choice == "existing":
        loader_ep = pick_from_menu(
            "Select loader entrypoint:",
            [(key, key) for key in sorted(known_loaders.keys())],
        )
        return None, loader_ep
    if choice == "custom":
        return None, prompt_required("Loader entrypoint")
    return None, None


def _resolve_loader_config(
    transport: str | None,
    source_format: str | None,
    loader: str | None,
    plugin_root: Path | None,
) -> dict[str, object]:
    if loader:
        return {"entrypoint": loader, "args": {}}

    loader_ep = None
    selected_transport = transport
    if not selected_transport:
        selected_transport, loader_ep = _choose_loader_transport_or_entrypoint(
            plugin_root
        )

    if loader_ep:
        return {"entrypoint": loader_ep, "args": {}}

    if selected_transport in {"fs", "http"} and not source_format:
        source_format = pick_from_menu(
            "Format:",
            [(name, name) for name in source_formats_for(selected_transport)],
        )
    if not selected_transport:
        logger.error("--transport is required when no --loader is provided")
        raise SystemExit(2)
    return default_loader_config(selected_transport, source_format)


def _select_parser_from_menu(plugin_root: Path | None) -> str:
    parsers = list_parsers(root=plugin_root)
    if parsers:
        choice = pick_from_menu(
            "Parser:",
            [
                ("existing", "Select existing parser (default)"),
                ("temporal_record", "Temporal record rehydration"),
                ("identity", "Identity parser"),
                ("custom", "Custom parser"),
            ],
        )
        if choice == "existing":
            return pick_from_menu(
                "Select parser entrypoint:",
                [(key, key) for key in sorted(parsers.keys())],
            )
        if choice == "temporal_record":
            return DEFAULT_TEMPORAL_RECORD_PARSER_EP
        if choice == "identity":
            return "identity"
        return prompt_required("Parser entrypoint")

    choice = pick_from_menu(
        "Parser:",
        [
            ("identity", "Identity parser (default)"),
            ("temporal_record", "Temporal record rehydration"),
            ("custom", "Custom parser"),
        ],
    )
    if choice == "temporal_record":
        return DEFAULT_TEMPORAL_RECORD_PARSER_EP
    if choice == "identity":
        return "identity"
    return prompt_required("Parser entrypoint")


def _resolve_parser_entrypoint(
    parser: str | None,
    plugin_root: Path | None,
) -> str:
    if parser:
        return parser
    if not sys.stdin.isatty():
        return "identity"
    return _select_parser_from_menu(plugin_root)


def handle(
    source_id: str | None,
    transport: str | None = None,
    format: str | None = None,
    loader: str | None = None,
    parser: str | None = None,
    plugin_root: Path | None = None,
    workspace: WorkspaceContext | None = None,
) -> None:
    source_id = _resolve_source_id(source_id)
    try:
        validate_source_id(source_id)
        loader_config = _resolve_loader_config(
            transport,
            format,
            loader,
            plugin_root,
        )
    except ValueError as exc:
        logger.error("%s", exc)
        raise SystemExit(2) from None
    parser_ep = _resolve_parser_entrypoint(parser, plugin_root)

    project_yaml = resolve_default_project_yaml(workspace)
    try:
        path = create_source_yaml(
            source_id=source_id,
            loader=loader_config,
            parser_ep=parser_ep,
            root=plugin_root,
            project_yaml=project_yaml,
        )
    except (FileExistsError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
    logger.info("Source: %s", path)
