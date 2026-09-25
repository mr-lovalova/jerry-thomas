import logging
from dataclasses import dataclass
from pathlib import Path

from jerrythomas.cli.prompts import (
    choose_domain,
    choose_dto,
    choose_name,
    pick_from_list,
    pick_from_menu,
    prompt_required,
)
from jerrythomas.cli.source_options import SOURCE_TRANSPORTS, source_formats_for
from jerrythomas.cli.workspace import WorkspaceContext, resolve_scaffold_project_yaml
from jerrythomas.services.scaffold.discovery import (
    list_domains,
    list_dtos,
    list_mappers,
    list_parsers,
    list_sources,
)
from jerrythomas.services.scaffold.layout import (
    default_mapper_name,
    default_mapper_name_for_identity,
    default_parser_name,
    default_stream_id_for_source,
    dto_class_name,
    dto_module_path,
    source_id_parts,
)
from jerrythomas.services.scaffold.paths import default_project_yaml_path, pkg_root
from jerrythomas.services.scaffold.source_yaml import (
    DEFAULT_TEMPORAL_RECORD_PARSER_EP,
    default_loader_config,
    validate_source_id,
)
from jerrythomas.services.scaffold.stream_plan import (
    DomainCreation,
    DomainPlan,
    DomainReference,
    MapperCreation,
    MapperPlan,
    MapperReference,
    ParserCreation,
    ParserPlan,
    ParserReference,
    PythonType,
    SourceCreation,
    SourcePlan,
    SourceReference,
    StreamPlan,
    execute_stream_plan,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParserSelection:
    plan: ParserPlan
    dto_to_create: str | None


@dataclass(frozen=True)
class MapperSelection:
    plan: MapperPlan
    dto_to_create: str | None


def _parser_menu_options(parsers: dict[str, str]) -> list[tuple[str, str]]:
    base = [
        ("create", "Create new parser (default)"),
        ("temporal_record", "Temporal record rehydration"),
        ("identity", "Identity parser"),
    ]
    if parsers:
        return [base[0], ("existing", "Select existing parser"), *base[1:]]
    return base


def _mapper_menu_options(mappers: dict[str, str]) -> list[tuple[str, str]]:
    base = [
        ("create", "Create new mapper (default)"),
        ("identity", "Identity mapper"),
    ]
    if mappers:
        return [base[0], ("existing", "Select existing mapper"), base[1]]
    return base


def _select_new_source_loader() -> dict[str, object]:
    choice = pick_from_menu(
        "Loader:",
        [
            ("fs", "Built-in fs"),
            ("http", "Built-in http"),
            ("synthetic", "Built-in synthetic"),
            ("custom", "Custom loader"),
        ],
        allow_default=False,
    )
    if choice in SOURCE_TRANSPORTS:
        if choice in {"fs", "http"}:
            source_format = pick_from_menu(
                "Format:",
                [(name, name) for name in source_formats_for(choice)],
                allow_default=False,
            )
        else:
            source_format = None
        return default_loader_config(choice, source_format)
    return {"entrypoint": prompt_required("Loader entrypoint"), "args": {}}


def _select_parser_plan(
    plugin_root: Path | None,
    package_name: str,
    provider: str,
    dataset: str,
) -> ParserSelection:
    parsers = list_parsers(root=plugin_root)
    parser_choice = pick_from_menu("Parser:", _parser_menu_options(parsers))

    if parser_choice == "existing":
        entrypoint = pick_from_menu(
            "Select parser entrypoint:",
            [(key, key) for key in sorted(parsers)],
        )
        return ParserSelection(ParserReference(entrypoint), None)

    if parser_choice == "temporal_record":
        return ParserSelection(
            ParserReference(DEFAULT_TEMPORAL_RECORD_PARSER_EP),
            None,
        )

    if parser_choice == "identity":
        return ParserSelection(ParserReference("core.identity"), None)
    if parser_choice != "create":
        raise ValueError(f"Unknown parser choice: {parser_choice}")

    dto_map = list_dtos(root=plugin_root)
    default_dto = dto_class_name(f"{provider}_{dataset}")
    dto_class, dto_is_new = choose_dto(sorted(dto_map), default_dto)
    if dto_is_new:
        dto = PythonType(dto_class, dto_module_path(package_name, dto_class))
        dto_to_create = dto_class
    else:
        dto = PythonType(dto_class, dto_map[dto_class])
        dto_to_create = None
    parser_name = choose_name(
        "Parser class name",
        default=default_parser_name(dto_class),
    )
    return ParserSelection(
        ParserCreation(parser_name, dto),
        dto_to_create,
    )


def _select_mapper_plan(
    plugin_root: Path | None,
    package_name: str,
    provider: str,
    dataset: str,
    source: SourcePlan,
    domain: DomainPlan,
) -> MapperSelection:
    mappers = list_mappers(root=plugin_root)
    mapper_choice = pick_from_menu("Mapper:", _mapper_menu_options(mappers))
    if mapper_choice == "existing":
        entrypoint = pick_from_menu(
            "Select mapper entrypoint:",
            [(key, key) for key in sorted(mappers)],
        )
        return MapperSelection(MapperReference(entrypoint), None)
    if mapper_choice == "identity":
        return MapperSelection(MapperReference("core.identity"), None)
    if mapper_choice != "create":
        raise ValueError(f"Unknown mapper choice: {mapper_choice}")

    input_choice = pick_from_menu(
        "Mapper input:",
        [
            ("dto", "DTO (default)"),
            ("identity", "Any"),
        ],
    )
    logger.info("Domain output: Domain record")
    if input_choice == "identity":
        input_type = PythonType("Any", "typing")
        mapper_name = choose_name(
            "Mapper name",
            default=default_mapper_name_for_identity(domain.name),
        )
        return MapperSelection(MapperCreation(mapper_name, input_type), None)
    if input_choice != "dto":
        raise ValueError(f"Unknown mapper input choice: {input_choice}")

    parser_dto = None
    if isinstance(source, SourceCreation) and isinstance(source.parser, ParserCreation):
        parser_dto = source.parser.dto
    if parser_dto is not None:
        input_type = PythonType(parser_dto.class_name, parser_dto.module)
        dto_to_create = None
    else:
        dto_map = list_dtos(root=plugin_root)
        dto_class, dto_is_new = choose_dto(
            sorted(dto_map),
            dto_class_name(f"{provider}_{dataset}"),
        )
        if dto_is_new:
            input_type = PythonType(
                dto_class,
                dto_module_path(package_name, dto_class),
            )
            dto_to_create = dto_class
        else:
            input_type = PythonType(dto_class, dto_map[dto_class])
            dto_to_create = None
    mapper_name = choose_name(
        "Mapper name",
        default=default_mapper_name(input_type.module, domain.name),
    )
    return MapperSelection(
        MapperCreation(mapper_name, input_type),
        dto_to_create,
    )


def _collect_stream_plan(
    plugin_root: Path | None,
    package_name: str,
    project_yaml: Path,
) -> StreamPlan:
    dto_to_create = None

    source_choice = pick_from_menu(
        "Source:",
        [
            ("create", "Create new source (default)"),
            ("existing", "Select existing source"),
        ],
    )
    if source_choice == "existing":
        sources = list_sources(project_yaml)
        if not sources:
            raise SystemExit("No sources found. Create one first.")
        source_id = pick_from_list("Select source:", sources)
        source_provider, source_dataset, _ = source_id_parts(source_id)
        if source_provider is None or source_dataset is None:
            raise ValueError(f"Invalid source id: {source_id}")
        provider = source_provider
        dataset = source_dataset
        source: SourcePlan = SourceReference(source_id)
    elif source_choice == "create":
        provider = prompt_required("Provider name (e.g. nasa)")
        dataset = prompt_required("Dataset name (e.g. weather)")
        source_id = choose_name("Source id", default=f"{provider}.{dataset}")
        try:
            validate_source_id(source_id)
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
        loader = _select_new_source_loader()
        parser_selection = _select_parser_plan(
            plugin_root,
            package_name,
            provider,
            dataset,
        )
        source = SourceCreation(
            source_id=source_id,
            loader=loader,
            parser=parser_selection.plan,
        )
        dto_to_create = parser_selection.dto_to_create
    else:
        raise ValueError(f"Unknown source choice: {source_choice}")

    domain_name, domain_is_new = choose_domain(
        list_domains(root=plugin_root),
        dataset,
    )
    domain: DomainPlan
    if domain_is_new:
        domain = DomainCreation(domain_name)
    else:
        domain = DomainReference(domain_name)

    mapper_selection = _select_mapper_plan(
        plugin_root,
        package_name,
        provider,
        dataset,
        source,
        domain,
    )
    if dto_to_create is not None and mapper_selection.dto_to_create is not None:
        raise ValueError("A stream plan cannot create two DTOs")
    if mapper_selection.dto_to_create is not None:
        dto_to_create = mapper_selection.dto_to_create
    stream_id = choose_name(
        "Stream id",
        default=default_stream_id_for_source(domain.name, source.source_id),
    )
    return StreamPlan(
        project_yaml=project_yaml,
        stream_id=stream_id,
        root=plugin_root,
        source=source,
        mapper=mapper_selection.plan,
        domain=domain,
        dto_to_create=dto_to_create,
    )


def handle(
    *,
    plugin_root: Path | None = None,
    workspace: WorkspaceContext | None = None,
    project: str | None = None,
) -> None:
    project_yaml = resolve_scaffold_project_yaml(project, workspace)
    root_dir, package_name, _ = pkg_root(plugin_root)
    project_yaml = project_yaml or default_project_yaml_path(root_dir)
    plan = _collect_stream_plan(plugin_root, package_name, project_yaml)
    try:
        result = execute_stream_plan(plan)
    except (FileExistsError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
    logger.info("Stream: %s", result.path)
    if result.entry_points_changed:
        logger.info("Reinstall plugin: pip install -e %s", root_dir)
