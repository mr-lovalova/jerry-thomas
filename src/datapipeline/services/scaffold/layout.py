import re


def to_snake(name: str) -> str:
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def _pascal_case(name: str) -> str:
    snake = to_snake(name)
    prefix_length = len(snake) - len(snake.lstrip("_"))
    words = snake[prefix_length:].split("_")
    return snake[:prefix_length] + "".join(part.capitalize() for part in words if part)


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")


def ep_key_from_name(name: str) -> str:
    return to_snake(name)


# Directory names
DIR_DTOS = "dtos"
DIR_PARSERS = "parsers"
DIR_LOADERS = "loaders"
DIR_MAPPERS = "mappers"
DIR_DOMAINS = "domains"

# Template paths
TPL_DTO = "dto.py.j2"
TPL_PARSER = "parser.py.j2"
TPL_LOADER_BASIC = "loaders/basic.py.j2"
TPL_MAPPER_SOURCE = "mappers/source.py.j2"
TPL_DOMAIN_RECORD = "record.py.j2"


def loader_class_name(name: str) -> str:
    class_name = _pascal_case(name)
    return class_name if class_name.endswith("Loader") else f"{class_name}Loader"


def domain_record_class(domain: str) -> str:
    return f"{_pascal_case(domain)}Record"


def dto_class_name(base: str) -> str:
    return f"{_pascal_case(base)}DTO"


def dto_module_path(package: str, dto_class: str) -> str:
    return f"{package}.{DIR_DTOS}.{to_snake(dto_class)}"


def default_parser_name(dto_class: str) -> str:
    return f"{dto_class}Parser"


def default_mapper_name(input_module: str, domain: str) -> str:
    input_mod = input_module.rsplit(".", 1)[-1]
    return f"map_{input_mod}_to_{domain}"


def default_stream_id(domain: str, dataset: str, variant: str | None = None) -> str:
    base = f"{slugify(domain)}.{slugify(dataset)}"
    return f"{base}.{slugify(variant)}" if variant else base


def source_id_parts(source_id: str) -> tuple[str | None, str | None, str | None]:
    parts = [part for part in source_id.split(".") if part]
    if len(parts) < 2:
        return None, None, None
    provider, dataset, *variant_parts = parts
    variant = ".".join(variant_parts) if variant_parts else None
    return provider, dataset, variant


def default_stream_id_for_source(
    domain: str,
    source_id: str,
    variant: str | None = None,
    *,
    fallback_dataset: str = "dataset",
) -> str:
    _provider, dataset, source_variant = source_id_parts(source_id)
    return default_stream_id(
        domain, dataset or fallback_dataset, variant or source_variant
    )


def default_mapper_name_for_identity(domain: str) -> str:
    return f"map_identity_to_{slugify(domain)}"
