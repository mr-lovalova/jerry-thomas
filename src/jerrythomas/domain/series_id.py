from math import isfinite
from urllib.parse import quote


SERIES_ID_SEPARATOR = "__"
SERIES_ID_COMPONENT_SEPARATOR = "|"


def base_id(series_id: str) -> str:
    base, separator, suffix = series_id.partition(SERIES_ID_SEPARATOR)
    if not separator:
        return series_id
    if not base or not suffix:
        raise ValueError(f"Invalid partitioned series id {series_id!r}")
    return base


def make_partitioned_series_id(base: str, suffix: str) -> str:
    if SERIES_ID_SEPARATOR in base:
        raise ValueError(
            "Series base id must not contain reserved separator "
            f"{SERIES_ID_SEPARATOR!r}"
        )
    return f"{base}{SERIES_ID_SEPARATOR}{suffix}" if suffix else base


def encode_series_id_component(field: str, value: object) -> str:
    if not field:
        raise ValueError("series identity fields must not be empty")
    encoded_field = quote(field, safe="")
    if value is None:
        encoded_value = "!n"
    elif type(value) is str:
        encoded_value = quote(value, safe="")
    elif type(value) is bool:
        encoded_value = f"!b:{int(value)}"
    elif type(value) is int:
        encoded_value = f"!i:{value}"
    elif type(value) is float:
        if not isfinite(value):
            raise ValueError(
                f"Series identity field {field!r} must contain a finite float."
            )
        encoded_value = f"!f:{value.hex()}"
    else:
        raise TypeError(
            f"Series identity field {field!r} must contain a string, integer, "
            f"float, boolean, or null; got {type(value).__name__}."
        )
    return f"@{encoded_field}:{encoded_value}"
