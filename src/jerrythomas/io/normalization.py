import json
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from typing import Any, Literal

from jerrythomas.domain.record import TemporalRecord, public_record_fields
from jerrythomas.domain.sample import Sample
from jerrythomas.domain.value import normalize_data_value

View = Literal["flat", "raw"]


def payload_for_view(item: Any, view: View = "raw") -> Any:
    if view == "raw":
        return raw_payload(item)
    if view == "flat":
        return flat_payload(item)
    raise ValueError(f"Unsupported view '{view}'")


def raw_payload(item: Any) -> Any:
    return _jsonable(item)


def flat_payload(item: Any) -> dict[str, Any]:
    if isinstance(item, Sample):
        payload: dict[str, Any] = {}
        flatten_fields("key", item.key, payload)
        flatten_fields("features", item.features.values, payload)
        if item.targets is not None:
            flatten_fields("targets", item.targets.values, payload)
        return payload

    raw = raw_payload(item)
    flattened: dict[str, Any] = {}
    if isinstance(raw, dict):
        for key, value in sorted(raw.items(), key=lambda kv: str(kv[0])):
            flatten_fields(str(key), value, flattened)
        return flattened

    flatten_fields("value", raw, flattened)
    return flattened


def flatten_fields(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if _is_scalar(value):
        _set_flat_field(out, prefix, value)
        return
    if isinstance(value, Mapping):
        for key, nested in sorted(value.items(), key=lambda kv: str(kv[0])):
            flatten_fields(f"{prefix}.{_json_key(key)}", nested, out)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if all(_is_scalar(item) for item in value):
            for idx, nested in enumerate(value):
                _set_flat_field(out, f"{prefix}.{idx}", nested)
            return
        _set_flat_field(out, prefix, json_text(raw_payload(value)))
        return
    raise TypeError(f"Unsupported output value type: {type(value).__name__}")


def _set_flat_field(out: dict[str, Any], field: str, value: Any) -> None:
    if field in out:
        raise ValueError(f"Flat output field {field!r} is produced more than once.")
    out[field] = normalize_data_value(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, float):
        return normalize_data_value(value)
    if _is_scalar(value):
        return value
    if isinstance(value, TemporalRecord):
        return {
            name: _jsonable(field_value)
            for name, field_value in public_record_fields(value).items()
        }
    if not isinstance(value, type) and is_dataclass(value):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in fields(value)
            if not field.name.startswith("_")
        }
    if isinstance(value, Mapping):
        return {_json_key(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(v) for v in value]
    raise TypeError(f"Unsupported output value type: {type(value).__name__}")


def _json_key(value: Any) -> str:
    if isinstance(value, str):
        return value
    raise TypeError(f"Unsupported output mapping key type: {type(value).__name__}")


def json_text(payload: Any, indent: int | None = None) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=indent,
        default=_encode_temporal,
        allow_nan=False,
    )


def _encode_temporal(value: Any) -> str:
    if isinstance(value, (datetime, date)):
        return str(value)
    raise TypeError(f"Unsupported output value type: {type(value).__name__}")


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool, datetime, date))
