from collections.abc import Callable
from typing import Any

from jerrythomas.io.normalization import (
    View,
    json_text,
    payload_for_view,
    raw_payload,
)


def json_line_serializer(view: View = "raw") -> Callable[[Any], str]:
    def serialize(item: Any) -> str:
        return json_text(payload_for_view(item, view)) + "\n"

    return serialize


def text_line_serializer() -> Callable[[Any], str]:
    def serialize(item: Any) -> str:
        raw = raw_payload(item)
        if isinstance(raw, str):
            text = raw
        elif isinstance(raw, dict) and isinstance(raw.get("text"), str):
            text = raw["text"]
        else:
            text = json_text(raw)
        return text if text.endswith("\n") else f"{text}\n"

    return serialize
