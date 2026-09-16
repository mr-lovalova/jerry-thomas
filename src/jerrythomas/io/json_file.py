import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from jerrythomas.io.sinks.files import AtomicTextFileSink


def read_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in '{path}'")
    return payload


def write_json_object(
    path: Path, payload: Mapping[str, Any], *, overwrite: bool = True
) -> None:
    sink = AtomicTextFileSink(path, overwrite=overwrite, newline="\n")
    try:
        json.dump(payload, sink.fh, indent=2, sort_keys=True, allow_nan=False)
        sink.close()
    except BaseException:
        sink.abort()
        raise
