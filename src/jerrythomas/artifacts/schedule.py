import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from jerrythomas.utils.time import parse_datetime


@dataclass(frozen=True)
class Schedule:
    partition_by: tuple[str, ...]
    times: dict[tuple, list[datetime]]

    def times_for(self, key: tuple) -> list[datetime]:
        return self.times.get(key, [])


def read_schedule(path: Path, partition_by: tuple[str, ...]) -> Schedule:
    times: dict[tuple, list[datetime]] = {}
    previous_key: tuple | None = None
    expected_fields = set(partition_by)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            row_fields = {field for field in row if field != "time"}
            if row_fields != expected_fields:
                raise ValueError(
                    f"Schedule artifact '{path}' row partition fields "
                    f"{sorted(row_fields)!r} do not match expected partition_by "
                    f"{list(partition_by)!r}."
                )
            value = row.get("time")
            if value is None:
                raise ValueError(
                    f"Schedule artifact '{path}' contains a row without time."
                )
            key = tuple(row[field] for field in partition_by)
            time = parse_datetime(str(value))
            values = times.setdefault(key, [])
            if (
                previous_key is not None and key != previous_key and key <= previous_key
            ) or (values and time <= values[-1]):
                raise ValueError(
                    f"Schedule artifact '{path}' rows must be strictly ordered by "
                    f"{[*partition_by, 'time']!r}."
                )
            values.append(time)
            previous_key = key
    return Schedule(
        partition_by=partition_by,
        times=times,
    )


def schedule_partition_by_from_metadata(
    artifact_id: str, meta: Mapping[str, Any]
) -> tuple[str, ...]:
    partition_by = meta.get("partition_by")
    if partition_by is None:
        raise RuntimeError(
            f"Schedule artifact '{artifact_id}' metadata field 'partition_by' "
            "is required."
        )
    if not isinstance(partition_by, list) or any(
        not isinstance(field, str) for field in partition_by
    ):
        raise RuntimeError(
            f"Schedule artifact '{artifact_id}' metadata field 'partition_by' must be "
            "a list of strings."
        )
    return tuple(partition_by)
