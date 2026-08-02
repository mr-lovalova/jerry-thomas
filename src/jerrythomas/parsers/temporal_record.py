from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from jerrythomas.domain.record import TemporalRecord
from jerrythomas.sources.parser import DataParser
from jerrythomas.utils.time import parse_datetime


class TemporalRecordParser(DataParser[TemporalRecord]):
    """Rehydrate mapping-like rows into canonical TemporalRecord instances."""

    def __init__(self, *, time_field: str = "time") -> None:
        self._time_field = str(time_field).strip() or "time"

    def parse(self, raw: Any) -> TemporalRecord | None:
        if raw is None:
            return None
        if isinstance(raw, TemporalRecord):
            return raw

        fields = self._coerce_fields(raw)
        if self._time_field not in fields:
            raise ValueError(
                f"TemporalRecordParser expected field '{self._time_field}'"
            )

        record = TemporalRecord(time=self._coerce_time(fields.pop(self._time_field)))
        fields.pop("time", None)
        for key, value in fields.items():
            setattr(record, key, value)
        return record

    def _coerce_fields(self, raw: Any) -> dict[str, Any]:
        if isinstance(raw, Mapping):
            return dict(raw)

        attrs = getattr(raw, "__dict__", None)
        if isinstance(attrs, dict):
            return {
                key: value
                for key, value in attrs.items()
                if not str(key).startswith("_")
            }

        raise TypeError(
            "TemporalRecordParser expects a mapping, object with __dict__, "
            "or TemporalRecord instance"
        )

    @staticmethod
    def _coerce_time(value: Any) -> datetime:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value
        if isinstance(value, str):
            return parse_datetime(value)
        raise TypeError(
            "TemporalRecordParser expected a datetime or ISO-8601 string in the "
            "time field"
        )
