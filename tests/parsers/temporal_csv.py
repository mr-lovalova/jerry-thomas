from datetime import datetime, timezone
from typing import Any, Mapping
from zoneinfo import ZoneInfo

from jerrythomas.domain.record import TemporalRecord
from jerrythomas.sources.parser import DataParser
from jerrythomas.utils.time import parse_datetime


def _as_timezone(name: str | None) -> timezone:
    if not name:
        return timezone.utc
    if name.upper() == "UTC":
        return timezone.utc
    try:
        return ZoneInfo(name)
    except Exception:
        return timezone.utc


def _parse_time(text: Any, time_format: str | None, tz: timezone) -> datetime:
    if text is None:
        raise ValueError("time value missing")
    if isinstance(text, datetime):
        dt = text
    else:
        raw = str(text).strip()
        if time_format:
            dt = datetime.strptime(raw, time_format)
        else:
            normalized = raw.replace(" ", "T") if "T" not in raw else raw
            dt = parse_datetime(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    else:
        dt = dt.astimezone(tz)
    return dt


def _parse_value(value: Any, decimal: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    raw = str(value).strip()
    if raw == "":
        return None
    normalized = raw.replace(decimal, ".") if decimal and decimal != "." else raw
    try:
        return float(normalized)
    except ValueError:
        return None


class TemporalCsvValueParser(DataParser[TemporalRecord]):
    """Parse CSV row mappings into TemporalRecord instances (test-only)."""

    def __init__(
        self,
        *,
        time_field: str,
        value_field: str,
        time_format: str | None = None,
        timezone_name: str | None = "UTC",
        decimal: str = ".",
        attributes: Mapping[str, str] | None = None,
    ) -> None:
        self.time_field = time_field
        self.value_field = value_field
        self.time_format = time_format
        self.tz = _as_timezone(timezone_name)
        self.decimal = decimal
        self.attributes = dict(attributes or {})

    def parse(self, raw: Mapping[str, Any]) -> TemporalRecord | None:
        time_raw = raw.get(self.time_field)
        value_raw = raw.get(self.value_field)

        ts = _parse_time(time_raw, self.time_format, self.tz)
        value = _parse_value(value_raw, self.decimal)

        rec = TemporalRecord(time=ts)
        setattr(rec, "value", value)
        for attr, field in self.attributes.items():
            setattr(rec, attr, raw.get(field))
        return rec
