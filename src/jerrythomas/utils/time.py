import re
from datetime import datetime, timedelta, timezone


CADENCE_PATTERN = r"^(0*[1-9]\d*)(min|m|h|d)$"
_CADENCE_PATTERN = re.compile(CADENCE_PATTERN)
_TIMECODE_PATTERN = re.compile(r"^([+-]?\d+)(s|min|m|h|d)$")
_SECONDS_PER_UNIT = {
    "s": 1,
    "m": 60,
    "min": 60,
    "h": 60 * 60,
    "d": 24 * 60 * 60,
}
_UTC_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def parse_timecode(value: str) -> timedelta:
    """Parse a signed duration such as 30s, 10min, -1h, or 2d."""
    if not isinstance(value, str):
        raise ValueError("parse_timecode expects a string")
    match = _TIMECODE_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"Unsupported timecode: {value}")

    amount = int(match.group(1))
    unit = match.group(2)
    return timedelta(seconds=amount * _SECONDS_PER_UNIT[unit])


def parse_cadence(value: str) -> timedelta:
    """Parse a positive dataset cadence in minutes, hours, or days."""
    if not isinstance(value, str):
        raise ValueError(f"Unsupported cadence: {value}")
    match = _CADENCE_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"Unsupported cadence: {value}")

    return timedelta(seconds=int(match.group(1)) * _SECONDS_PER_UNIT[match.group(2)])


def floor_time_to_cadence(ts: datetime, cadence: timedelta) -> datetime:
    if ts.tzinfo is None:
        epoch = _UTC_EPOCH.replace(tzinfo=None)
        return epoch + ((ts - epoch) // cadence) * cadence

    utc_timestamp = ts.astimezone(timezone.utc)
    floored = _UTC_EPOCH + ((utc_timestamp - _UTC_EPOCH) // cadence) * cadence
    return floored.astimezone(ts.tzinfo)


def count_cadence_buckets(
    start: datetime,
    end: datetime,
    cadence: timedelta,
) -> int:
    anchored_start = floor_time_to_cadence(start, cadence)
    anchored_end = floor_time_to_cadence(end, cadence)
    if anchored_end < anchored_start:
        return 0
    return ((anchored_end - anchored_start) // cadence) + 1


def parse_datetime(value: str) -> datetime:
    """Parse an ISO-8601 datetime.

    - Accepts 'Z' or numeric offsets (e.g. '+00:00', '+01:30').
    - If input is timezone-naive, assume UTC.
    """
    if not isinstance(value, str):
        raise ValueError("parse_datetime expects a string")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"Invalid ISO-8601 datetime: {value}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed
