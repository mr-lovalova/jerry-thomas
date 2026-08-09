from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, tzinfo

import pytest

from jerrythomas.domain.record import TemporalRecord


@dataclass(eq=False, slots=True)
class _SlottedRecord(TemporalRecord):
    value: float


def test_temporal_record_rejects_timezone_without_utc_offset() -> None:
    class MissingOffsetTimezone(tzinfo):
        def utcoffset(self, dt: datetime | None) -> timedelta | None:
            return None

    with pytest.raises(ValueError, match="time must be timezone-aware"):
        TemporalRecord(datetime(2024, 1, 1, tzinfo=MissingOffsetTimezone()))


def test_temporal_record_equality_includes_slotted_fields() -> None:
    time = datetime(2024, 1, 1, tzinfo=timezone.utc)

    assert _SlottedRecord(time, 1.0) != _SlottedRecord(time, 2.0)
