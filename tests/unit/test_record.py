from datetime import datetime, timedelta, tzinfo

import pytest

from jerrythomas.domain.record import TemporalRecord


def test_temporal_record_rejects_timezone_without_utc_offset() -> None:
    class MissingOffsetTimezone(tzinfo):
        def utcoffset(self, dt: datetime | None) -> timedelta | None:
            return None

    with pytest.raises(ValueError, match="time must be timezone-aware"):
        TemporalRecord(datetime(2024, 1, 1, tzinfo=MissingOffsetTimezone()))
