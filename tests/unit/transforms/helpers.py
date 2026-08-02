from datetime import datetime, timezone
from typing import Any

from jerrythomas.domain.series import SeriesRecord
from jerrythomas.domain.record import TemporalRecord
from jerrythomas.domain.sample import Sample
from jerrythomas.domain.vector import Vector


def make_time_record(value: float | None, hour: int) -> TemporalRecord:
    record = TemporalRecord(
        time=datetime(2024, 1, 1, hour=hour, tzinfo=timezone.utc),
    )
    setattr(record, "value", value)
    return record


def make_feature_record(
    value: float | None, hour: int, feature_id: str
) -> SeriesRecord:
    return SeriesRecord(
        id=feature_id,
        time=datetime(2024, 1, 1, hour=hour, tzinfo=timezone.utc),
        value=value,
        _establishes_domain=True,
    )


def make_vector(group: int, values: dict[str, Any]) -> Sample:
    return Sample(key=(group,), features=Vector(values=values))
