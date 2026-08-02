from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from jerrythomas.domain.record import TemporalRecord
from jerrythomas.parsers.temporal_record import TemporalRecordParser


@dataclass
class _WeatherRecord(TemporalRecord):
    symbol: str


def test_temporal_record_parser_rehydrates_mapping_rows() -> None:
    parser = TemporalRecordParser()

    record = parser.parse(
        {
            "time": "2024-01-01T12:00:00Z",
            "symbol": "AAPL",
            "value": 123.4,
        }
    )

    assert isinstance(record, TemporalRecord)
    assert record.time == datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    assert record.symbol == "AAPL"
    assert record.value == 123.4


def test_temporal_record_parser_supports_custom_time_field() -> None:
    parser = TemporalRecordParser(time_field="timestamp")

    record = parser.parse({"timestamp": "2024-01-01T12:00:00+01:00", "value": 1})

    assert record.time == datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc)
    assert record.value == 1


def test_temporal_record_parser_does_not_overwrite_time_with_residual_field() -> None:
    parser = TemporalRecordParser(time_field="timestamp")

    record = parser.parse(
        {
            "timestamp": "2024-01-01T12:00:00+01:00",
            "time": "raw-label",
            "value": 1,
        }
    )

    assert record.time == datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc)
    assert record.value == 1


def test_temporal_record_parser_passes_through_existing_temporal_records() -> None:
    parser = TemporalRecordParser()
    raw = _WeatherRecord(
        time=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
        symbol="MSFT",
    )

    parsed = parser.parse(raw)

    assert parsed is raw


def test_temporal_record_parser_assumes_utc_for_naive_datetime_values() -> None:
    parser = TemporalRecordParser()

    record = parser.parse({"time": datetime(2024, 1, 1, 12, 0), "value": 2})

    assert record.time == datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)


def test_temporal_record_parser_rehydrates_object_attributes() -> None:
    parser = TemporalRecordParser()

    class _Row:
        def __init__(self) -> None:
            self.time = "2024-01-01T12:00:00Z"
            self.symbol = "NVDA"
            self._private = "ignored"

    record = parser.parse(_Row())

    assert record.time == datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    assert record.symbol == "NVDA"
    assert not hasattr(record, "_private")


def test_temporal_record_parser_rejects_missing_time_field() -> None:
    parser = TemporalRecordParser()

    with pytest.raises(ValueError, match="expected field 'time'"):
        parser.parse({"value": 1})
