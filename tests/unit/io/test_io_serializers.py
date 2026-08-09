import json
from dataclasses import dataclass
from datetime import date, datetime, timezone

import pytest

from jerrythomas.domain.sample import Sample
from jerrythomas.domain.series import SeriesRecord
from jerrythomas.domain.record import TemporalRecord
from jerrythomas.domain.vector import Vector
from jerrythomas.io.serializers import (
    json_line_serializer,
    text_line_serializer,
)


@dataclass(slots=True)
class _SlottedTemporalRecord(TemporalRecord):
    security_id: str
    close: float


def test_record_json_serializer_emits_plain_payload() -> None:
    @dataclass
    class Record:
        time: datetime
        value: float

    rec = Record(datetime(2024, 1, 1, tzinfo=timezone.utc), 42.5)
    serializer = json_line_serializer()

    payload = json.loads(serializer(rec))

    assert payload == {
        "time": "2024-01-01 00:00:00+00:00",
        "value": 42.5,
    }


def test_series_json_serializer_emits_only_projected_fields() -> None:
    time = datetime(2024, 7, 4, tzinfo=timezone.utc)
    record = SeriesRecord(
        "feature_a",
        time,
        7.0,
        ("AAPL",),
        _establishes_domain=True,
    )

    payload = json.loads(json_line_serializer()(record))

    assert payload == {
        "id": "feature_a",
        "time": "2024-07-04 00:00:00+00:00",
        "value": 7.0,
        "entity_key": ["AAPL"],
    }


def test_temporal_record_serializer_includes_mapped_fields() -> None:
    record = TemporalRecord(datetime(2024, 7, 4, tzinfo=timezone.utc))
    record.security_id = "AAPL"
    record.close = 42.5

    payload = json.loads(json_line_serializer()(record))

    assert payload == {
        "time": "2024-07-04 00:00:00+00:00",
        "security_id": "AAPL",
        "close": 42.5,
    }


def test_temporal_record_serializer_includes_slotted_and_dynamic_fields() -> None:
    record = _SlottedTemporalRecord(
        datetime(2024, 7, 4, tzinfo=timezone.utc),
        security_id="AAPL",
        close=42.5,
    )
    record.exchange = "XNAS"

    payload = json.loads(json_line_serializer()(record))

    assert payload == {
        "time": "2024-07-04 00:00:00+00:00",
        "security_id": "AAPL",
        "close": 42.5,
        "exchange": "XNAS",
    }


def test_json_serializer_flat_view_emits_flattened_payload() -> None:
    serializer = json_line_serializer(view="flat")

    payload = json.loads(serializer({"features": {"x": 1.0}}))

    assert payload == {"features.x": 1.0}


def test_json_serializer_flat_view_flattens_sample_key() -> None:
    serializer = json_line_serializer(view="flat")
    sample = Sample(
        key=("2024-01-01", "AAPL"),
        features=Vector(values={"x": 1.0}),
    )

    payload = json.loads(serializer(sample))

    assert payload == {
        "key.0": "2024-01-01",
        "key.1": "AAPL",
        "features.x": 1.0,
    }


def test_flat_json_serializer_rejects_colliding_sample_feature_ids() -> None:
    serializer = json_line_serializer(view="flat")
    sample = Sample(
        key=("sample",),
        features=Vector(values={"x": [10, 11], "x.0": 99}),
    )

    with pytest.raises(ValueError, match=r"features\.x\.0"):
        serializer(sample)


def test_json_serializer_raw_sample_emits_nested_payload() -> None:
    serializer = json_line_serializer(view="raw")

    payload = json.loads(
        serializer(
            Sample(
                key=("k1",),
                features=Vector(values={"x": 1.0}),
                targets=Vector(values={"y": 2.0}),
            )
        )
    )

    assert payload == {
        "key": ["k1"],
        "features": {"values": {"x": 1.0}},
        "targets": {"values": {"y": 2.0}},
    }


def test_text_serializer_writes_text_payload() -> None:
    serializer = text_line_serializer()
    assert serializer({"text": "hello world"}) == "hello world\n"


def test_json_serializer_supports_dataclasses_sequences_and_dates() -> None:
    @dataclass
    class Payload:
        day: date
        values: tuple[int, ...]

    payload = json.loads(json_line_serializer()(Payload(date(2024, 7, 4), (1, 2))))

    assert payload == {"day": "2024-07-04", "values": [1, 2]}


def test_serializers_reject_arbitrary_objects() -> None:
    class Unsupported:
        def __init__(self) -> None:
            self.value = 1

    with pytest.raises(TypeError, match="Unsupported output value type: Unsupported"):
        json_line_serializer()(Unsupported())


def test_serializers_normalize_nan_as_missing() -> None:
    value = float("nan")

    assert json.loads(json_line_serializer()({"value": value})) == {"value": None}
    assert json.loads(json_line_serializer(view="flat")({"value": value})) == {
        "value": None
    }
    assert text_line_serializer()({"value": value}) == '{"value": null}\n'


@pytest.mark.parametrize("value", [float("inf"), float("-inf")])
@pytest.mark.parametrize(
    "serializer",
    [
        json_line_serializer(),
        json_line_serializer(view="flat"),
        text_line_serializer(),
    ],
)
def test_serializers_reject_infinity(value: float, serializer) -> None:
    with pytest.raises(ValueError, match="must not contain infinity"):
        serializer({"value": value})


def test_json_serializer_rejects_non_string_mapping_keys() -> None:
    with pytest.raises(
        TypeError,
        match="Unsupported output mapping key type: int",
    ):
        json_line_serializer()({1: "value"})
