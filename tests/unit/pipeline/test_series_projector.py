import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pytest

from datapipeline.config.dataset.series import SeriesConfig
from datapipeline.domain.series_id import base_id
from datapipeline.domain.sample_key import SampleKeyContract
from datapipeline.pipelines.series import projector as projector_module
from datapipeline.pipelines.series.projector import SeriesProjector


@dataclass
class _Record:
    station_id: object = None
    sensor: object = None
    time: datetime = datetime(2024, 1, 1, tzinfo=timezone.utc)
    _establishes_domain: bool = field(default=True, init=False, repr=False)


def _projected_id(
    record: _Record,
    partition_by: tuple[str, ...],
    sample_keys: tuple[str, ...] = (),
) -> str:
    projector = SeriesProjector(partition_by, SampleKeyContract(sample_keys))
    config = SeriesConfig(stream="stream", id="temp", field="sensor")
    return next(projector.project(record, (config,))).id


def test_series_projector_without_id_components() -> None:
    assert _projected_id(_Record(), ()) == "temp"


def test_series_projector_projects_long_identity_as_entity_key() -> None:
    projector = SeriesProjector(
        ("station_id",),
        SampleKeyContract(("station_id",)),
    )
    configs = (
        SeriesConfig(stream="stream", id="station", field="station_id"),
        SeriesConfig(stream="stream", id="sensor", field="sensor"),
    )

    records = tuple(projector.project(_Record("north", 7), configs))

    assert [(record.id, record.value) for record in records] == [
        ("station", "north"),
        ("sensor", 7),
    ]
    assert [record.entity_key for record in records] == [("north",), ("north",)]


def test_series_projector_normalizes_nested_nan_without_mutating_record() -> None:
    source_record = _Record(sensor={"values": [1.0, float("nan")]})
    projector = SeriesProjector((), SampleKeyContract(()))
    config = SeriesConfig(stream="stream", id="sensor", field="sensor")

    [projected] = projector.project(source_record, (config,))

    assert projected.value == {"values": [1.0, None]}
    assert isinstance(source_record.sensor, dict)
    assert math.isnan(source_record.sensor["values"][1])


@pytest.mark.parametrize("value", [float("inf"), float("-inf")])
def test_series_projector_rejects_nested_infinity(value: float) -> None:
    record = _Record(sensor={"values": [1.0, value]})
    projector = SeriesProjector((), SampleKeyContract(()))
    config = SeriesConfig(stream="stream", id="sensor", field="sensor")

    with pytest.raises(ValueError, match="must not contain infinity"):
        next(projector.project(record, (config,)))


def test_series_projector_derives_wide_series_identity() -> None:
    identifier = _projected_id(
        _Record(station_id="north"),
        ("station_id",),
    )

    assert identifier == "temp__@station_id:north"


def test_series_projector_derives_hybrid_series_identity() -> None:
    identifier = _projected_id(
        _Record(station_id="north", sensor="temperature"),
        ("station_id", "sensor"),
        ("station_id",),
    )

    assert identifier == "temp__@sensor:temperature"


def test_series_projector_computes_shared_values_once_per_record(monkeypatch) -> None:
    encoded_fields: list[str] = []
    domain_records: list[_Record] = []
    encode = projector_module.encode_series_id_component
    establishes_domain = projector_module.record_establishes_domain

    def count_encode(field: str, value: object) -> str:
        encoded_fields.append(field)
        return encode(field, value)

    def count_domain_record(record: _Record) -> bool:
        domain_records.append(record)
        return establishes_domain(record)

    monkeypatch.setattr(projector_module, "encode_series_id_component", count_encode)
    monkeypatch.setattr(
        projector_module,
        "record_establishes_domain",
        count_domain_record,
    )
    projector = SeriesProjector(
        ("station_id", "sensor"),
        SampleKeyContract(()),
    )
    configs = (
        SeriesConfig(stream="stream", id="temperature", field="sensor"),
        SeriesConfig(stream="stream", id="humidity", field="sensor"),
    )
    source_record = _Record("north", 7)

    records = tuple(projector.project(source_record, configs))

    assert len(records) == 2
    assert encoded_fields == ["station_id", "sensor"]
    assert domain_records == [source_record]


def test_series_projector_tags_non_string_scalar_types() -> None:
    assert _projected_id(_Record(station_id=123), ("station_id",)) == (
        "temp__@station_id:!i:123"
    )
    assert _projected_id(_Record(station_id=True), ("station_id",)) == (
        "temp__@station_id:!b:1"
    )
    assert _projected_id(_Record(station_id=1.0), ("station_id",)) == (
        "temp__@station_id:!f:0x1.0000000000000p+0"
    )


def test_series_projector_distinguishes_null_empty_and_scalar_types() -> None:
    identifiers = {
        _projected_id(_Record(station_id=None), ("station_id",)),
        _projected_id(_Record(station_id=""), ("station_id",)),
        _projected_id(_Record(station_id="1"), ("station_id",)),
        _projected_id(_Record(station_id=1), ("station_id",)),
        _projected_id(_Record(station_id=True), ("station_id",)),
        _projected_id(_Record(station_id=1.0), ("station_id",)),
    }

    assert len(identifiers) == 6


def test_series_projector_escapes_component_delimiters() -> None:
    identifier = _projected_id(
        _Record(station_id="north__west|@sensor:x", sensor="A:B/100%"),
        ("station_id", "sensor"),
    )

    assert identifier == (
        "temp__@station_id:north__west%7C%40sensor%3Ax|@sensor:A%3AB%2F100%25"
    )
    assert base_id(identifier) == "temp"


def test_distinct_component_tuples_cannot_generate_the_same_id() -> None:
    first = _projected_id(
        _Record(station_id="north|@sensor:south", sensor="x"),
        ("station_id", "sensor"),
    )
    second = _projected_id(
        _Record(station_id="north", sensor="south|@sensor:x"),
        ("station_id", "sensor"),
    )

    assert first != second


def test_series_projector_rejects_unsupported_component_types() -> None:
    with pytest.raises(TypeError, match="string, integer, float, boolean, or null"):
        _projected_id(_Record(station_id=object()), ("station_id",))


def test_series_projector_rejects_non_finite_components() -> None:
    with pytest.raises(ValueError, match="finite float"):
        _projected_id(_Record(station_id=float("nan")), ("station_id",))
