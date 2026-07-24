import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from datapipeline.artifacts.ticks import (
    TickGrid,
    read_tick_grid,
    tick_grid_by_from_metadata,
)
from datapipeline.config.transforms import EnsureCadenceConfig
from datapipeline.transforms.stream.ensure_ticks import (
    EnsureCadenceTransform,
    EnsureTicksTransform,
)
from datapipeline.transforms.utils import record_establishes_domain
from tests.unit.transforms.helpers import make_time_record


def _time(hour: int) -> datetime:
    return datetime(2024, 1, 1, hour=hour, tzinfo=timezone.utc)


def _record(
    value: float | None,
    hour: int,
    security_id: str | None = None,
):
    record = make_time_record(value, hour)
    if security_id is not None:
        record.security_id = security_id
    return record


def _global_ticks(*hours: int) -> TickGrid:
    return TickGrid(grid_by=(), ticks={(): [_time(hour) for hour in hours]})


def test_ensure_cadence_inserts_fixed_duration_ticks() -> None:
    records = list(
        EnsureCadenceTransform(
            cadence="1h",
            partition_fields=(),
        ).apply(iter([_record(1.0, 0), _record(2.0, 2)]))
    )

    assert [(record.time.hour, record.value) for record in records] == [
        (0, 1.0),
        (1, None),
        (2, 2.0),
    ]
    assert [record_establishes_domain(record) for record in records] == [
        True,
        False,
        True,
    ]


@pytest.mark.parametrize("cadence", ["0m", "-1h"])
def test_ensure_cadence_rejects_nonpositive_duration(cadence: str) -> None:
    with pytest.raises(ValidationError, match="cadence must be positive"):
        EnsureCadenceConfig(cadence=cadence)


def test_ensure_ticks_fills_leading_internal_and_trailing_ticks() -> None:
    records = list(
        EnsureTicksTransform(
            ticks=_global_ticks(0, 1, 2, 3),
            partition_fields=(),
        ).apply(iter([_record(1.0, 0), _record(2.0, 2)]))
    )

    assert [(record.time.hour, record.value) for record in records] == [
        (0, 1.0),
        (1, None),
        (2, 2.0),
        (3, None),
    ]
    assert [record_establishes_domain(record) for record in records] == [
        True,
        False,
        True,
        False,
    ]


def test_ensure_ticks_uses_each_partition_grid() -> None:
    ticks = TickGrid(
        grid_by=("security_id",),
        ticks={
            ("AAPL",): [_time(0), _time(1), _time(2)],
            ("MSFT",): [_time(0), _time(1), _time(2)],
        },
    )
    source = [_record(10.0, 1, "AAPL"), _record(20.0, 1, "MSFT")]

    records = EnsureTicksTransform(
        ticks=ticks,
        partition_fields=("security_id",),
    ).apply(iter(source))

    assert [
        (record.security_id, record.time.hour, record.value) for record in records
    ] == [
        ("AAPL", 0, None),
        ("AAPL", 1, 10.0),
        ("AAPL", 2, None),
        ("MSFT", 0, None),
        ("MSFT", 1, 20.0),
        ("MSFT", 2, None),
    ]


def test_ensure_ticks_requires_grid_fields_to_match_partition_fields() -> None:
    ticks = TickGrid(grid_by=(), ticks={(): [_time(0)]})

    with pytest.raises(ValueError, match="must match stream partition_by"):
        EnsureTicksTransform(
            ticks=ticks,
            partition_fields=("security_id",),
        )


def test_ensure_ticks_keeps_records_outside_grid() -> None:
    records = EnsureTicksTransform(
        ticks=_global_ticks(0, 1),
        partition_fields=(),
    ).apply(iter([_record(3.0, 3)]))

    assert [(record.time.hour, record.value) for record in records] == [
        (0, None),
        (1, None),
        (3, 3.0),
    ]


def test_ensure_ticks_does_not_create_records_without_a_source_record() -> None:
    records = EnsureTicksTransform(
        ticks=_global_ticks(0, 1),
        partition_fields=(),
    ).apply(iter([]))

    assert list(records) == []


def test_ensure_ticks_placeholders_clear_unrelated_payload() -> None:
    record = _record(3.0, 1, "AAPL")
    record.volume = 100
    ticks = TickGrid(
        grid_by=("security_id",),
        ticks={("AAPL",): [_time(0), _time(1)]},
    )

    records = list(
        EnsureTicksTransform(
            ticks=ticks,
            partition_fields=("security_id",),
        ).apply(iter([record]))
    )

    assert records[0].security_id == "AAPL"
    assert records[0].value is None
    assert records[0].volume is None


def test_ensure_ticks_does_not_slice_the_grid() -> None:
    class NoSliceList(list[datetime]):
        def __getitem__(self, index):
            if isinstance(index, slice):
                raise AssertionError("tick grid must not be sliced")
            return super().__getitem__(index)

    ticks = TickGrid(
        grid_by=(),
        ticks={(): NoSliceList([_time(0), _time(1), _time(2)])},
    )

    records = EnsureTicksTransform(
        ticks=ticks,
        partition_fields=(),
    ).apply(iter([_record(1.0, 0), _record(2.0, 2)]))

    assert [(record.time.hour, record.value) for record in records] == [
        (0, 1.0),
        (1, None),
        (2, 2.0),
    ]


def test_read_tick_grid_keeps_canonical_rows(tmp_path) -> None:
    path = tmp_path / "ticks.jsonl"
    rows = [
        {"time": _time(0).isoformat()},
        {"time": _time(1).isoformat()},
        {"time": _time(2).isoformat()},
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    ticks = read_tick_grid(path, ())

    assert ticks.ticks_for(()) == [_time(0), _time(1), _time(2)]


@pytest.mark.parametrize(
    "hours",
    [
        (0, 0),
        (1, 0),
    ],
)
def test_read_tick_grid_rejects_noncanonical_order(tmp_path, hours) -> None:
    path = tmp_path / "ticks.jsonl"
    path.write_text(
        "".join(json.dumps({"time": _time(hour).isoformat()}) + "\n" for hour in hours),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="strictly ordered"):
        read_tick_grid(path, ())


def test_read_tick_grid_rejects_partition_that_reappears(tmp_path) -> None:
    path = tmp_path / "ticks.jsonl"
    rows = [
        {"time": _time(0).isoformat(), "security_id": "AAPL"},
        {"time": _time(0).isoformat(), "security_id": "MSFT"},
        {"time": _time(1).isoformat(), "security_id": "AAPL"},
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="strictly ordered"):
        read_tick_grid(path, ("security_id",))


def test_read_tick_grid_rejects_unexpected_fields(tmp_path) -> None:
    path = tmp_path / "ticks.jsonl"
    path.write_text(
        json.dumps({"time": _time(0).isoformat(), "security_id": "AAPL"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="row grid fields"):
        read_tick_grid(path, ())


def test_read_tick_grid_requires_time(tmp_path) -> None:
    path = tmp_path / "ticks.jsonl"
    path.write_text(json.dumps({"security_id": "AAPL"}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="without time"):
        read_tick_grid(path, ("security_id",))


def test_tick_grid_metadata_requires_a_string_list() -> None:
    with pytest.raises(RuntimeError, match="must be a list of strings"):
        tick_grid_by_from_metadata("model_grid", {"grid_by": "security_id"})


def test_tick_grid_metadata_is_required() -> None:
    with pytest.raises(RuntimeError, match="metadata field 'grid_by' is required"):
        tick_grid_by_from_metadata("model_grid", {})
