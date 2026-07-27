from dataclasses import dataclass
from datetime import UTC, datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from datapipeline.config.streams import AlignedStreamConfig
from datapipeline.domain.record import TemporalRecord
from datapipeline.services.streams.combine import build_combine_stage
from datapipeline.transforms.utils import (
    record_establishes_domain,
    set_record_domain_anchor,
)


@dataclass
class _Record(TemporalRecord):
    id_: str
    value: float


def _record(day: int, value: float) -> _Record:
    return _Record(
        time=datetime(2025, 1, day, tzinfo=UTC),
        id_="A",
        value=value,
    )


def _config() -> AlignedStreamConfig:
    return AlignedStreamConfig.model_validate(
        {
            "id": "aligned.out",
            "from": {"align": ["stream.a", "stream.b"]},
            "combine": {"entrypoint": "calculate", "args": {"offset": 3}},
        }
    )


def test_combiner_calls_function_with_positional_records(monkeypatch) -> None:
    def calculate(left, right, offset):
        return _record(1, left.value + right.value + offset)

    monkeypatch.setattr(
        "datapipeline.services.streams.combine.load_entrypoint",
        lambda _group, _entrypoint: calculate,
    )
    combine = build_combine_stage(_config(), ("id_",))

    records = list(combine(iter([(_record(1, 1), _record(1, 10))])))

    assert [record.value for record in records] == [14]


def test_combiner_drops_none(monkeypatch) -> None:
    monkeypatch.setattr(
        "datapipeline.services.streams.combine.load_entrypoint",
        lambda _group, _entrypoint: lambda _left, _right, offset: None,
    )
    combine = build_combine_stage(_config(), ("id_",))

    assert list(combine(iter([(_record(1, 1), _record(1, 10))]))) == []


def test_aligned_combiner_uses_any_real_input_as_domain_anchor(monkeypatch) -> None:
    def calculate(left, right, offset):
        return _record(1, left.value + right.value + offset)

    monkeypatch.setattr(
        "datapipeline.services.streams.combine.load_entrypoint",
        lambda _group, _entrypoint: calculate,
    )
    combine = build_combine_stage(_config(), ("id_",))
    primary = _record(1, 1)
    set_record_domain_anchor(primary, False)

    [record] = combine(iter([(primary, _record(1, 10))]))

    assert record_establishes_domain(record)


def test_aligned_combiner_preserves_placeholder_only_rows(monkeypatch) -> None:
    def calculate(left, right, offset):
        return _record(1, left.value + right.value + offset)

    monkeypatch.setattr(
        "datapipeline.services.streams.combine.load_entrypoint",
        lambda _group, _entrypoint: calculate,
    )
    combine = build_combine_stage(_config(), ("id_",))
    left = _record(1, 1)
    right = _record(1, 10)
    set_record_domain_anchor(left, False)
    set_record_domain_anchor(right, False)

    [record] = combine(iter([(left, right)]))

    assert not record_establishes_domain(record)


def test_combiner_rejects_changed_time(monkeypatch) -> None:
    def change_time(left, _right, offset):
        left.time = datetime(2025, 1, 2, tzinfo=UTC)
        return left

    monkeypatch.setattr(
        "datapipeline.services.streams.combine.load_entrypoint",
        lambda _group, _entrypoint: change_time,
    )
    combine = build_combine_stage(_config(), ("id_",))

    with pytest.raises(ValueError, match="must preserve input time"):
        list(combine(iter([(_record(1, 1), _record(1, 10))])))


def test_combiner_normalizes_equivalent_fall_back_time_to_input(
    monkeypatch,
) -> None:
    timezone_ny = ZoneInfo("America/New_York")
    expected_time = datetime(2024, 11, 3, 5, 30, tzinfo=UTC)
    local_time = datetime(
        2024,
        11,
        3,
        1,
        30,
        tzinfo=timezone_ny,
        fold=0,
    )
    left = _Record(time=expected_time, id_="A", value=1)
    right = _Record(time=expected_time, id_="A", value=10)

    def calculate(left, right, offset):
        return SimpleNamespace(
            time=local_time,
            id_=left.id_,
            value=left.value + right.value + offset,
        )

    monkeypatch.setattr(
        "datapipeline.services.streams.combine.load_entrypoint",
        lambda _group, _entrypoint: calculate,
    )
    combine = build_combine_stage(_config(), ("id_",))

    [record] = combine(iter([(left, right)]))

    assert record.time is left.time
    assert record.time.tzinfo is UTC


@pytest.mark.parametrize(
    ("invalid_time", "error_type", "message"),
    [
        (
            "2025-01-01T00:00:00Z",
            TypeError,
            "combine output time must be a datetime; got str",
        ),
        (
            datetime(2025, 1, 1),
            ValueError,
            "combine output time must be timezone-aware",
        ),
    ],
)
def test_combiner_rejects_invalid_time(
    monkeypatch,
    invalid_time,
    error_type,
    message,
) -> None:
    def calculate(left, right, offset):
        return SimpleNamespace(
            time=invalid_time,
            id_=left.id_,
            value=left.value + right.value + offset,
        )

    monkeypatch.setattr(
        "datapipeline.services.streams.combine.load_entrypoint",
        lambda _group, _entrypoint: calculate,
    )
    combine = build_combine_stage(_config(), ("id_",))

    with pytest.raises(error_type, match=message):
        list(combine(iter([(_record(1, 1), _record(1, 10))])))


def test_combiner_rejects_changed_partition(monkeypatch) -> None:
    def change_partition(left, _right, offset):
        left.id_ = "B"
        return left

    monkeypatch.setattr(
        "datapipeline.services.streams.combine.load_entrypoint",
        lambda _group, _entrypoint: change_partition,
    )
    combine = build_combine_stage(_config(), ("id_",))

    with pytest.raises(ValueError, match="must preserve partition field 'id_'"):
        list(combine(iter([(_record(1, 1), _record(1, 10))])))
