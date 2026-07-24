from dataclasses import dataclass
from datetime import UTC, datetime

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
        "datapipeline.services.streams.combine.load_ep",
        lambda _group, _entrypoint: calculate,
    )
    combine = build_combine_stage(_config(), ("id_",))

    records = list(combine(iter([(_record(1, 1), _record(1, 10))])))

    assert [record.value for record in records] == [14]


def test_combiner_drops_none(monkeypatch) -> None:
    monkeypatch.setattr(
        "datapipeline.services.streams.combine.load_ep",
        lambda _group, _entrypoint: lambda _left, _right, offset: None,
    )
    combine = build_combine_stage(_config(), ("id_",))

    assert list(combine(iter([(_record(1, 1), _record(1, 10))]))) == []


def test_aligned_combiner_uses_any_real_input_as_domain_anchor(monkeypatch) -> None:
    def calculate(left, right, offset):
        return _record(1, left.value + right.value + offset)

    monkeypatch.setattr(
        "datapipeline.services.streams.combine.load_ep",
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
        "datapipeline.services.streams.combine.load_ep",
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
        "datapipeline.services.streams.combine.load_ep",
        lambda _group, _entrypoint: change_time,
    )
    combine = build_combine_stage(_config(), ("id_",))

    with pytest.raises(ValueError, match="must preserve input time"):
        list(combine(iter([(_record(1, 1), _record(1, 10))])))


def test_combiner_rejects_changed_partition(monkeypatch) -> None:
    def change_partition(left, _right, offset):
        left.id_ = "B"
        return left

    monkeypatch.setattr(
        "datapipeline.services.streams.combine.load_ep",
        lambda _group, _entrypoint: change_partition,
    )
    combine = build_combine_stage(_config(), ("id_",))

    with pytest.raises(ValueError, match="must preserve partition field 'id_'"):
        list(combine(iter([(_record(1, 1), _record(1, 10))])))
