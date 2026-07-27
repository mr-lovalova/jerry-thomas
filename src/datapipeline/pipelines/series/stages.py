from collections import deque
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

from datapipeline.config.dataset.series import SequenceConfig
from datapipeline.domain.series import SeriesRecord, SeriesSequence
from datapipeline.pipelines.series.projector import SeriesProjector
from datapipeline.pipelines.sort import SortProgress, batch_sort
from datapipeline.transforms.utils import record_establishes_domain
from datapipeline.utils.time import floor_time_to_cadence, parse_cadence


def project_series(
    projector: SeriesProjector,
    records: Iterable[Any],
) -> Iterator[SeriesRecord]:
    for record in records:
        yield from projector.project(record)


def sequence_series(
    sequence: SequenceConfig,
    records: Iterator[SeriesRecord],
) -> Iterator[SeriesSequence]:
    sequencer = SeriesSequencer(sequence)
    for record in records:
        result = sequencer.append(record)
        if result is not None:
            yield result


class SeriesSequencer:
    def __init__(self, config: SequenceConfig) -> None:
        self.config = config
        self._active_key: tuple[str, tuple] | None = None
        self._window: deque[Any] = deque(maxlen=config.size)
        self._domain_anchors: deque[bool] = deque(maxlen=config.size)
        self._position = 0

    def append(self, record: SeriesRecord) -> SeriesSequence | None:
        if isinstance(record.value, list):
            raise TypeError(f"Series {record.id!r} sequence requires scalar values.")

        key = record.id, record.entity_key
        if key != self._active_key:
            self._active_key = key
            self._window.clear()
            self._domain_anchors.clear()
            self._position = 0

        position = self._position
        self._window.append(record.value)
        self._domain_anchors.append(record_establishes_domain(record))
        self._position += 1

        window_start = position - self.config.size + 1
        if len(self._window) != self.config.size:
            return None
        if window_start % self.config.stride != 0:
            return None
        return SeriesSequence(
            time=record.time,
            values=list(self._window),
            id=record.id,
            entity_key=record.entity_key,
            _establishes_domain=any(self._domain_anchors),
        )


def order_series(
    buffer_bytes: int,
    group_by_cadence: str,
    sample_keys: Sequence[str],
    progress: SortProgress,
    records: Iterator[SeriesRecord | SeriesSequence],
) -> Iterable[SeriesRecord | SeriesSequence]:
    key = _time_then_id
    if sample_keys:
        key = _sample_group_then_time_and_id(group_by_cadence)
    return batch_sort(
        records,
        buffer_bytes=buffer_bytes,
        key=key,
        progress=progress,
    )


def _time_then_id(
    item: SeriesRecord | SeriesSequence,
) -> tuple[Any, str]:
    return item.time, item.id


def _sample_group_then_time_and_id(group_by_cadence: str):
    cadence = parse_cadence(group_by_cadence)

    def key(item: SeriesRecord | SeriesSequence) -> tuple[Any, ...]:
        time_value = item.time
        return (
            floor_time_to_cadence(time_value, cadence),
            *item.entity_key,
            time_value,
            item.id,
        )

    return key
