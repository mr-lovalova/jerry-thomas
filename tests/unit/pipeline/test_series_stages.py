import gc
import weakref
from datetime import datetime, timedelta, timezone

import pytest

from datapipeline.config.dataset.dataset import DatasetConfig, SampleConfig
from datapipeline.config.dataset.series import SequenceConfig, SeriesConfig
from datapipeline.domain.series import SeriesRecord
from datapipeline.domain.record import TemporalRecord
from datapipeline.domain.sample_key import SampleKeyContract
from datapipeline.execution.context import PipelineContext
from datapipeline.pipelines.series.stages import (
    SeriesSequencer,
    project_series,
    sequence_series,
)
from datapipeline.pipelines.series.pipeline import build_series_stages
from datapipeline.pipelines.series.projector import SeriesProjector
from datapipeline.runtime import Runtime, SourceRuntimeStream
from datapipeline.transforms.utils import (
    record_establishes_domain,
    set_record_domain_anchor,
)


class _EmptySource:
    def stream(self):
        return iter(())


def _identity(records):
    return records


def _context(
    tmp_path,
    partition_by: tuple[str, ...] = (),
) -> PipelineContext:
    runtime = Runtime(
        project_yaml=tmp_path / "project.yaml",
        artifacts_root=tmp_path / "artifacts",
        dataset=DatasetConfig(sample=SampleConfig(cadence="1h")),
    )
    runtime.streams["stream"] = SourceRuntimeStream(
        source=_EmptySource(),
        mapper=_identity,
        preprocess=(),
        transforms=(),
        partition_by=partition_by,
        presorted=False,
    )
    return PipelineContext(runtime)


def _series_record(
    value: object,
    position: int,
    series_id: str = "x",
    entity_key: tuple = (),
) -> SeriesRecord:
    return SeriesRecord(
        id=series_id,
        time=datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=position),
        value=value,
        entity_key=entity_key,
        _establishes_domain=True,
    )


def test_series_stages_sequence_without_scaling(tmp_path) -> None:
    stages = build_series_stages(
        _context(tmp_path),
        SeriesConfig(
            stream="stream",
            id="x",
            field="value",
            scale=True,
            sequence=SequenceConfig(size=3, stride=2),
        ),
        group_by_cadence="1h",
    )

    assert [stage.name for stage in stages] == [
        "project_series",
        "sequence_series",
    ]


def test_series_stages_omit_disabled_stages(tmp_path) -> None:
    stages = build_series_stages(
        _context(tmp_path),
        SeriesConfig(stream="stream", id="x", field="value"),
    )

    assert [stage.name for stage in stages] == ["project_series"]


def test_partitioned_series_stages_include_ordering(tmp_path) -> None:
    stages = build_series_stages(
        _context(tmp_path, partition_by=("symbol",)),
        SeriesConfig(stream="stream", id="x", field="value"),
        sample_keys=("symbol",),
        group_by_cadence="1h",
    )

    assert [stage.name for stage in stages] == [
        "project_series",
        "order_series",
    ]
    assert stages[-1].progress is not None


def test_sequence_series_builds_strided_windows() -> None:
    records = iter([_series_record(value, value) for value in range(1, 6)])

    sequences = list(sequence_series(SequenceConfig(size=3, stride=2), records))

    assert [sequence.values for sequence in sequences] == [
        [1, 2, 3],
        [3, 4, 5],
    ]


def test_sequence_series_tracks_each_series_and_entity_independently() -> None:
    records = iter(
        [
            _series_record(1, 0, entity_key=("A",)),
            _series_record(2, 1, entity_key=("A",)),
            _series_record(10, 2, entity_key=("B",)),
            _series_record(20, 3, entity_key=("B",)),
        ]
    )

    sequences = list(sequence_series(SequenceConfig(size=2), records))

    assert [(sequence.entity_key, sequence.values) for sequence in sequences] == [
        (("A",), [1, 2]),
        (("B",), [10, 20]),
    ]


def test_series_sequencer_retains_only_the_active_entity() -> None:
    sequencer = SeriesSequencer(SequenceConfig(size=2))

    for position in range(10_000):
        sequencer.append(
            _series_record(position, position, entity_key=(f"entity-{position}",))
        )

    assert len(sequencer._window) == 1


def test_sequence_series_preserves_none_values() -> None:
    records = iter([_series_record(1.0, 0), _series_record(None, 1)])

    [sequence] = sequence_series(SequenceConfig(size=2), records)

    assert sequence.values == [1.0, None]


def test_sequence_establishes_domain_when_any_input_record_does() -> None:
    placeholder = _series_record(None, 0)
    set_record_domain_anchor(placeholder, False)
    observed = _series_record(1.0, 1)

    [sequence] = sequence_series(
        SequenceConfig(size=2),
        iter([placeholder, observed]),
    )

    assert record_establishes_domain(sequence)


def test_sequence_of_placeholders_does_not_establish_domain() -> None:
    records = [_series_record(None, position) for position in range(2)]
    for record in records:
        set_record_domain_anchor(record, False)

    [sequence] = sequence_series(SequenceConfig(size=2), iter(records))

    assert not record_establishes_domain(sequence)


def test_projected_series_does_not_retain_source_record() -> None:
    source_record = TemporalRecord(time=datetime(2024, 1, 1, tzinfo=timezone.utc))
    source_record.value = 1.0
    record_ref = weakref.ref(source_record)
    [projected] = SeriesProjector((), SampleKeyContract(())).project(
        source_record,
        (SeriesConfig(stream="stream", id="x", field="value"),),
    )

    del source_record
    gc.collect()

    assert record_ref() is None
    assert projected.time == datetime(2024, 1, 1, tzinfo=timezone.utc)


def test_projected_series_preserves_domain_anchor() -> None:
    source_record = TemporalRecord(time=datetime(2024, 1, 1, tzinfo=timezone.utc))
    source_record.value = None
    set_record_domain_anchor(source_record, False)

    [projected] = SeriesProjector((), SampleKeyContract(())).project(
        source_record,
        (SeriesConfig(stream="stream", id="x", field="value"),),
    )

    assert not record_establishes_domain(projected)


def test_project_series_rejects_sample_key_type_drift() -> None:
    first = TemporalRecord(time=datetime(2024, 1, 1, tzinfo=timezone.utc))
    first.value = 1.0
    first.security_id = True
    second = TemporalRecord(time=datetime(2024, 1, 2, tzinfo=timezone.utc))
    second.value = 2.0
    second.security_id = 1

    with pytest.raises(TypeError, match="changed type"):
        list(
            project_series(
                SeriesProjector(
                    ("security_id",),
                    SampleKeyContract(["security_id"]),
                ),
                SeriesConfig(stream="stream", id="value", field="value"),
                iter([first, second]),
            )
        )
