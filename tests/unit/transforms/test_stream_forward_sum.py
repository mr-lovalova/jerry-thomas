from collections.abc import Iterator, Sequence
from math import fsum
from pathlib import Path
from random import Random
from typing import Any

import pytest

from jerrythomas.config.dataset.dataset import DatasetConfig, SampleConfig
from jerrythomas.execution.pipeline import Input, Pipeline, Stage
from jerrythomas.execution.runner import run_pipeline
from jerrythomas.runtime import Runtime
from jerrythomas.transforms.stream.forward_sum import ForwardSumTransform
from tests.unit.transforms.helpers import make_time_record


def _forward_values(
    values: Sequence[Any],
    window: int,
) -> list[float | None]:
    records = (make_time_record(value, 0) for value in values)
    transform = ForwardSumTransform(
        field="value",
        window=window,
        partition_fields=(),
        to="forward",
    )
    return [record.forward for record in transform.apply(records)]


def test_forward_sum_excludes_current_record() -> None:
    assert _forward_values([1.0, 2.0, 3.0, 4.0], window=2) == [
        5.0,
        7.0,
        None,
        None,
    ]


def test_forward_sum_supports_one_record_window() -> None:
    assert _forward_values([1.0, 2.0, 3.0], window=1) == [2.0, 3.0, None]


@pytest.mark.parametrize(
    "values",
    [[], [1.0], [1.0, 2.0], [1.0, 2.0, 3.0]],
)
def test_forward_sum_requires_a_complete_future_window(values: list[float]) -> None:
    assert _forward_values(values, window=3) == [None] * len(values)


def test_forward_sum_resets_between_partitions() -> None:
    records = [
        make_time_record(1.0, 0),
        make_time_record(2.0, 1),
        make_time_record(3.0, 2),
        make_time_record(10.0, 0),
        make_time_record(20.0, 1),
        make_time_record(30.0, 2),
    ]
    for record, partition in zip(records, ["A", "A", "A", "B", "B", "B"]):
        record.partition = partition

    transform = ForwardSumTransform(
        field="value",
        window=2,
        partition_fields=("partition",),
        to="forward",
    )

    assert [record.forward for record in transform.apply(iter(records))] == [
        5.0,
        None,
        None,
        50.0,
        None,
        None,
    ]


def test_forward_sum_missing_values_only_invalidate_windows_containing_them() -> None:
    assert _forward_values([1.0, 2.0, None, 4.0, 5.0], window=2) == [
        None,
        None,
        9.0,
        None,
        None,
    ]


def test_forward_sum_treats_nan_as_missing() -> None:
    assert _forward_values([1.0, 2.0, float("nan"), 4.0], window=2) == [
        None,
        None,
        None,
        None,
    ]


@pytest.mark.parametrize("value", [True, "1.0", object()])
def test_forward_sum_rejects_non_numeric_values(value: object) -> None:
    with pytest.raises(TypeError, match="numeric values"):
        _forward_values([1.0, 2.0, value], window=2)


@pytest.mark.parametrize("value", [float("inf"), float("-inf")])
def test_forward_sum_rejects_infinite_values(value: float) -> None:
    with pytest.raises(ValueError, match="finite numeric values"):
        _forward_values([1.0, 2.0, value], window=2)


def test_forward_sum_validates_values_that_cannot_complete_a_window() -> None:
    with pytest.raises(TypeError, match="numeric values"):
        _forward_values([1.0, "invalid"], window=2)


def test_forward_sum_requires_configured_field() -> None:
    transform = ForwardSumTransform(
        field="missing",
        window=1,
        partition_fields=(),
        to="forward",
    )

    with pytest.raises(KeyError, match="missing"):
        list(transform.apply(iter([make_time_record(1.0, 0)])))


def test_forward_sum_reports_floating_point_overflow() -> None:
    with pytest.raises(OverflowError, match="supported floating-point range"):
        _forward_values([0.0, 1e308, 1e308], window=2)


def test_forward_sum_preserves_cancellation() -> None:
    assert _forward_values(
        [0.0, 1e16, 1.0, -1e16],
        window=3,
    ) == [1.0, None, None, None]


def test_forward_sum_matches_fsum_reference() -> None:
    random = Random(731)
    values = [random.uniform(-1e6, 1e6) for _ in range(200)]
    window = 17
    expected = [
        fsum(values[index + 1 : index + window + 1])
        if index + window < len(values)
        else None
        for index in range(len(values))
    ]

    assert _forward_values(values, window) == pytest.approx(expected)


def test_forward_sum_clones_records_without_changing_order() -> None:
    records = [make_time_record(value, hour) for hour, value in enumerate([1.0, 2.0])]
    transform = ForwardSumTransform("value", 1, (), "forward")

    output = list(transform.apply(iter(records)))

    assert [record.time for record in output] == [record.time for record in records]
    assert output[0] is not records[0]
    assert not hasattr(records[0], "forward")


def test_pipeline_closes_source_when_forward_sum_output_is_closed(
    tmp_path: Path,
) -> None:
    closed = False

    def source() -> Iterator[Any]:
        nonlocal closed
        try:
            for hour in range(4):
                yield make_time_record(float(hour), hour)
        finally:
            closed = True

    project_yaml = tmp_path / "project.yaml"
    project_yaml.write_text(
        "schema_version: 6\nartifact_revision: 1\n",
        encoding="utf-8",
    )
    runtime = Runtime(
        project_yaml=project_yaml,
        artifacts_root=tmp_path / "artifacts",
        dataset=DatasetConfig(sample=SampleConfig(rounding="ceil", cadence="1h")),
    )
    transform = ForwardSumTransform("value", 2, (), "forward")
    pipeline = Pipeline(
        name="forward",
        input=Input("source", source),
        stages=(Stage("forward_sum", transform.apply),),
    )

    output = run_pipeline(runtime, pipeline)
    next(output)
    output.close()

    assert closed
