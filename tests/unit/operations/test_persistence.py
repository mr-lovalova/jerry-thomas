import gzip
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pytest
import pyarrow.parquet as parquet

import jerrythomas.operations.persistence as persistence
from jerrythomas.artifacts.models import ScalarVectorMetadataEntry
from jerrythomas.cli.visuals.execution import make_execution_observer
from jerrythomas.cli.visuals.execution_context import (
    reset_current_execution_event_handler,
    set_current_execution_event_handler,
)
from jerrythomas.execution.observability import (
    FileResult,
    execution_observer,
    operation_scope,
)
from jerrythomas.io.output import OutputTarget
from jerrythomas.io.dataset_table import DatasetTable
from jerrythomas.domain.sample import Sample
from jerrythomas.domain.vector import Vector
from jerrythomas.operations.persistence import (
    DatasetTableOutput,
    RoutedDatasetTableOutput,
    RoutedRuntimeOutput,
    RuntimeOutput,
    RuntimeOutputBatch,
    persist_runtime_result,
)


class _CaptureHandler:
    def __init__(self) -> None:
        self.events = []

    def __call__(self, event) -> None:
        self.events.append(event)


class _ClosableRows:
    def __init__(
        self,
        rows=(),
        *,
        iteration_error: Exception | None = None,
        close_error: Exception | None = None,
    ) -> None:
        self._rows = iter(rows)
        self._iteration_error = iteration_error
        self._close_error = close_error
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._iteration_error is not None:
            raise self._iteration_error
        return next(self._rows)

    def close(self) -> None:
        self.closed = True
        if self._close_error is not None:
            raise self._close_error


class _IterFailureRows:
    def __init__(self) -> None:
        self.closed = False

    def __iter__(self):
        raise RuntimeError("iterator failed")

    def close(self) -> None:
        self.closed = True


def _dataset_table() -> DatasetTable:
    return DatasetTable(
        sample_keys=("ticker",),
        sample_key_types=("string",),
        feature_entries=(
            ScalarVectorMetadataEntry(
                id="price",
                base_id="price",
                kind="scalar",
                present_count=2,
                null_count=0,
                value_types=("float",),
            ),
        ),
        target_entries=(),
    )


def _dataset_sample(day: int, ticker: str, price: object) -> Sample:
    return Sample(
        key=(datetime(2024, 1, day, tzinfo=timezone.utc), ticker),
        features=Vector(values={"price": price}),
    )


def test_runtime_output_requires_a_representation() -> None:
    with pytest.raises(ValueError, match="requires rows, payload, or render_html"):
        RuntimeOutput()


def test_runtime_output_rejects_rows_and_payload() -> None:
    with pytest.raises(ValueError, match="cannot define both rows and payload"):
        RuntimeOutput(rows=(), payload={})


def test_runtime_output_allows_rows_and_html_representations() -> None:
    output = RuntimeOutput(rows=(), render_html=lambda: "<html></html>")

    assert output.rows == ()
    assert output.render_html is not None


def test_html_only_runtime_output_rejects_non_html_target(tmp_path: Path) -> None:
    destination = tmp_path / "output.jsonl"

    with pytest.raises(ValueError, match="does not support a non-HTML representation"):
        persist_runtime_result(
            RuntimeOutput(render_html=lambda: "<html></html>"),
            target=OutputTarget(
                transport="fs",
                format="jsonl",
                view="raw",
                encoding="utf-8",
                destination=destination,
            ),
            logger=logging.getLogger(__name__),
        )

    assert not destination.exists()


def test_failed_runtime_write_preserves_existing_file(tmp_path) -> None:
    destination = tmp_path / "out.jsonl"
    destination.write_text("previous\n", encoding="utf-8")

    def rows():
        yield {"value": 1}
        raise RuntimeError("boom")

    target = OutputTarget(
        transport="fs",
        format="jsonl",
        view="raw",
        encoding="utf-8",
        destination=destination,
    )

    with pytest.raises(RuntimeError, match="boom"):
        persist_runtime_result(
            RuntimeOutput(rows=rows()),
            target=target,
            logger=logging.getLogger(__name__),
        )

    assert destination.read_text(encoding="utf-8") == "previous\n"


def test_failed_runtime_write_closes_rows(tmp_path) -> None:
    closed = False

    def rows():
        nonlocal closed
        try:
            yield {"value": object()}
        finally:
            closed = True

    stream = rows()
    target = OutputTarget(
        transport="fs",
        format="jsonl",
        view="raw",
        encoding="utf-8",
        destination=tmp_path / "out.jsonl",
    )

    with pytest.raises(TypeError, match="Unsupported output value type"):
        persist_runtime_result(
            RuntimeOutput(rows=stream),
            target=target,
            logger=logging.getLogger(__name__),
        )

    assert closed


def test_runtime_writer_open_failure_closes_rows(monkeypatch, tmp_path) -> None:
    rows = _ClosableRows()
    target = OutputTarget(
        transport="fs",
        format="jsonl",
        view="raw",
        encoding="utf-8",
        destination=tmp_path / "out.jsonl",
    )

    def fail_writer_open(_target):
        raise OSError("open failed")

    monkeypatch.setattr(persistence, "writer_factory", fail_writer_open)

    with pytest.raises(OSError, match="open failed"):
        persist_runtime_result(
            RuntimeOutput(rows=rows),
            target=target,
            logger=logging.getLogger(__name__),
        )

    assert rows.closed


def test_runtime_row_cleanup_failure_aborts_output(tmp_path) -> None:
    destination = tmp_path / "out.jsonl"
    destination.write_text("previous\n", encoding="utf-8")
    rows = _ClosableRows(
        ({"value": 1},),
        close_error=OSError("cleanup failed"),
    )
    target = OutputTarget(
        transport="fs",
        format="jsonl",
        view="raw",
        encoding="utf-8",
        destination=destination,
    )

    with pytest.raises(OSError, match="cleanup failed"):
        persist_runtime_result(
            RuntimeOutput(rows=rows),
            target=target,
            logger=logging.getLogger(__name__),
        )

    assert rows.closed
    assert destination.read_text(encoding="utf-8") == "previous\n"


def test_runtime_row_cleanup_does_not_replace_processing_failure(tmp_path) -> None:
    rows = _ClosableRows(
        iteration_error=RuntimeError("processing failed"),
        close_error=OSError("cleanup failed"),
    )
    target = OutputTarget(
        transport="fs",
        format="jsonl",
        view="raw",
        encoding="utf-8",
        destination=tmp_path / "out.jsonl",
    )

    with pytest.raises(RuntimeError, match="processing failed"):
        persist_runtime_result(
            RuntimeOutput(rows=rows),
            target=target,
            logger=logging.getLogger(__name__),
        )

    assert rows.closed


def test_runtime_iterator_failure_closes_rows(tmp_path) -> None:
    rows = _IterFailureRows()
    target = OutputTarget(
        transport="fs",
        format="jsonl",
        view="raw",
        encoding="utf-8",
        destination=tmp_path / "out.jsonl",
    )

    with pytest.raises(RuntimeError, match="iterator failed"):
        persist_runtime_result(
            RuntimeOutput(rows=rows),
            target=target,
            logger=logging.getLogger(__name__),
        )

    assert rows.closed


def test_runtime_batch_failure_closes_pending_rows(tmp_path) -> None:
    pending_rows = _ClosableRows()
    target = OutputTarget(
        transport="fs",
        format="jsonl",
        view="raw",
        encoding="utf-8",
        destination=tmp_path / "out.jsonl",
    )
    result = RuntimeOutputBatch(
        outputs=(
            RuntimeOutput(
                rows=_ClosableRows(iteration_error=RuntimeError("processing failed")),
                target=target,
            ),
            RuntimeOutput(rows=pending_rows, target=target),
        )
    )

    with pytest.raises(RuntimeError, match="processing failed"):
        persist_runtime_result(
            result,
            target=None,
            logger=logging.getLogger(__name__),
        )

    assert pending_rows.closed


def test_routed_runtime_output_rejects_colliding_destinations(tmp_path) -> None:
    targets = {
        output_id: OutputTarget(
            transport="fs",
            format="jsonl",
            view="raw",
            encoding="utf-8",
            destination=tmp_path / filename,
        )
        for output_id, filename in (
            ("train", "SPLIT.jsonl"),
            ("test", "split.jsonl"),
        )
    }

    with pytest.raises(ValueError, match="resolve to the same destination"):
        persist_runtime_result(
            RoutedRuntimeOutput(
                rows=iter(()),
                targets=targets,
            ),
            target=None,
            logger=logging.getLogger(__name__),
        )

    assert list(tmp_path.iterdir()) == []


def test_invalid_routed_runtime_output_closes_rows() -> None:
    rows = _ClosableRows()

    with pytest.raises(ValueError, match="at least one target"):
        persist_runtime_result(
            RoutedRuntimeOutput(
                rows=rows,
                targets={},
            ),
            target=None,
            logger=logging.getLogger(__name__),
        )

    assert rows.closed


def test_runtime_persistence_emits_flat_output(tmp_path) -> None:
    destination = tmp_path / "out.jsonl"
    target = OutputTarget(
        transport="fs",
        format="jsonl",
        view="raw",
        encoding="utf-8",
        destination=destination,
    )
    capture = _CaptureHandler()
    token = set_current_execution_event_handler(capture)
    try:
        logger = logging.getLogger(__name__)
        observer = make_execution_observer(logger)
        with execution_observer(observer):
            with operation_scope("serve:train"):
                persist_runtime_result(
                    RuntimeOutput(rows=iter([{"value": 1}])),
                    target=target,
                    heartbeat_interval_seconds=0,
                    logger=logger,
                )
    finally:
        reset_current_execution_event_handler(token)

    outputs = [event for event in capture.events if isinstance(event, FileResult)]
    assert len(outputs) == 1
    assert outputs[0].label == "Output"
    assert outputs[0].path == destination


def test_runtime_payload_emits_output_path(monkeypatch, tmp_path) -> None:
    destination = tmp_path / "coverage.json"
    results: list[tuple[str, Path]] = []
    monkeypatch.setattr(
        persistence,
        "emit_file_result",
        lambda label, path: results.append((label, path)),
    )

    persist_runtime_result(
        RuntimeOutput(payload={"covered": True}),
        target=OutputTarget(
            transport="fs",
            format="jsonl",
            view="raw",
            encoding="utf-8",
            destination=destination,
        ),
        logger=logging.getLogger(__name__),
    )

    assert results == [("Output", destination)]


def test_runtime_persistence_reports_stdout(capsys, caplog) -> None:
    logger = logging.getLogger("jerrythomas.tests.persistence.stdout")
    with caplog.at_level(logging.INFO, logger=logger.name):
        persist_runtime_result(
            RuntimeOutput(rows=({"value": 1},)),
            target=OutputTarget(
                transport="stdout",
                format="jsonl",
                view="raw",
                encoding=None,
                destination=None,
            ),
            logger=logger,
        )

    assert capsys.readouterr().out == '{"value": 1}\n'
    assert [record.getMessage() for record in caplog.records] == ["Output: stdout"]


def test_runtime_persistence_reports_html_path(monkeypatch, tmp_path) -> None:
    destination = tmp_path / "matrix.html"
    results: list[tuple[str, Path]] = []
    monkeypatch.setattr(
        persistence,
        "emit_file_result",
        lambda label, path: results.append((label, path)),
    )

    persist_runtime_result(
        RuntimeOutput(render_html=lambda: "<html></html>"),
        target=OutputTarget(
            transport="fs",
            format="html",
            view="flat",
            encoding="utf-8",
            destination=destination,
        ),
        logger=logging.getLogger(__name__),
    )

    assert destination.read_text(encoding="utf-8") == "<html></html>"
    assert results == [("Output", destination)]


def test_html_output_cleanup_failure_prevents_commit(tmp_path) -> None:
    destination = tmp_path / "matrix.html"
    destination.write_text("previous", encoding="utf-8")
    rows = _ClosableRows(close_error=OSError("cleanup failed"))

    with pytest.raises(OSError, match="cleanup failed"):
        persist_runtime_result(
            RuntimeOutput(
                rows=rows,
                render_html=lambda: "replacement",
            ),
            target=OutputTarget(
                transport="fs",
                format="html",
                view="flat",
                encoding="utf-8",
                destination=destination,
            ),
            logger=logging.getLogger(__name__),
        )

    assert rows.closed
    assert destination.read_text(encoding="utf-8") == "previous"


def test_failed_html_commit_preserves_existing_file(monkeypatch, tmp_path) -> None:
    destination = tmp_path / "matrix.html"
    destination.write_text("previous", encoding="utf-8")

    def fail_commit(*_args):
        raise OSError("commit failed")

    monkeypatch.setattr(
        "jerrythomas.io.sinks.files._commit_temp_file",
        fail_commit,
    )

    with pytest.raises(OSError, match="commit failed"):
        persist_runtime_result(
            RuntimeOutput(render_html=lambda: "replacement"),
            target=OutputTarget(
                transport="fs",
                format="html",
                view="flat",
                encoding="utf-8",
                destination=destination,
            ),
            logger=logging.getLogger(__name__),
        )

    assert destination.read_text(encoding="utf-8") == "previous"
    assert list(tmp_path.iterdir()) == [destination]


def test_routed_runtime_output_routes_rows_to_output_targets(
    monkeypatch,
    tmp_path,
) -> None:
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    train_target = OutputTarget(
        transport="fs",
        format="jsonl",
        view="raw",
        encoding="utf-8",
        destination=train_path,
    )
    val_target = OutputTarget(
        transport="fs",
        format="jsonl",
        view="raw",
        encoding="utf-8",
        destination=val_path,
    )
    empty_path = tmp_path / "empty.jsonl"
    empty_target = OutputTarget(
        transport="fs",
        format="jsonl",
        view="raw",
        encoding="utf-8",
        destination=empty_path,
    )
    rows = [
        {"output": "train", "value": 1},
        {"output": "val", "value": 2},
        {"output": None, "value": 3},
    ]
    results: list[tuple[str, str, int | Path]] = []
    monkeypatch.setattr(
        persistence,
        "emit_rows_written",
        lambda output_id, row_count: results.append(("rows", output_id, row_count)),
    )
    monkeypatch.setattr(
        persistence,
        "emit_file_result",
        lambda label, path: results.append(("file", label, path)),
    )

    persist_runtime_result(
        RoutedRuntimeOutput(
            rows=iter((("train", rows[0]), ("val", rows[1]))),
            targets={
                "train": train_target,
                "val": val_target,
                "empty": empty_target,
            },
        ),
        target=None,
        logger=logging.getLogger(__name__),
    )

    train_rows = [
        json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()
    ]
    val_rows = [
        json.loads(line) for line in val_path.read_text(encoding="utf-8").splitlines()
    ]
    assert train_rows == [{"output": "train", "value": 1}]
    assert val_rows == [{"output": "val", "value": 2}]
    assert empty_path.read_text(encoding="utf-8") == ""
    assert results == [
        ("rows", "train", 1),
        ("file", "train", train_path),
        ("rows", "val", 1),
        ("file", "val", val_path),
        ("rows", "empty", 0),
        ("file", "empty", empty_path),
    ]


def test_routed_runtime_output_rejects_unknown_output_id(tmp_path) -> None:
    targets = {
        output_id: OutputTarget(
            transport="fs",
            format="jsonl",
            view="raw",
            encoding="utf-8",
            destination=tmp_path / f"{output_id}.jsonl",
        )
        for output_id in ("train", "val")
    }

    with pytest.raises(
        ValueError,
        match="unknown output ID 'test'",
    ):
        persist_runtime_result(
            RoutedRuntimeOutput(
                rows=iter(
                    (
                        ("train", {"value": 1}),
                        ("test", {"value": 2}),
                    )
                ),
                targets=targets,
            ),
            target=None,
            logger=logging.getLogger(__name__),
        )

    assert list(tmp_path.iterdir()) == []


def test_routed_rows_failure_preserves_every_destination(tmp_path) -> None:
    destinations = {
        output_id: tmp_path / f"{output_id}.jsonl"
        for output_id in ("fold_0.train", "fold_1.train")
    }
    for destination in destinations.values():
        destination.write_text("previous\n", encoding="utf-8")

    def rows():
        yield "fold_0.train", {"value": 1}
        yield "fold_1.train", {"value": 2}
        raise RuntimeError("assembly failed")

    with pytest.raises(RuntimeError, match="assembly failed"):
        persist_runtime_result(
            RoutedRuntimeOutput(
                rows=rows(),
                targets={
                    output_id: OutputTarget(
                        transport="fs",
                        format="jsonl",
                        view="raw",
                        encoding="utf-8",
                        destination=destination,
                    )
                    for output_id, destination in destinations.items()
                },
            ),
            target=None,
            logger=logging.getLogger(__name__),
        )

    assert all(
        destination.read_text(encoding="utf-8") == "previous\n"
        for destination in destinations.values()
    )
    assert set(tmp_path.iterdir()) == set(destinations.values())


def test_routed_runtime_output_writes_gzip_targets(tmp_path) -> None:
    targets = {
        output_id: OutputTarget(
            transport="fs",
            format="jsonl",
            view="raw",
            encoding="utf-8",
            destination=tmp_path / f"{output_id}.jsonl.gz",
            compression="gzip",
        )
        for output_id in ("train", "validation")
    }
    rows = [
        {"output": "train", "value": 1},
        {"output": "validation", "value": 2},
    ]

    persist_runtime_result(
        RoutedRuntimeOutput(
            rows=iter(
                (
                    ("train", rows[0]),
                    ("validation", rows[1]),
                )
            ),
            targets=targets,
        ),
        target=None,
        logger=logging.getLogger(__name__),
    )

    for output_id, expected in (("train", rows[:1]), ("validation", rows[1:])):
        destination = targets[output_id].destination
        assert destination is not None
        with gzip.open(destination, "rt", encoding="utf-8") as stream:
            assert [json.loads(line) for line in stream] == expected


def test_routed_runtime_output_limit_applies_per_output(tmp_path) -> None:
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    train_target = OutputTarget(
        transport="fs",
        format="jsonl",
        view="raw",
        encoding="utf-8",
        destination=train_path,
    )
    val_target = OutputTarget(
        transport="fs",
        format="jsonl",
        view="raw",
        encoding="utf-8",
        destination=val_path,
    )
    rows = [
        {"split": None, "value": 0},
        {"split": "train", "value": 1},
        {"split": "train", "value": 2},
        {"split": "val", "value": 3},
        {"split": "val", "value": 4},
    ]

    persist_runtime_result(
        RoutedRuntimeOutput(
            rows=iter((row["split"], row) for row in rows if row["split"] is not None),
            targets={"train": train_target, "val": val_target},
            limit_per_output=1,
        ),
        target=None,
        logger=logging.getLogger(__name__),
    )

    train_rows = [
        json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()
    ]
    val_rows = [
        json.loads(line) for line in val_path.read_text(encoding="utf-8").splitlines()
    ]
    assert train_rows == [{"split": "train", "value": 1}]
    assert val_rows == [{"split": "val", "value": 3}]


def test_dataset_table_output_persists_parquet_rows(tmp_path) -> None:
    destination = tmp_path / "samples.parquet"
    target = OutputTarget(
        transport="fs",
        format="parquet",
        view="flat",
        encoding=None,
        destination=destination,
    )

    persist_runtime_result(
        DatasetTableOutput(
            rows=iter(
                (
                    _dataset_sample(1, "AAPL", 10.0),
                    _dataset_sample(2, "MSFT", 20.0),
                )
            ),
            table=_dataset_table(),
        ),
        target=target,
        logger=logging.getLogger(__name__),
    )

    result = parquet.read_table(destination)
    assert result.column_names == [
        "sample.time",
        "sample.ticker",
        "features.price",
    ]
    assert result.column("features.price").to_pylist() == [10.0, 20.0]


def test_routed_dataset_table_output_persists_each_parquet_output(tmp_path) -> None:
    targets = {
        output_id: OutputTarget(
            transport="fs",
            format="parquet",
            view="flat",
            encoding=None,
            destination=tmp_path / f"{output_id}.parquet",
        )
        for output_id in ("train", "validation")
    }
    tables = {
        "train": _dataset_table(),
        "validation": DatasetTable(
            sample_keys=("ticker",),
            sample_key_types=("string",),
            feature_entries=(
                ScalarVectorMetadataEntry(
                    id="score",
                    base_id="score",
                    kind="scalar",
                    present_count=1,
                    null_count=0,
                    value_types=("float",),
                ),
            ),
            target_entries=(),
        ),
    }
    samples = (
        _dataset_sample(1, "AAPL", 10.0),
        Sample(
            key=(datetime(2024, 1, 2, tzinfo=timezone.utc), "MSFT"),
            features=Vector(values={"score": 20.0}),
        ),
    )

    persist_runtime_result(
        RoutedDatasetTableOutput(
            rows=iter(
                (
                    ("train", samples[0]),
                    ("validation", samples[1]),
                )
            ),
            tables=tables,
            targets=targets,
        ),
        target=None,
        logger=logging.getLogger(__name__),
    )

    assert parquet.read_table(targets["train"].destination).column(
        "sample.ticker"
    ).to_pylist() == ["AAPL"]
    assert parquet.read_table(targets["validation"].destination).column(
        "sample.ticker"
    ).to_pylist() == ["MSFT"]
    assert parquet.read_table(targets["train"].destination).column_names == [
        "sample.time",
        "sample.ticker",
        "features.price",
    ]
    assert parquet.read_table(targets["validation"].destination).column_names == [
        "sample.time",
        "sample.ticker",
        "features.score",
    ]


def test_routed_dataset_table_output_requires_a_table_for_every_target(
    tmp_path,
) -> None:
    targets = {
        output_id: OutputTarget(
            transport="fs",
            format="parquet",
            view="flat",
            encoding=None,
            destination=tmp_path / f"{output_id}.parquet",
        )
        for output_id in ("train", "validation")
    }

    with pytest.raises(
        ValueError,
        match=r"missing tables for output IDs: \['validation'\]",
    ):
        RoutedDatasetTableOutput(
            rows=iter(()),
            tables={"train": _dataset_table()},
            targets=targets,
        )


def test_routed_dataset_table_output_rejects_a_table_without_a_target(
    tmp_path,
) -> None:
    target = OutputTarget(
        transport="fs",
        format="parquet",
        view="flat",
        encoding=None,
        destination=tmp_path / "train.parquet",
    )

    with pytest.raises(
        ValueError,
        match=r"tables without targets for output IDs: \['validation'\]",
    ):
        RoutedDatasetTableOutput(
            rows=iter(()),
            tables={
                "train": _dataset_table(),
                "validation": _dataset_table(),
            },
            targets={"train": target},
        )


def test_dataset_table_projection_failure_preserves_destination(tmp_path) -> None:
    destination = tmp_path / "samples.parquet"
    destination.write_bytes(b"previous")

    with pytest.raises(TypeError, match="features.price.*requires float"):
        persist_runtime_result(
            DatasetTableOutput(
                rows=iter((_dataset_sample(1, "AAPL", "wrong"),)),
                table=_dataset_table(),
            ),
            target=OutputTarget(
                transport="fs",
                format="parquet",
                view="flat",
                encoding=None,
                destination=destination,
            ),
            logger=logging.getLogger(__name__),
        )

    assert destination.read_bytes() == b"previous"
    assert list(tmp_path.iterdir()) == [destination]
