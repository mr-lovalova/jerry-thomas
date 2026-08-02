import gzip
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from jerrythomas.config.dataset.dataset import DatasetConfig, SampleConfig
from jerrythomas.config.execution import ExecutionConfig
from jerrythomas.domain.record import TemporalRecord
from jerrythomas.io.output import OutputTarget
from jerrythomas.runtime import Runtime, SourceRuntimeStream
from jerrythomas.services.materialize import (
    materialize_stream,
    resolve_materialize_output,
)


class _Source:
    def __init__(self, rows):
        self._rows = rows

    def stream(self):
        return iter(self._rows)


def _ts(day: int) -> datetime:
    return datetime(2024, 1, day, tzinfo=timezone.utc)


def _mapper(rows):
    for row in rows:
        rec = TemporalRecord(time=row["time"])
        rec.security_id = row["security_id"]
        rec.close = row["close"]
        yield rec


def _runtime(
    tmp_path: Path,
    rows: list[dict],
) -> Runtime:
    runtime = Runtime(
        project_yaml=tmp_path / "project.yaml",
        artifacts_root=tmp_path / "artifacts",
        dataset=DatasetConfig(sample=SampleConfig(cadence="1h")),
        execution=ExecutionConfig(),
    )
    runtime.streams["prices.raw"] = SourceRuntimeStream(
        source=_Source(rows),
        mapper=_mapper,
        preprocess=(),
        transforms=(),
        partition_by=(),
        presorted=False,
    )
    return runtime


def _output(path: Path) -> OutputTarget:
    return resolve_materialize_output(path)


def test_materialize_output_requires_jsonl_suffix(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"\.jsonl or \.jsonl\.gz"):
        resolve_materialize_output(tmp_path / "prices.csv")


@pytest.fixture
def one_row_runtime(tmp_path: Path) -> Runtime:
    return _runtime(
        tmp_path,
        [{"time": _ts(1), "security_id": "AAPL", "close": 10.0}],
    )


def test_materialize_stream_writes_jsonl(tmp_path: Path) -> None:
    rows = [
        {"time": _ts(2), "security_id": "MSFT", "close": 20.0},
        {"time": _ts(1), "security_id": "AAPL", "close": 10.0},
    ]
    runtime = _runtime(tmp_path, rows)
    output = tmp_path / "interim" / "prices.materialized.jsonl"

    result = materialize_stream(
        runtime=runtime,
        stream_id="prices.raw",
        output=_output(output),
    )

    payloads = [
        json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert [(row["security_id"], row["close"]) for row in payloads] == [
        ("AAPL", 10.0),
        ("MSFT", 20.0),
    ]
    assert result == output.resolve()


def test_materialize_stream_writes_gzip_jsonl(tmp_path: Path) -> None:
    runtime = _runtime(
        tmp_path,
        [{"time": _ts(1), "security_id": "AAPL", "close": 10.0}],
    )
    output = tmp_path / "interim" / "prices.materialized.jsonl.gz"

    result = materialize_stream(
        runtime=runtime,
        stream_id="prices.raw",
        output=_output(output),
    )

    with gzip.open(output, "rt", encoding="utf-8") as stream:
        payloads = [json.loads(line) for line in stream]
    assert [(row["security_id"], row["close"]) for row in payloads] == [("AAPL", 10.0)]
    assert result == output.resolve()


def test_materialize_stream_normalizes_nan_as_null(tmp_path: Path) -> None:
    runtime = _runtime(
        tmp_path,
        [{"time": _ts(1), "security_id": "AAPL", "close": float("nan")}],
    )
    output = tmp_path / "prices.jsonl"

    materialize_stream(
        runtime=runtime,
        stream_id="prices.raw",
        output=_output(output),
    )

    assert json.loads(output.read_text(encoding="utf-8"))["close"] is None


@pytest.mark.parametrize("value", [float("inf"), float("-inf")])
def test_materialize_stream_rejects_infinity_atomically(
    tmp_path: Path,
    value: float,
) -> None:
    runtime = _runtime(
        tmp_path,
        [{"time": _ts(1), "security_id": "AAPL", "close": value}],
    )
    output = tmp_path / "prices.jsonl"
    output.write_text("previous\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must not contain infinity"):
        materialize_stream(
            runtime=runtime,
            stream_id="prices.raw",
            output=_output(output),
            overwrite=True,
        )

    assert output.read_text(encoding="utf-8") == "previous\n"


def test_materialize_stream_refuses_existing_output_without_overwrite(
    tmp_path: Path,
    one_row_runtime: Runtime,
) -> None:
    output = tmp_path / "prices.jsonl"
    output.write_text("", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--overwrite"):
        materialize_stream(
            runtime=one_row_runtime,
            stream_id="prices.raw",
            output=_output(output),
        )


def test_materialize_stream_refuses_managed_artifact_paths(
    tmp_path: Path,
    one_row_runtime: Runtime,
) -> None:
    output = tmp_path / "artifacts" / "schedule.jsonl"

    with pytest.raises(ValueError, match="outside the managed artifacts root"):
        materialize_stream(
            runtime=one_row_runtime,
            stream_id="prices.raw",
            output=_output(output),
            overwrite=True,
        )

    assert not output.exists()


@pytest.mark.parametrize("filename", ["prices.jsonl", "prices.jsonl.gz"])
def test_materialize_stream_does_not_clobber_output_created_during_stream(
    tmp_path: Path,
    monkeypatch,
    one_row_runtime: Runtime,
    filename: str,
) -> None:
    output = tmp_path / filename
    concurrent_output = b"concurrent output\n"

    def racing_rows(_context, _stream_id):
        def rows():
            output.write_bytes(concurrent_output)
            yield {"time": _ts(1), "security_id": "AAPL", "close": 10.0}

        return rows()

    monkeypatch.setattr(
        "jerrythomas.services.materialize.run_stream_pipeline",
        racing_rows,
    )

    with pytest.raises(FileExistsError, match="already exists"):
        materialize_stream(
            runtime=one_row_runtime,
            stream_id="prices.raw",
            output=_output(output),
        )

    assert output.read_bytes() == concurrent_output


def test_materialize_stream_closes_rows_after_writer_failure(
    tmp_path: Path,
    monkeypatch,
    one_row_runtime: Runtime,
) -> None:
    rows_closed = False

    def rows():
        nonlocal rows_closed
        try:
            yield object()
        finally:
            rows_closed = True

    monkeypatch.setattr(
        "jerrythomas.services.materialize.run_stream_pipeline",
        lambda _context, _stream_id: rows(),
    )

    output = tmp_path / "prices.jsonl"
    with pytest.raises(TypeError, match="Unsupported output value type"):
        materialize_stream(
            runtime=one_row_runtime,
            stream_id="prices.raw",
            output=_output(output),
        )

    assert rows_closed
    assert not output.exists()
