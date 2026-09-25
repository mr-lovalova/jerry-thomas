import json
import logging
from datetime import datetime, timezone
from types import SimpleNamespace

from jerrythomas.artifacts.models import CoverageStatsArtifact, VectorMetadata
from jerrythomas.artifacts.specs import COVERAGE_STATS, VECTOR_METADATA
from jerrythomas.config.dataset.dataset import DatasetConfig, SampleConfig
from jerrythomas.config.dataset.series import SeriesConfig
from jerrythomas.config.tasks.coverage import CoverageTask
from jerrythomas.config.tasks.matrix import MatrixTask
from jerrythomas.domain.sample import Sample
from jerrythomas.domain.vector import Vector
from jerrythomas.io.output import OutputTarget
from jerrythomas.operations.persistence import persist_runtime_result
from jerrythomas.operations.runtime import coverage as coverage_ops
from jerrythomas.operations.runtime import matrix as matrix_ops
from jerrythomas.operations.runtime.execution import OutputOptions


def _coverage_stats() -> CoverageStatsArtifact:
    return CoverageStatsArtifact.model_validate(
        {
            "schema_version": 3,
            "stage": "postprocessed",
            "total_samples": 2,
            "empty_samples": 0,
            "features": {
                "bases": [
                    {
                        "id": "speed",
                        "present_samples": 2,
                        "non_null_samples": 1,
                    }
                ],
                "columns": [
                    {
                        "id": "speed",
                        "base_id": "speed",
                        "kind": "scalar",
                        "present_samples": 2,
                        "non_null_samples": 1,
                    }
                ],
            },
            "targets": {"bases": [], "columns": []},
        }
    )


def _metadata() -> VectorMetadata:
    return VectorMetadata.model_validate(
        {
            "schema_version": 4,
            "catalog": {
                "features": [
                    {
                        "id": "speed",
                        "base_id": "speed",
                        "kind": "scalar",
                        "present_count": 1,
                        "null_count": 0,
                        "value_types": ["float"],
                    }
                ],
                "targets": [],
                "counts": {"feature_vectors": 1, "target_vectors": 0},
                "window": {
                    "start": datetime(2024, 1, 1, tzinfo=timezone.utc),
                    "end": datetime(2024, 1, 1, tzinfo=timezone.utc),
                    "mode": "union",
                    "size": 1,
                },
            },
            "layout": {"kind": "unsplit"},
        }
    )


def _load_coverage_stats(spec):
    assert spec.key == COVERAGE_STATS
    return _coverage_stats()


def _load_metadata(spec):
    assert spec.key == VECTOR_METADATA
    return _metadata()


def _matrix_runtime():
    return SimpleNamespace(
        require_dataset=lambda: DatasetConfig(
            sample=SampleConfig(rounding="ceil", cadence="1h"),
            features=[SeriesConfig(id="speed", stream="stream", field="value")],
        ),
        artifacts=SimpleNamespace(load=_load_metadata),
    )


def _patch_matrix(monkeypatch) -> None:
    samples = [Sample(key="g0", features=Vector(values={"speed": 1.0}))]
    monkeypatch.setattr(
        matrix_ops,
        "run_sample_pipeline",
        lambda *_args: iter(samples),
    )
    monkeypatch.setattr(
        matrix_ops,
        "run_dataset_pipeline",
        lambda *_args: iter(samples),
    )


def _persist_result(result, target: OutputTarget) -> None:
    persist_runtime_result(
        result,
        target=target,
        logger=logging.getLogger(__name__),
    )


def test_inspect_coverage_reads_typed_coverage_stats_artifact(
    monkeypatch,
    tmp_path,
) -> None:
    destination = (tmp_path / "coverage.txt").resolve()

    result = coverage_ops.run_coverage_operation(
        runtime=SimpleNamespace(artifacts=SimpleNamespace(load=_load_coverage_stats)),
        task=CoverageTask(id="coverage", options={"threshold": 0.8}),
        options=OutputOptions(),
    )
    _persist_result(
        result,
        OutputTarget(
            transport="fs",
            format="txt",
            view="flat",
            encoding="utf-8",
            destination=destination,
        ),
    )

    report = destination.read_text(encoding="utf-8")
    assert '"report": "coverage"' in report
    assert '"threshold": 0.8' in report
    assert '"below_threshold_columns": [' in report
    assert '"speed"' in report


def test_inspect_coverage_writes_one_json_report(monkeypatch, tmp_path) -> None:
    destination = (tmp_path / "coverage.jsonl").resolve()

    result = coverage_ops.run_coverage_operation(
        runtime=SimpleNamespace(artifacts=SimpleNamespace(load=_load_coverage_stats)),
        task=CoverageTask(id="coverage"),
        options=OutputOptions(),
    )
    _persist_result(
        result,
        OutputTarget(
            transport="fs",
            format="jsonl",
            view="raw",
            encoding="utf-8",
            destination=destination,
        ),
    )

    rows = destination.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 1
    payload = json.loads(rows[0])
    assert payload["stage"] == "postprocessed"
    assert payload["features"]["columns"][0]["coverage"] == 0.5


def test_inspect_matrix_writes_jsonl(monkeypatch, tmp_path) -> None:
    _patch_matrix(monkeypatch)
    destination = (tmp_path / "matrix.jsonl").resolve()

    result = matrix_ops.run_matrix_operation(
        runtime=_matrix_runtime(),
        task=MatrixTask(id="matrix", options={"max_cells": 10}),
        options=OutputOptions(),
    )
    _persist_result(
        result,
        OutputTarget(
            transport="fs",
            format="jsonl",
            view="raw",
            encoding="utf-8",
            destination=destination,
        ),
    )

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload == {
        "vector": "feature",
        "identifier": "speed",
        "group": "g0",
        "status": "present",
    }


def test_inspect_matrix_writes_html(monkeypatch, tmp_path) -> None:
    _patch_matrix(monkeypatch)
    destination = (tmp_path / "matrix.html").resolve()
    result = matrix_ops.run_matrix_operation(
        runtime=_matrix_runtime(),
        task=MatrixTask(id="matrix"),
        options=OutputOptions(),
    )

    _persist_result(
        result,
        OutputTarget(
            transport="fs",
            format="html",
            view="flat",
            encoding=None,
            destination=destination,
        ),
    )

    assert "Availability Matrix" in destination.read_text(encoding="utf-8")


def test_assembled_matrix_does_not_postprocess(monkeypatch) -> None:
    _patch_matrix(monkeypatch)

    def fail_postprocessed_pipeline(*_args):
        raise AssertionError("assembled matrix must not run postprocessing")

    monkeypatch.setattr(
        matrix_ops,
        "run_dataset_pipeline",
        fail_postprocessed_pipeline,
    )
    matrix_ops.run_matrix_operation(
        runtime=_matrix_runtime(),
        task=MatrixTask(id="matrix", options={"stage": "assembled"}),
        options=OutputOptions(),
    )


def test_matrix_limit_caps_samples_after_postprocess(monkeypatch) -> None:
    _patch_matrix(monkeypatch)
    samples = [
        Sample(key=f"g{index}", features=Vector(values={"speed": float(index)}))
        for index in range(3)
    ]
    monkeypatch.setattr(
        matrix_ops,
        "run_dataset_pipeline",
        lambda *_args: iter(samples[1:]),
    )

    result = matrix_ops.run_matrix_operation(
        runtime=_matrix_runtime(),
        task=MatrixTask(id="matrix"),
        options=OutputOptions(limit=1),
    )

    assert result.rows is not None
    assert [row["group"] for row in result.rows] == ["g1"]


def test_postprocessed_matrix_keeps_headers_when_every_sample_is_dropped(
    monkeypatch,
    tmp_path,
) -> None:
    _patch_matrix(monkeypatch)
    monkeypatch.setattr(
        matrix_ops,
        "run_dataset_pipeline",
        lambda *_args: iter(()),
    )
    destination = (tmp_path / "empty-matrix.html").resolve()

    result = matrix_ops.run_matrix_operation(
        runtime=_matrix_runtime(),
        task=MatrixTask(id="matrix"),
        options=OutputOptions(),
    )
    _persist_result(
        result,
        OutputTarget(
            transport="fs",
            format="html",
            view="flat",
            encoding=None,
            destination=destination,
        ),
    )

    document = destination.read_text(encoding="utf-8")
    assert "<th scope='col'>speed</th>" in document
    assert '"rows": []' in document
