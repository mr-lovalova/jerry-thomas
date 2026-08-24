import json
from datetime import datetime, timezone

from jerrythomas.artifacts.models import VectorMetadata
from jerrythomas.artifacts.specs import VECTOR_METADATA
from jerrythomas.config.dataset.dataset import DatasetConfig, SampleConfig
from jerrythomas.config.dataset.series import SeriesConfig, TargetSeriesConfig
from jerrythomas.config.tasks.coverage_stats import CoverageStatsTask
from jerrythomas.domain.sample import Sample
from jerrythomas.domain.vector import Vector
from jerrythomas.operations.artifacts.coverage_stats import (
    build_coverage_stats_artifact,
)
from jerrythomas.runtime import Runtime


def _ts(day: int) -> datetime:
    return datetime(2024, 1, day, tzinfo=timezone.utc)


def _metadata() -> VectorMetadata:
    return VectorMetadata.model_validate(
        {
            "schema_version": 4,
            "catalog": {
                "features": [
                    {
                        "id": "speed",
                        "base_id": "speed",
                        "kind": "list",
                        "present_count": 1,
                        "null_count": 0,
                        "element_types": ["float", "null"],
                        "length": 2,
                        "observed_elements": 1,
                    }
                ],
                "targets": [
                    {
                        "id": "return",
                        "base_id": "return",
                        "kind": "scalar",
                        "present_count": 1,
                        "null_count": 0,
                        "value_types": ["float"],
                    }
                ],
                "counts": {"feature_vectors": 1, "target_vectors": 1},
                "window": {
                    "start": _ts(1),
                    "end": _ts(2),
                    "mode": "intersection",
                    "size": 2,
                },
            },
            "layout": {"kind": "unsplit"},
        }
    )


def _runtime(tmp_path) -> Runtime:
    project_yaml = tmp_path / "project.yaml"
    project_yaml.write_text(
        "schema_version: 6\nartifact_revision: 1\n", encoding="utf-8"
    )
    return Runtime(
        project_yaml=project_yaml,
        artifacts_root=tmp_path / "artifacts",
        dataset=DatasetConfig(
            sample=SampleConfig(cadence="1h"),
            features=[SeriesConfig(id="speed", stream="stream", field="value")],
            targets=[
                TargetSeriesConfig(
                    id="return",
                    stream="stream",
                    field="value",
                    horizon="0s",
                )
            ],
        ),
    )


def _register_metadata(monkeypatch, runtime: Runtime) -> None:
    def load_artifact(spec):
        assert spec.key == VECTOR_METADATA
        return _metadata()

    monkeypatch.setattr(runtime.artifacts, "load", load_artifact)


def test_build_coverage_stats_artifact_writes_bounded_v3_summary(
    monkeypatch,
    tmp_path,
) -> None:
    runtime = _runtime(tmp_path)
    samples = [
        Sample(
            key=(_ts(1),),
            features=Vector(values={"speed": [1.0, None]}),
            targets=Vector(values={"return": None}),
        ),
        Sample(key=(_ts(2),), features=Vector(values={}), targets=Vector(values={})),
    ]
    _register_metadata(monkeypatch, runtime)
    monkeypatch.setattr(
        "jerrythomas.operations.artifacts.coverage_stats.run_dataset_pipeline",
        lambda *_args: iter(samples),
    )

    task = CoverageStatsTask()
    build_coverage_stats_artifact(runtime, task)

    payload = json.loads(
        (runtime.artifacts_root / task.output).read_text(encoding="utf-8")
    )
    assert payload["schema_version"] == 3
    assert payload["stage"] == "postprocessed"
    assert payload["total_samples"] == 2
    assert payload["empty_samples"] == 1
    assert payload["features"]["columns"] == [
        {
            "id": "speed",
            "present_samples": 1,
            "non_null_samples": 1,
            "base_id": "speed",
            "kind": "list",
            "length": 2,
            "observed_elements": 1,
        }
    ]
    assert payload["targets"]["columns"][0]["non_null_samples"] == 0
    assert "group_feature_status" not in payload


def test_assembled_coverage_stats_do_not_apply_postprocess(
    monkeypatch,
    tmp_path,
) -> None:
    runtime = _runtime(tmp_path)
    _register_metadata(monkeypatch, runtime)
    monkeypatch.setattr(
        "jerrythomas.operations.artifacts.coverage_stats.run_sample_pipeline",
        lambda *_args, **_kwargs: iter(()),
    )

    def fail_postprocessed_pipeline(*_args):
        raise AssertionError("assembled coverage stats must not run postprocessing")

    monkeypatch.setattr(
        "jerrythomas.operations.artifacts.coverage_stats.run_dataset_pipeline",
        fail_postprocessed_pipeline,
    )

    build_coverage_stats_artifact(
        runtime,
        CoverageStatsTask(stage="assembled"),
    )


def test_postprocessed_coverage_stats_keep_columns_when_every_sample_is_dropped(
    monkeypatch,
    tmp_path,
) -> None:
    runtime = _runtime(tmp_path)
    _register_metadata(monkeypatch, runtime)
    monkeypatch.setattr(
        "jerrythomas.operations.artifacts.coverage_stats.run_dataset_pipeline",
        lambda *_args: iter(()),
    )

    task = CoverageStatsTask()
    build_coverage_stats_artifact(runtime, task)

    payload = json.loads(
        (runtime.artifacts_root / task.output).read_text(encoding="utf-8")
    )
    assert payload["total_samples"] == 0
    assert [entry["id"] for entry in payload["features"]["columns"]] == ["speed"]
    assert [entry["id"] for entry in payload["targets"]["columns"]] == ["return"]
