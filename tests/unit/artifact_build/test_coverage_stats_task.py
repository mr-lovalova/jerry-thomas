import json
from datetime import datetime, timezone

from datapipeline.artifacts.models import VectorMetadata
from datapipeline.artifacts.specs import VECTOR_METADATA
from datapipeline.config.dataset.dataset import DatasetConfig, SampleConfig
from datapipeline.config.dataset.series import SeriesConfig, TargetSeriesConfig
from datapipeline.config.tasks.coverage_stats import CoverageStatsTask
from datapipeline.domain.sample import Sample
from datapipeline.domain.vector import Vector
from datapipeline.execution.pipeline import Stage
from datapipeline.operations.artifacts.coverage_stats import (
    build_coverage_stats_artifact,
)
from datapipeline.pipelines.dataset.postprocess import PostprocessPlan
from datapipeline.runtime import Runtime


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
        "schema_version: 4\nartifact_revision: 1\n", encoding="utf-8"
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


def _postprocess_plan(*stages: Stage) -> PostprocessPlan:
    return PostprocessPlan(stages=stages)


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
        "datapipeline.operations.artifacts.coverage_stats.open_samples",
        lambda *_args, **_kwargs: iter(samples),
    )
    monkeypatch.setattr(
        "datapipeline.operations.artifacts.coverage_stats.build_postprocess_plan",
        lambda _config, _schema: _postprocess_plan(),
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
        "datapipeline.operations.artifacts.coverage_stats.open_samples",
        lambda *_args, **_kwargs: iter(()),
    )

    def fail_plan(*_args):
        raise AssertionError(
            "assembled coverage stats must not build a postprocess plan"
        )

    monkeypatch.setattr(
        "datapipeline.operations.artifacts.coverage_stats.build_postprocess_plan",
        fail_plan,
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
    sample = Sample(
        key=(_ts(1),),
        features=Vector(values={"speed": [None, None]}),
        targets=Vector(values={"return": None}),
    )
    drop_all = Stage(name="drop_all", apply=lambda _samples: iter(()))
    _register_metadata(monkeypatch, runtime)
    monkeypatch.setattr(
        "datapipeline.operations.artifacts.coverage_stats.open_samples",
        lambda *_args, **_kwargs: iter((sample,)),
    )
    monkeypatch.setattr(
        "datapipeline.operations.artifacts.coverage_stats.build_postprocess_plan",
        lambda _config, _schema: _postprocess_plan(drop_all),
    )

    task = CoverageStatsTask()
    build_coverage_stats_artifact(runtime, task)

    payload = json.loads(
        (runtime.artifacts_root / task.output).read_text(encoding="utf-8")
    )
    assert payload["total_samples"] == 0
    assert [entry["id"] for entry in payload["features"]["columns"]] == ["speed"]
    assert [entry["id"] for entry in payload["targets"]["columns"]] == ["return"]
