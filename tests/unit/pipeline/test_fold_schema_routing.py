from datetime import datetime, timezone

import datapipeline.pipelines.dataset.pipeline as dataset_pipeline
from datapipeline.artifacts.models import (
    FoldOutputMetadata,
    ScalarVectorMetadataEntry,
    VectorMetadataCounts,
    VectorSchema,
    Window,
)
from datapipeline.config.dataset.dataset import DatasetConfig, SampleConfig
from datapipeline.config.dataset.series import SeriesConfig
from datapipeline.config.dataset.split import DatasetFold, TimeInterval, TimeSplitConfig
from datapipeline.domain.sample import Sample
from datapipeline.domain.vector import Vector
from datapipeline.execution.context import PipelineContext
from datapipeline.execution.pipeline import Input
from datapipeline.pipelines.dataset.pipeline import (
    FoldOutputPlan,
    run_fold_outputs_pipeline,
)
from datapipeline.runtime import Runtime


_TIME = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _schema(*feature_ids: str) -> VectorSchema:
    return VectorSchema(
        features=tuple(
            ScalarVectorMetadataEntry(
                id=feature_id,
                base_id="metric",
                kind="scalar",
                present_count=1,
                null_count=0,
                first_observed=_TIME,
                last_observed=_TIME,
            )
            for feature_id in feature_ids
        ),
        counts=VectorMetadataCounts(
            feature_vectors=1,
            target_vectors=0,
        ),
    )


def _output_plan(
    fold_id: str,
    output_id: str,
    schema: VectorSchema,
) -> FoldOutputPlan:
    return FoldOutputPlan(
        fold_id=fold_id,
        schema=schema,
        outputs={
            output_id: FoldOutputMetadata(
                role="train",
                labels=("all",),
                window=Window(
                    start=_TIME,
                    end=_TIME,
                    mode="intersection",
                    size=1,
                ),
            )
        },
    )


def test_shared_fold_scan_labels_once_and_projects_each_training_schema(
    monkeypatch,
    tmp_path,
) -> None:
    feature = SeriesConfig(id="metric", stream="metrics", field="value")
    dataset = DatasetConfig(
        sample=SampleConfig(cadence="1d"),
        features=[feature],
        split=TimeSplitConfig(
            intervals=[TimeInterval(id="all")],
            folds=[
                DatasetFold(id="early", train=["all"]),
                DatasetFold(id="later", train=["all"]),
            ],
        ),
    )
    runtime = Runtime(
        project_yaml=tmp_path / "project.yaml",
        artifacts_root=tmp_path / "artifacts",
        dataset=dataset,
    )
    sample = Sample(
        key=(_TIME,),
        features=Vector(
            {
                "metric__@bucket:known": 1.0,
                "metric__@bucket:future": None,
            }
        ),
    )
    monkeypatch.setattr(
        dataset_pipeline,
        "build_sample_input",
        lambda *_args: Input(
            name="assemble_samples",
            open=lambda: iter((sample,)),
        ),
    )
    label_calls = []
    label = dataset_pipeline.TimeLabeler.label

    def counted_label(labeler, key):
        label_calls.append(key)
        return label(labeler, key)

    monkeypatch.setattr(dataset_pipeline.TimeLabeler, "label", counted_label)
    early = _output_plan(
        "early",
        "early.train",
        _schema("metric__@bucket:known"),
    )
    later = _output_plan(
        "later",
        "later.train",
        _schema(
            "metric__@bucket:known",
            "metric__@bucket:future",
        ),
    )

    output = list(
        run_fold_outputs_pipeline(
            PipelineContext(runtime),
            (early, later),
        )
    )

    assert [row.key for _, row in output] == [sample.key, sample.key]
    assert [(output_id, row.features.values) for output_id, row in output] == [
        ("early.train", {"metric__@bucket:known": 1.0}),
        (
            "later.train",
            {
                "metric__@bucket:known": 1.0,
                "metric__@bucket:future": None,
            },
        ),
    ]
    assert label_calls == [sample.key]
