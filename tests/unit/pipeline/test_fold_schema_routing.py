from datetime import datetime, timezone

import jerrythomas.pipelines.dataset.pipeline as dataset_pipeline
from jerrythomas.artifacts.models import (
    FoldOutputMetadata,
    ScalarVectorMetadataEntry,
    VectorMetadataCounts,
    VectorSchema,
    Window,
)
from jerrythomas.config.dataset.dataset import DatasetConfig, SampleConfig
from jerrythomas.config.dataset.series import SeriesConfig
from jerrythomas.config.dataset.split import DatasetFold, TimeInterval, TimeSplitConfig
from jerrythomas.domain.sample import Sample
from jerrythomas.domain.vector import Vector
from jerrythomas.execution.pipeline import Input
from jerrythomas.pipelines.dataset.pipeline import (
    FoldOutputPlan,
    run_fold_outputs_pipeline,
)
from jerrythomas.pipelines.dataset.postprocess import PostprocessPlan
from jerrythomas.pipelines.sample.keys import window_key_plan
from jerrythomas.runtime import Runtime


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


def _route(
    feature_ids: frozenset[str],
    target_ids: frozenset[str],
) -> dataset_pipeline._FoldRoute:
    key_plan = window_key_plan(_TIME, _TIME, "1d")
    assert key_plan is not None
    return dataset_pipeline._FoldRoute(
        output_by_label={"all": ("all.train", key_plan)},
        feature_ids=feature_ids,
        target_ids=target_ids,
        postprocess=PostprocessPlan(stages=()),
        scaler=None,
    )


def test_fold_route_reuses_sample_when_schema_already_matches() -> None:
    sample = Sample(
        key=(_TIME,),
        features=Vector({"price": 10.0, "volume": 20.0}),
        targets=Vector({"return": 0.1}),
    )

    selected = next(
        _route(
            frozenset(("price", "volume")),
            frozenset(("return",)),
        ).select((("all", sample),))
    )

    assert selected is sample


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
            runtime,
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
