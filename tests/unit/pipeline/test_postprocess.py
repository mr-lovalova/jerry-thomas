import json

from jerrythomas.artifacts.registry import VECTOR_METADATA_SPEC
from jerrythomas.artifacts.specs import VECTOR_METADATA
from jerrythomas.artifacts.models import VectorMetadataCatalog
from jerrythomas.config.dataset.dataset import DatasetConfig, SampleConfig
from jerrythomas.config.dataset.postprocess import PostprocessConfig
from jerrythomas.domain.sample import Sample
from jerrythomas.domain.vector import Vector
from jerrythomas.pipelines.dataset.postprocess import build_postprocess_plan
from jerrythomas.pipelines.dataset.pipeline import build_dataset_pipeline
from jerrythomas.runtime import Runtime


def _runtime(
    tmp_path,
    catalog: dict,
    dataset: DatasetConfig | None = None,
) -> Runtime:
    artifacts_root = tmp_path / "artifacts"
    artifacts_root.mkdir()
    project = tmp_path / "project.yaml"
    project.write_text("schema_version: 6\nartifact_revision: 1\n", encoding="utf-8")
    if dataset is None:
        dataset = DatasetConfig(sample=SampleConfig(cadence="1h"))
    runtime = Runtime(
        project_yaml=project,
        artifacts_root=artifacts_root,
        dataset=dataset,
    )

    metadata_path = artifacts_root / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "catalog": catalog,
                "layout": {"kind": "unsplit"},
            }
        ),
        encoding="utf-8",
    )
    runtime.artifacts.register(VECTOR_METADATA, metadata_path.name)
    return runtime


def _catalog(runtime: Runtime) -> VectorMetadataCatalog:
    return runtime.artifacts.load(VECTOR_METADATA_SPEC).catalog


def test_dataset_pipeline_assembles_before_postprocess(tmp_path) -> None:
    runtime = _runtime(
        tmp_path,
        catalog={
            "counts": {"feature_vectors": 0, "target_vectors": 0},
            "features": [
                {
                    "id": "feature",
                    "base_id": "feature",
                    "kind": "scalar",
                    "present_count": 0,
                    "null_count": 0,
                }
            ],
            "targets": [],
        },
    )

    pipeline = build_dataset_pipeline(
        runtime,
        _catalog(runtime),
        None,
    )

    assert pipeline.name == "dataset"
    assert pipeline.input.name == "assemble_samples"
    assert [stage.name for stage in pipeline.stages] == [
        "conform_features",
        "reject_undeclared_targets",
    ]
    assert pipeline.input.progress is None


def test_postprocess_has_one_explicit_execution_order(tmp_path) -> None:
    runtime = _runtime(
        tmp_path,
        catalog={
            "counts": {"feature_vectors": 2, "target_vectors": 0},
            "features": [
                {
                    "id": "sparse",
                    "base_id": "sparse",
                    "kind": "scalar",
                    "present_count": 0,
                    "null_count": 0,
                },
                {
                    "id": "value",
                    "base_id": "value",
                    "kind": "scalar",
                    "present_count": 1,
                    "null_count": 0,
                },
            ],
            "targets": [],
        },
        dataset=DatasetConfig(
            sample=SampleConfig(cadence="1h"),
            postprocess=PostprocessConfig.model_validate(
                {"features": {"threshold": 1.0, "ids": ["value"]}}
            ),
        ),
    )
    samples = [
        Sample(key=(0,), features=Vector(values={"sparse": None})),
        Sample(
            key=(1,),
            features=Vector(values={"sparse": None, "value": 2.0}),
        ),
    ]

    schema = _catalog(runtime)
    plan = build_postprocess_plan(runtime.dataset.postprocess, schema)
    output = list(plan.apply(iter(samples)))

    assert [stage.name for stage in plan.stages] == [
        "conform_features",
        "reject_undeclared_targets",
        "filter_samples_by_features",
    ]
    assert [sample.features.values for sample in output] == [
        {"sparse": None, "value": 2.0}
    ]


def test_postprocess_applies_explicit_target_policies(tmp_path) -> None:
    runtime = _runtime(
        tmp_path,
        catalog={
            "counts": {"feature_vectors": 1, "target_vectors": 1},
            "features": [
                {
                    "id": "feature",
                    "base_id": "feature",
                    "kind": "scalar",
                    "present_count": 1,
                    "null_count": 0,
                }
            ],
            "targets": [
                {
                    "id": "sparse",
                    "base_id": "sparse",
                    "kind": "scalar",
                    "present_count": 0,
                    "null_count": 0,
                },
                {
                    "id": "target",
                    "base_id": "target",
                    "kind": "scalar",
                    "present_count": 1,
                    "null_count": 0,
                },
            ],
        },
        dataset=DatasetConfig.model_validate(
            {
                "sample": {"cadence": "1h"},
                "features": [{"id": "feature", "stream": "features", "field": "value"}],
                "targets": [
                    {
                        "id": "sparse",
                        "stream": "targets",
                        "field": "sparse",
                        "horizon": "0s",
                    },
                    {
                        "id": "target",
                        "stream": "targets",
                        "field": "value",
                        "horizon": "0s",
                    },
                ],
                "postprocess": {"targets": {"threshold": 1.0, "ids": ["target"]}},
            }
        ),
    )
    sample = Sample(
        key=(0,),
        features=Vector(values={"feature": 2.0}),
        targets=Vector(values={"sparse": None, "target": 1.0}),
    )

    output = list(
        build_postprocess_plan(
            runtime.dataset.postprocess,
            _catalog(runtime),
        ).apply(iter([sample]))
    )

    assert output[0].features.values == {"feature": 2.0}
    assert output[0].targets is not None
    assert output[0].targets.values == {"sparse": None, "target": 1.0}


def test_metadata_coverage_counts_never_change_the_dataset_schema(
    tmp_path,
) -> None:
    runtime = _runtime(
        tmp_path,
        catalog={
            "counts": {"feature_vectors": 100, "target_vectors": 0},
            "window": {
                "start": "2024-01-01T00:00:00Z",
                "end": "2024-01-05T00:00:00Z",
                "mode": "union",
                "size": 5,
            },
            "features": [
                {
                    "id": "sparse",
                    "base_id": "sparse",
                    "kind": "scalar",
                    "present_count": 3,
                    "null_count": 0,
                },
                {
                    "id": "complete",
                    "base_id": "complete",
                    "kind": "scalar",
                    "present_count": 100,
                    "null_count": 0,
                },
            ],
            "targets": [],
        },
    )
    sample = Sample(
        key=(0,),
        features=Vector(values={"sparse": 1.0, "complete": 2.0}),
    )

    output = list(
        build_postprocess_plan(
            runtime.dataset.postprocess,
            _catalog(runtime),
        ).apply(iter([sample]))
    )

    assert output[0].features.values == {"sparse": 1.0, "complete": 2.0}
    assert [
        entry.id
        for entry in runtime.artifacts.load(VECTOR_METADATA_SPEC).catalog.features
    ] == ["sparse", "complete"]
