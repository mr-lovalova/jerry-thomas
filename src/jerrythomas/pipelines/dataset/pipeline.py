from collections.abc import Generator, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
from itertools import islice

from jerrythomas.artifacts.models import (
    FoldedMetadataLayout,
    FoldOutputMetadata,
    VectorMetadataFold,
    VectorSchema,
)
from jerrythomas.artifacts.registry import SCALER_SPEC, VECTOR_METADATA_SPEC
from jerrythomas.artifacts.scaler import (
    FoldedScalerArtifact,
    StandardScalerArtifact,
)
from jerrythomas.artifacts.specs import dataset_requires_scaler
from jerrythomas.config.dataset.split import resolve_fold_output
from jerrythomas.domain.sample import Sample
from jerrythomas.domain.vector import Vector
from jerrythomas.execution.pipeline import Pipeline, Stage
from jerrythomas.execution.runner import run_pipeline
from jerrythomas.pipelines.dataset.postprocess import (
    PostprocessPlan,
    build_postprocess_plan,
)
from jerrythomas.pipelines.dataset.split import HashLabeler, TimeLabeler, build_labeler
from jerrythomas.pipelines.sample.input import build_sample_input
from jerrythomas.pipelines.sample.keys import (
    RectangularKeyPlan,
    merge_rectangular_key_plans,
    metadata_key_plan,
)
from jerrythomas.transforms.vector.scaler import SampleScaler
from jerrythomas.runtime import Runtime

_FOLD_BATCH_SIZE = 256


@dataclass(frozen=True)
class FoldOutputPlan:
    fold_id: str
    schema: VectorSchema
    outputs: Mapping[str, FoldOutputMetadata]


def resolve_fold_output_plans(
    runtime: Runtime,
    output_ids: Sequence[str],
) -> tuple[FoldOutputPlan, ...]:
    split = runtime.dataset.split
    if split is None:
        raise ValueError("Fold outputs require dataset split configuration.")

    metadata = runtime.artifacts.load(VECTOR_METADATA_SPEC)
    if not isinstance(metadata.layout, FoldedMetadataLayout):
        raise RuntimeError(
            "Split dataset requires folded metadata. Rebuild build/metadata.json."
        )
    metadata_by_fold = {fold.id: fold for fold in metadata.layout.folds}
    selected: dict[
        str,
        tuple[VectorMetadataFold, dict[str, FoldOutputMetadata]],
    ] = {}
    for output_id in output_ids:
        fold, role, labels = resolve_fold_output(split, output_id)
        try:
            fold_metadata = metadata_by_fold[fold.id]
        except KeyError as exc:
            raise RuntimeError(
                f"Metadata has no contract for dataset fold {fold.id!r}."
            ) from exc
        output_metadata = next(
            (output for output in fold_metadata.outputs if output.role == role),
            None,
        )
        if output_metadata is None or output_metadata.labels != labels:
            raise RuntimeError(
                f"Metadata contract for dataset output {output_id!r} does not "
                "match the configured fold. Rebuild build/metadata.json."
            )
        entry = selected.get(fold.id)
        if entry is None:
            outputs: dict[str, FoldOutputMetadata] = {}
            selected[fold.id] = fold_metadata, outputs
        else:
            _, outputs = entry
        outputs[output_id] = output_metadata

    return tuple(
        FoldOutputPlan(
            fold_id=fold.id,
            schema=fold.training_schema,
            outputs=outputs,
        )
        for fold, outputs in selected.values()
    )


@dataclass(frozen=True)
class _FoldRoute:
    output_by_label: Mapping[str, tuple[str, RectangularKeyPlan]]
    feature_ids: frozenset[str]
    target_ids: frozenset[str]
    postprocess: PostprocessPlan
    scaler: SampleScaler | None

    def select(
        self,
        labeled: Sequence[tuple[str, Sample]],
    ) -> Iterator[Sample]:
        for label, sample in labeled:
            output = self.output_by_label.get(label)
            if output is None:
                continue
            _, key_plan = output
            if key_plan.contains(sample.key):
                features = _select_vector(sample.features, self.feature_ids)
                targets = sample.targets
                if not self.target_ids:
                    targets = None
                elif targets is None:
                    targets = Vector({})
                else:
                    targets = _select_vector(targets, self.target_ids)

                if features is sample.features and targets is sample.targets:
                    yield sample
                else:
                    yield replace(sample, features=features, targets=targets)


def _select_vector(vector: Vector, selected_ids: frozenset[str]) -> Vector:
    if vector.values.keys() <= selected_ids:
        return vector
    return replace(
        vector,
        values={
            series_id: value
            for series_id, value in vector.values.items()
            if series_id in selected_ids
        },
    )


def run_dataset_pipeline(
    runtime: Runtime,
    schema: VectorSchema,
    key_plan: RectangularKeyPlan | None,
) -> Iterator[Sample]:
    return run_pipeline(
        runtime,
        build_dataset_pipeline(
            runtime,
            schema,
            key_plan,
        ),
    )


def run_sample_pipeline(
    runtime: Runtime,
    schema: VectorSchema,
    key_plan: RectangularKeyPlan | None,
) -> Iterator[Sample]:
    return run_pipeline(runtime, build_sample_pipeline(runtime, schema, key_plan))


def build_sample_pipeline(
    runtime: Runtime,
    schema: VectorSchema,
    key_plan: RectangularKeyPlan | None,
) -> Pipeline:
    return Pipeline(
        name="dataset",
        input=build_sample_input(
            runtime,
            tuple(entry.id for entry in schema.features),
            tuple(entry.id for entry in schema.targets),
            key_plan,
        ),
    )


def build_dataset_pipeline(
    runtime: Runtime,
    schema: VectorSchema,
    key_plan: RectangularKeyPlan | None,
) -> Pipeline:
    pipeline = build_sample_pipeline(runtime, schema, key_plan)
    postprocess = build_postprocess_plan(
        runtime.dataset.postprocess,
        schema,
    )
    return replace(
        pipeline,
        stages=postprocess.stages,
    )


def run_scaled_dataset_pipeline(
    runtime: Runtime,
    schema: VectorSchema,
    key_plan: RectangularKeyPlan | None,
) -> Iterator[Sample]:
    artifact = runtime.artifacts.load(SCALER_SPEC)
    if not isinstance(artifact, StandardScalerArtifact):
        raise RuntimeError(
            "A dataset without folds requires a standard scaler artifact."
        )
    scaler = _sample_scaler(runtime, artifact)
    pipeline = build_dataset_pipeline(runtime, schema, key_plan)
    return run_pipeline(
        runtime,
        replace(
            pipeline,
            stages=(
                Stage(name="scale_samples", apply=scaler.apply),
                *pipeline.stages,
            ),
        ),
    )


def run_fold_dataset_pipeline(
    runtime: Runtime,
    output: FoldOutputPlan,
) -> Generator[Sample, None, None]:
    routed = run_fold_outputs_pipeline(
        runtime,
        (output,),
    )
    try:
        for _output_id, sample in routed:
            yield sample
    finally:
        close = getattr(routed, "close", None)
        if callable(close):
            close()


def run_fold_outputs_pipeline(
    runtime: Runtime,
    outputs: Sequence[FoldOutputPlan],
) -> Iterator[tuple[str, Sample]]:
    dataset = runtime.dataset
    split = dataset.split
    if split is None:
        raise ValueError("Fold dataset output requires dataset split configuration.")
    plans = tuple(outputs)
    if not plans:
        raise ValueError("Fold dataset output requires at least one selected fold.")

    scaler_artifact: FoldedScalerArtifact | None = None
    if dataset_requires_scaler(dataset):
        artifact = runtime.artifacts.load(SCALER_SPEC)
        if not isinstance(artifact, FoldedScalerArtifact):
            raise RuntimeError("A split dataset requires a folded scaler artifact.")
        scaler_artifact = artifact

    route_key_plans = tuple(
        {
            output_id: key_plan
            for output_id, metadata in plan.outputs.items()
            if (
                key_plan := metadata_key_plan(
                    metadata.window,
                    metadata.sample,
                    dataset.sample.cadence,
                    dataset.sample.keys,
                )
            )
            is not None
        }
        for plan in plans
    )
    selected_key_plans = tuple(
        key_plan for key_plans in route_key_plans for key_plan in key_plans.values()
    )
    if not selected_key_plans:
        return iter(())

    routes = tuple(
        _FoldRoute(
            output_by_label={
                label: (output_id, key_plans[output_id])
                for output_id, output in plan.outputs.items()
                if output_id in key_plans
                for label in output.labels
            },
            feature_ids=frozenset(entry.id for entry in plan.schema.features),
            target_ids=frozenset(entry.id for entry in plan.schema.targets),
            postprocess=build_postprocess_plan(
                dataset.postprocess,
                plan.schema,
            ),
            scaler=(
                None
                if scaler_artifact is None
                else _sample_scaler(runtime, scaler_artifact.for_fold(plan.fold_id))
            ),
        )
        for plan, key_plans in zip(plans, route_key_plans, strict=True)
    )
    return run_pipeline(
        runtime,
        Pipeline(
            name="dataset",
            input=build_sample_input(
                runtime,
                {entry.id for plan in plans for entry in plan.schema.features},
                {entry.id for plan in plans for entry in plan.schema.targets},
                merge_rectangular_key_plans(selected_key_plans),
            ),
            stages=(
                Stage(
                    name="prepare_fold_outputs",
                    apply=partial(
                        _prepare_fold_outputs,
                        build_labeler(split),
                        routes,
                    ),
                ),
            ),
        ),
    )


def _prepare_fold_outputs(
    labeler: HashLabeler | TimeLabeler,
    routes: Sequence[_FoldRoute],
    samples: Iterator[Sample],
) -> Iterator[tuple[str, Sample]]:
    while batch := tuple(islice(samples, _FOLD_BATCH_SIZE)):
        labeled = tuple((labeler.label(sample.key), sample) for sample in batch)
        labels_by_key = {sample.key: label for label, sample in labeled}
        for route in routes:
            selected = route.select(labeled)
            processed = (
                selected if route.scaler is None else route.scaler.apply(selected)
            )
            for sample in route.postprocess.apply(processed):
                output = route.output_by_label.get(labels_by_key[sample.key])
                if output is not None:
                    yield output[0], sample


def _sample_scaler(
    runtime: Runtime,
    artifact: StandardScalerArtifact,
) -> SampleScaler:
    dataset = runtime.dataset
    return SampleScaler(
        artifact,
        scaled_feature_ids=tuple(
            config.id for config in dataset.features if config.scale
        ),
        scaled_target_ids=tuple(
            config.id for config in dataset.targets if config.scale
        ),
    )
