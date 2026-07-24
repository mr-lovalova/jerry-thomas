from collections.abc import Collection, Generator, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from functools import partial
from itertools import islice

from datapipeline.artifacts.registry import SCALER_SPEC
from datapipeline.artifacts.scaler import (
    FoldedScalerArtifact,
    StandardScalerArtifact,
)
from datapipeline.artifacts.specs import dataset_requires_scaler
from datapipeline.config.dataset.split import DatasetFold, TimeSplitConfig
from datapipeline.config.dataset.series import SeriesConfig
from datapipeline.domain.sample import Sample
from datapipeline.execution.context import PipelineContext
from datapipeline.execution.pipeline import Pipeline, Stage
from datapipeline.execution.runner import run_pipeline
from datapipeline.pipelines.dataset.postprocess import (
    PostprocessPlan,
    build_postprocess_plan,
)
from datapipeline.pipelines.dataset.split import (
    HashLabeler,
    TargetHorizonPolicy,
    TimeLabeler,
    build_labeler,
)
from datapipeline.pipelines.sample.input import build_sample_input
from datapipeline.transforms.vector.scaler import SampleScaler

_FOLD_BATCH_SIZE = 256


@dataclass(frozen=True)
class FoldOutputPlan:
    fold: DatasetFold
    output_by_label: Mapping[str, str]

    def __post_init__(self) -> None:
        if not self.output_by_label:
            raise ValueError(f"Dataset fold {self.fold.id!r} has no selected outputs.")
        fold_labels = frozenset(
            (*self.fold.train, *self.fold.validation, *self.fold.test)
        )
        unknown = self.output_by_label.keys() - fold_labels
        if unknown:
            raise ValueError(
                f"Dataset fold {self.fold.id!r} does not contain selected labels: "
                + ", ".join(sorted(unknown))
            )


@dataclass(frozen=True)
class _FoldRoute:
    output_by_label: Mapping[str, str]
    scaler: SampleScaler | None
    horizon_policy: TargetHorizonPolicy | None

    def select(
        self,
        labeled: Sequence[tuple[str, Sample]],
    ) -> Iterator[Sample]:
        policy = self.horizon_policy
        return (
            sample
            for label, sample in labeled
            if label in self.output_by_label
            and (policy is None or policy.allows(label, sample.key))
        )


def run_dataset_pipeline(
    context: PipelineContext,
    feature_configs: Sequence[SeriesConfig],
    group_by_cadence: str,
    target_configs: Sequence[SeriesConfig] | None = None,
    rectangular: bool = True,
    sample_keys: Sequence[str] = (),
) -> Generator[Sample, None, None]:
    return run_pipeline(
        context,
        build_dataset_pipeline(
            context,
            feature_configs,
            group_by_cadence,
            target_configs=target_configs,
            rectangular=rectangular,
            sample_keys=sample_keys,
        ),
    )


def build_dataset_pipeline(
    context: PipelineContext,
    feature_configs: Sequence[SeriesConfig],
    group_by_cadence: str,
    target_configs: Sequence[SeriesConfig] | None = None,
    rectangular: bool = True,
    sample_keys: Sequence[str] = (),
) -> Pipeline:
    postprocess = build_postprocess_plan(context)
    return Pipeline(
        name="dataset",
        input=build_sample_input(
            context,
            feature_configs,
            group_by_cadence,
            target_configs,
            rectangular,
            sample_keys,
        ),
        stages=postprocess.stages,
    )


def run_scaled_dataset_pipeline(
    context: PipelineContext,
    feature_configs: Sequence[SeriesConfig],
    group_by_cadence: str,
    target_configs: Sequence[SeriesConfig] | None = None,
    rectangular: bool = True,
    sample_keys: Sequence[str] = (),
) -> Generator[Sample, None, None]:
    artifact = context.require_artifact(SCALER_SPEC)
    if not isinstance(artifact, StandardScalerArtifact):
        raise RuntimeError(
            "A dataset without folds requires a standard scaler artifact."
        )
    scaler = _sample_scaler(artifact, feature_configs, target_configs)
    postprocess = build_postprocess_plan(context)
    return run_pipeline(
        context,
        Pipeline(
            name="dataset",
            input=build_sample_input(
                context,
                feature_configs,
                group_by_cadence,
                target_configs,
                rectangular,
                sample_keys,
            ),
            stages=(
                Stage(name="scale_samples", apply=scaler.apply),
                *postprocess.stages,
            ),
        ),
    )


def run_fold_dataset_pipeline(
    context: PipelineContext,
    feature_configs: Sequence[SeriesConfig],
    group_by_cadence: str,
    fold: DatasetFold,
    labels: Collection[str],
    target_configs: Sequence[SeriesConfig] | None = None,
    rectangular: bool = True,
    sample_keys: Sequence[str] = (),
) -> Generator[Sample, None, None]:
    included = frozenset(labels)
    if not included:
        raise ValueError(f"Dataset fold {fold.id!r} has no selected output labels.")
    routed = run_fold_outputs_pipeline(
        context,
        feature_configs,
        group_by_cadence,
        (
            FoldOutputPlan(
                fold=fold,
                output_by_label={label: fold.id for label in included},
            ),
        ),
        target_configs=target_configs,
        rectangular=rectangular,
        sample_keys=sample_keys,
    )
    try:
        for _output_id, sample in routed:
            yield sample
    finally:
        routed.close()


def run_fold_outputs_pipeline(
    context: PipelineContext,
    feature_configs: Sequence[SeriesConfig],
    group_by_cadence: str,
    outputs: Sequence[FoldOutputPlan],
    target_configs: Sequence[SeriesConfig] | None = None,
    rectangular: bool = True,
    sample_keys: Sequence[str] = (),
) -> Generator[tuple[str, Sample], None, None]:
    split = context.runtime.dataset.split
    if split is None:
        raise ValueError("Fold dataset output requires dataset split configuration.")
    plans = tuple(outputs)
    if not plans:
        raise ValueError("Fold dataset output requires at least one selected fold.")

    scaler_artifact: FoldedScalerArtifact | None = None
    if dataset_requires_scaler(context.runtime.dataset):
        artifact = context.require_artifact(SCALER_SPEC)
        if not isinstance(artifact, FoldedScalerArtifact):
            raise RuntimeError("A split dataset requires a folded scaler artifact.")
        scaler_artifact = artifact

    target_horizon = context.runtime.dataset.max_target_horizon
    routes = tuple(
        _FoldRoute(
            output_by_label=plan.output_by_label,
            scaler=(
                None
                if scaler_artifact is None
                else _sample_scaler(
                    scaler_artifact.for_fold(plan.fold.id),
                    feature_configs,
                    target_configs,
                )
            ),
            horizon_policy=(
                TargetHorizonPolicy(split, plan.fold, target_horizon)
                if isinstance(split, TimeSplitConfig)
                and target_horizon > timedelta()
                else None
            ),
        )
        for plan in plans
    )
    postprocess = build_postprocess_plan(context)
    return run_pipeline(
        context,
        Pipeline(
            name="dataset",
            input=build_sample_input(
                context,
                feature_configs,
                group_by_cadence,
                target_configs,
                rectangular,
                sample_keys,
            ),
            stages=(
                Stage(
                    name="prepare_fold_outputs",
                    apply=partial(
                        _prepare_fold_outputs,
                        build_labeler(split),
                        routes,
                        postprocess,
                    ),
                ),
            ),
        ),
    )


def _prepare_fold_outputs(
    labeler: HashLabeler | TimeLabeler,
    routes: Sequence[_FoldRoute],
    postprocess: PostprocessPlan,
    samples: Iterator[Sample],
) -> Iterator[tuple[str, Sample]]:
    while batch := tuple(islice(samples, _FOLD_BATCH_SIZE)):
        labeled = tuple((labeler.label(sample.key), sample) for sample in batch)
        for route in routes:
            selected = route.select(labeled)
            processed = (
                selected if route.scaler is None else route.scaler.apply(selected)
            )
            for sample in postprocess.apply(processed):
                output_id = route.output_by_label.get(labeler.label(sample.key))
                if output_id is not None:
                    yield output_id, sample


def _sample_scaler(
    artifact: StandardScalerArtifact,
    feature_configs: Sequence[SeriesConfig],
    target_configs: Sequence[SeriesConfig] | None,
) -> SampleScaler:
    return SampleScaler(
        artifact,
        scaled_feature_ids=tuple(
            config.id for config in feature_configs if config.scale
        ),
        scaled_target_ids=tuple(
            config.id
            for config in (() if target_configs is None else target_configs)
            if config.scale
        ),
    )
