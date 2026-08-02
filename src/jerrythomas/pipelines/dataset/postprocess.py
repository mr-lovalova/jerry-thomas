from collections.abc import Iterator
from dataclasses import dataclass

from jerrythomas.artifacts.models import VectorSchema
from jerrythomas.config.dataset.postprocess import PostprocessConfig
from jerrythomas.domain.sample import Sample
from jerrythomas.execution.pipeline import Stage
from jerrythomas.transforms.vector.sample_filter import (
    FilterFeatureSamplesTransform,
    FilterTargetSamplesTransform,
)
from jerrythomas.transforms.vector.conform import (
    ConformFeaturesTransform,
    ConformTargetsTransform,
)


@dataclass(frozen=True)
class PostprocessPlan:
    stages: tuple[Stage, ...]

    def apply(self, samples: Iterator[Sample]) -> Iterator[Sample]:
        stream = samples
        for stage in self.stages:
            stream = iter(stage.apply(stream))
        return stream


def build_postprocess_plan(
    config: PostprocessConfig,
    schema: VectorSchema,
) -> PostprocessPlan:
    if not schema.features:
        raise RuntimeError(
            "Metadata has no feature entries. Rebuild build/metadata.json."
        )

    feature_entries = schema.features
    target_entries = schema.targets
    stages: list[Stage] = []

    stages.append(
        Stage(
            name="conform_features",
            apply=ConformFeaturesTransform(feature_entries).apply,
        )
    )

    if target_entries:
        stages.append(
            Stage(
                name="conform_targets",
                apply=ConformTargetsTransform(target_entries).apply,
            )
        )
    else:
        stages.append(
            Stage(
                name="reject_undeclared_targets",
                apply=_reject_undeclared_targets,
            )
        )

    if config.features is not None:
        feature_filter = FilterFeatureSamplesTransform(
            [entry.id for entry in feature_entries],
            config.features.threshold,
            config.features.ids,
        )
        stages.append(
            Stage(
                name="filter_samples_by_features",
                apply=feature_filter.apply,
            )
        )

    if config.targets is not None:
        target_filter = FilterTargetSamplesTransform(
            [entry.id for entry in target_entries],
            config.targets.threshold,
            config.targets.ids,
        )
        stages.append(
            Stage(
                name="filter_samples_by_targets",
                apply=target_filter.apply,
            )
        )

    return PostprocessPlan(stages=tuple(stages))


def _reject_undeclared_targets(stream: Iterator[Sample]) -> Iterator[Sample]:
    for sample in stream:
        if sample.targets is not None:
            raise RuntimeError(
                "Metadata has no target entries, but the pipeline produced targets. "
                "Rebuild build/metadata.json."
            )
        yield sample
