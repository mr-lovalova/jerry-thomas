from pathlib import Path

from jerrythomas.analysis.vector.coverage_stats import CoverageStatsAccumulator
from jerrythomas.artifacts.models import CoverageStatsArtifact
from jerrythomas.artifacts.output import ArtifactOutput
from jerrythomas.artifacts.registry import VECTOR_METADATA_SPEC
from jerrythomas.config.tasks.coverage_stats import CoverageStatsTask
from jerrythomas.io.json_file import write_json_object
from jerrythomas.pipelines.dataset.pipeline import (
    run_dataset_pipeline,
    run_sample_pipeline,
)
from jerrythomas.pipelines.sample.keys import require_metadata_key_plan
from jerrythomas.runtime import Runtime


def build_coverage_stats_artifact(
    runtime: Runtime,
    task_cfg: CoverageStatsTask,
) -> ArtifactOutput:
    dataset = runtime.require_dataset()
    metadata = runtime.artifacts.load(VECTOR_METADATA_SPEC)
    schema = metadata.catalog
    key_plan = require_metadata_key_plan(
        schema.window,
        schema.sample,
        dataset.sample.cadence,
        dataset.sample.keys,
    )

    if task_cfg.stage == "postprocessed":
        samples = run_dataset_pipeline(runtime, schema, key_plan)
    else:
        samples = run_sample_pipeline(runtime, schema, key_plan)

    feature_accumulator = CoverageStatsAccumulator(schema.features)
    target_accumulator = CoverageStatsAccumulator(schema.targets)
    total_samples = 0
    empty_samples = 0
    try:
        for sample in samples:
            total_samples += 1
            target_values = sample.targets.values if sample.targets is not None else {}
            if not sample.features.values and not target_values:
                empty_samples += 1
            feature_accumulator.update(sample.features.values)
            target_accumulator.update(target_values)
    finally:
        close = getattr(samples, "close", None)
        if callable(close):
            close()

    artifact = CoverageStatsArtifact(
        stage=task_cfg.stage,
        total_samples=total_samples,
        empty_samples=empty_samples,
        features=feature_accumulator.finish(),
        targets=target_accumulator.finish(),
    )
    relative_path = Path(task_cfg.path)
    destination = (runtime.artifacts_root / relative_path).resolve()
    write_json_object(destination, artifact.model_dump(mode="json"))

    return ArtifactOutput(
        meta={
            "stage": task_cfg.stage,
            "samples": total_samples,
            "features": len(artifact.features.columns),
            "targets": len(artifact.targets.columns),
        },
    )
