import shutil
from collections.abc import Sequence

from jerrythomas.artifacts.specs import SERIES
from jerrythomas.config.dataset.dataset import DatasetConfig, SampleConfig
from jerrythomas.config.dataset.series import SeriesConfig, TargetSeriesConfig
from jerrythomas.config.tasks.series import SeriesTask
from jerrythomas.operations.artifacts.series import build_series_artifact
from jerrythomas.runtime import Runtime


def register_series(
    runtime: Runtime,
    features: Sequence[SeriesConfig],
    cadence: str,
    *,
    targets: Sequence[TargetSeriesConfig] = (),
    sample_keys: Sequence[str] = (),
) -> None:
    current = runtime.dataset
    runtime.dataset = DatasetConfig(
        sample=SampleConfig(
            rounding=current.sample.rounding, cadence=cadence, keys=list(sample_keys)
        ),
        features=list(features),
        targets=list(targets),
        split=current.split,
        postprocess=current.postprocess,
    )
    shutil.rmtree(runtime.artifacts_root / "build/series", ignore_errors=True)
    task = SeriesTask()
    result = build_series_artifact(runtime, task)
    runtime.artifacts.register(
        SERIES,
        relative_path=task.output,
        meta=result.meta,
    )
