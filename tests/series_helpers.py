import shutil
from collections.abc import Sequence

from datapipeline.artifacts.specs import SERIES
from datapipeline.config.dataset.dataset import DatasetConfig, SampleConfig
from datapipeline.config.dataset.series import SeriesConfig, TargetSeriesConfig
from datapipeline.config.tasks.series import SeriesTask
from datapipeline.operations.artifacts.series import build_series_artifact
from datapipeline.runtime import Runtime


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
        sample=SampleConfig(cadence=cadence, keys=list(sample_keys)),
        features=list(features),
        targets=list(targets),
        split=current.split,
        postprocess=current.postprocess,
    )
    shutil.rmtree(runtime.artifacts_root / "build/series", ignore_errors=True)
    result = build_series_artifact(runtime, SeriesTask())
    runtime.artifacts.register(
        SERIES,
        relative_path=result.relative_path,
        meta=result.meta,
    )
