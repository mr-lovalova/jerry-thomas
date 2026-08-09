from collections.abc import Sequence

from jerrythomas.config.dataset.series import SeriesConfig
from jerrythomas.config.preview import PreviewStage


def preview_output_plan(
    series: Sequence[SeriesConfig],
    preview: PreviewStage,
) -> tuple[tuple[str, SeriesConfig], ...]:
    if preview in {"samples", "postprocess"}:
        return ()
    if preview == "series":
        return tuple((config.id, config) for config in series)

    seen: set[str] = set()
    plan: list[tuple[str, SeriesConfig]] = []
    for config in series:
        if config.stream in seen:
            continue
        seen.add(config.stream)
        plan.append((config.stream, config))
    return tuple(plan)
