from datetime import datetime

from datapipeline.artifacts.models import VectorMetadata
from datapipeline.artifacts.registry import (
    VECTOR_METADATA_SPEC,
    ArtifactNotRegisteredError,
)
from datapipeline.runtime import Runtime

WindowBounds = tuple[datetime | None, datetime | None]


def resolve_window_bounds(
    runtime: Runtime,
    rectangular_required: bool,
) -> WindowBounds:
    existing = getattr(runtime, "window_bounds", None)
    cached_bounds = _usable_cached_bounds(existing, rectangular_required)
    if cached_bounds is not None:
        return cached_bounds

    start, end = _metadata_window_bounds(runtime)

    if rectangular_required and (start is None or end is None):
        raise RuntimeError(
            "Window bounds unavailable (rebuild metadata to materialize build/metadata.json with a window); rectangular output required."
        )
    return start, end


def _usable_cached_bounds(
    existing: object,
    rectangular_required: bool,
) -> WindowBounds | None:
    if not isinstance(existing, tuple) or len(existing) != 2:
        return None
    cached_start, cached_end = existing
    if not rectangular_required and (
        cached_start is not None or cached_end is not None
    ):
        return cached_start, cached_end
    if cached_start is not None and cached_end is not None:
        return cached_start, cached_end
    return None


def _optional_metadata(runtime: Runtime) -> VectorMetadata | None:
    try:
        return runtime.artifacts.load(VECTOR_METADATA_SPEC)
    except ArtifactNotRegisteredError:
        return None


def _metadata_window_bounds(runtime: Runtime) -> WindowBounds:
    metadata = _optional_metadata(runtime)
    if metadata is None:
        return None, None
    if metadata.catalog.window is None:
        return None, None
    return metadata.catalog.window.start, metadata.catalog.window.end
