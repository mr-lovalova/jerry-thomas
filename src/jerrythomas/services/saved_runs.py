from pathlib import Path

from jerrythomas.config.profiles.serve import ServeProfile
from jerrythomas.io.runs import SavedRun, load_run
from jerrythomas.profiles.loader import (
    apply_profile_defaults,
    bind_profile,
    profile_specs_with_defaults,
)
from jerrythomas.profiles.runtime_profiles import resolve_serve_profiles
from jerrythomas.services.project_definition import load_project_definition


def load_dataset_run(
    project: str | Path,
    dataset: str,
    version: str,
    run_id: str,
) -> SavedRun:
    """Locate a saved run using configured dataset output roots, then verify its identity."""
    if not run_id or run_id in {".", ".."} or any(c in run_id for c in "/\\:\x00"):
        raise ValueError("run_id must be a single run directory name")
    definition = load_project_definition(Path(project))
    config = definition.require_dataset(dataset)
    # Apply the same version validation as a dataset declaration, without changing
    # the resolved definition or using its current version to locate older runs.
    type(config).model_validate({**config.model_dump(), "version": version})
    operations = {task.id: task for task in definition.runtime_operations}
    profiles, defaults = profile_specs_with_defaults(definition.project, "serve")
    selected = []
    for profile in profiles:
        if not isinstance(profile, ServeProfile):
            continue
        task = operations.get(profile.operation)
        if task is None or task.dataset != dataset:
            continue
        bound = bind_profile(
            apply_profile_defaults(profile, defaults),
            definition,
            dataset_version=version,
        )
        assert isinstance(bound, ServeProfile)
        selected.append(bound)
    resolved = resolve_serve_profiles(definition, selected, None, None, None)
    roots = {
        profile.output.run.serve_root
        for profile in resolved
        if profile.output.run is not None
    }
    candidates = {
        (root / "runs" / run_id / "run.json").resolve()
        for root in roots
        if (root / "runs" / run_id / "run.json").is_file()
    }
    if not candidates:
        raise FileNotFoundError(
            f"No saved run {run_id!r} for dataset {dataset!r}, version {version!r} "
            "under its configured serve output directories."
        )
    if len(candidates) != 1:
        raise ValueError(
            f"Saved run {run_id!r} is ambiguous across configured output directories."
        )
    run = load_run(candidates.pop())
    if (
        run.metadata.command != "serve"
        or run.metadata.run_id != run_id
        or run.metadata.dataset_id != dataset
        or run.metadata.dataset_version != version
    ):
        raise ValueError(
            "Saved run identity does not match the requested dataset/version/run"
        )
    return run
