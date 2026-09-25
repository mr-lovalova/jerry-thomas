from jerrythomas.artifacts.executor import run_build_if_needed
from jerrythomas.artifacts.registry import SERIES_SPEC
from jerrythomas.execution.settings import CommandObservability
from jerrythomas.profiles.orchestration import run_profiles
from jerrythomas.profiles.request_builder import (
    build_build_run_request,
    build_runtime_run_request,
)
from jerrythomas.services.runtime_compiler import compile_runtime

from tests.named_dataset_helpers import create_project, write_config


def test_build_hydrates_callers_dataset_after_shared_schedule_context(tmp_path):
    project = create_project(tmp_path)
    write_config(
        tmp_path / "operations" / "calendar.yaml",
        {
            "kind": "artifact",
            "entrypoint": "core.schedule",
            "stream": "rows",
            "partition_by": ["id_"],
            "path": "shared/calendar.jsonl",
        },
    )
    write_config(
        tmp_path / "streams" / "scheduled.yaml",
        {
            "id": "scheduled",
            "from": {"stream": "rows"},
            "transforms": [{"operation": "ensure_schedule", "schedule": "calendar"}],
        },
    )
    write_config(
        tmp_path / "datasets" / "alpha.yaml",
        {
            "version": "v1",
            "sample": {"cadence": "1d", "rounding": "exact", "keys": ["id_"]},
            "features": [{"id": "x", "stream": "scheduled", "field": "value"}],
        },
    )
    write_config(
        tmp_path / "profiles" / "build.alpha.yaml",
        {"operation": "dataset.alpha.series"},
    )
    request = build_build_run_request(
        str(project),
        profile_name="alpha",
        command_observability=CommandObservability(visuals=False, log_level="CRITICAL"),
    )
    assert request is not None
    definition = request.definition
    runtime = compile_runtime(definition, "alpha")
    dataset = runtime.require_dataset()
    assert definition.artifact_graph.dependency_closure({"dataset.alpha.series"}) == (
        "calendar",
        "dataset.alpha.series",
    )

    assert run_build_if_needed(
        definition,
        required_artifacts={"dataset.alpha.series"},
        settings=request.jobs[0].settings,
        runtime=runtime,
    )

    assert runtime.dataset_id == "alpha"
    assert runtime.require_dataset() is dataset
    assert runtime.artifacts.has("calendar")
    assert runtime.artifacts.load(SERIES_SPEC).rows == 3
    assert not runtime.artifacts.has("dataset.beta.series")


def test_explicit_operations_reuse_default_artifacts_and_preserve_dataset_outputs(
    tmp_path,
):
    project = create_project(tmp_path)
    write_config(
        tmp_path / "operations" / "serve-alpha.yaml",
        {
            "kind": "output",
            "entrypoint": "core.dataset",
            "dataset": "alpha",
            "requires": ["dataset.alpha.coverage_stats"],
        },
    )
    observability = CommandObservability(visuals=False, log_level="CRITICAL")
    implicit = build_runtime_run_request(
        "serve", str(project), profile_name="alpha", command_observability=observability
    )
    assert implicit is not None
    (original,) = run_profiles(implicit)
    original_rows = [path.read_bytes() for path in original.outputs]
    original_recipe = original.load_recipe()
    entrypoints = {
        "series": "core.dataset_series",
        "scaler": "core.scaler",
        "metadata": "core.dataset_metadata",
        "coverage_stats": "core.coverage_statistics",
    }
    assert set(original_recipe.configuration["artifacts"]) == {
        f"dataset.alpha.{kind}" for kind in entrypoints
    }
    assert {
        key: config["entrypoint"]
        for key, config in original_recipe.configuration["artifacts"].items()
    } == {
        f"dataset.alpha.{kind}": entrypoint for kind, entrypoint in entrypoints.items()
    }
    assert original_recipe.configuration["jobs"][0]["operation"]["kind"] == "output"
    assert (
        original_recipe.configuration["jobs"][0]["operation"]["entrypoint"]
        == "core.dataset"
    )

    for kind, entrypoint in entrypoints.items():
        write_config(
            tmp_path / "operations" / f"dataset.alpha.{kind}.yaml",
            {
                "kind": "artifact",
                "entrypoint": entrypoint,
                "dataset": "alpha",
                "path": implicit.definition.artifact_graph.tasks_by_id[
                    f"dataset.alpha.{kind}"
                ].output,
            },
        )
    explicit = build_runtime_run_request(
        "serve",
        str(project),
        profile_name="alpha",
        artifact_mode="require_current",
        command_observability=observability,
    )
    assert explicit is not None
    assert explicit.definition.artifact_hashes == implicit.definition.artifact_hashes

    # Requiring the existing cache must succeed without rebuilding any artifact.
    (saved,) = run_profiles(explicit)
    recipe = saved.load_recipe()
    assert [path.read_bytes() for path in saved.outputs] == original_rows
    assert (
        recipe.configuration["artifacts"] == original_recipe.configuration["artifacts"]
    )
    assert recipe.artifacts == original_recipe.artifacts
