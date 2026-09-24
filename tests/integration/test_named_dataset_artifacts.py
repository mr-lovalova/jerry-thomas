from jerrythomas.artifacts.executor import run_build_if_needed
from jerrythomas.artifacts.registry import SERIES_SPEC
from jerrythomas.execution.settings import CommandObservability
from jerrythomas.profiles.request_builder import build_build_run_request
from jerrythomas.services.runtime_compiler import compile_runtime

from tests.named_dataset_helpers import create_project, write_config


def test_build_hydrates_callers_dataset_after_shared_schedule_context(tmp_path):
    project = create_project(tmp_path)
    write_config(
        tmp_path / "operations" / "calendar.yaml",
        {
            "kind": "artifact",
            "entrypoint": "core.artifact.schedule",
            "stream": "rows",
            "partition_by": ["id_"],
            "output": "shared/calendar.jsonl",
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
