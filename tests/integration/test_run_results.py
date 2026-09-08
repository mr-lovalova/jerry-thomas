import json

import pytest

from jerrythomas.config.profiles.output import ServeOutputConfig
from jerrythomas.execution.settings import CommandObservability
from jerrythomas.profiles.orchestration import run_profiles
from jerrythomas.profiles.request_builder import build_runtime_run_request


@pytest.mark.parametrize("fixture", ["regression_project", "walk_forward_project"])
@pytest.mark.parametrize(
    ("fmt", "compression", "suffix"),
    [
        ("jsonl", None, ".jsonl"),
        ("jsonl", "gzip", ".jsonl.gz"),
        ("parquet", None, ".parquet"),
    ],
)
def test_run_results_identify_committed_files(
    copy_fixture, fixture, fmt, compression, suffix
):
    root = copy_fixture(fixture)
    request = build_runtime_run_request(
        "serve",
        str(root / "project.yaml"),
        profile_name="dataset",
        cli_output=ServeOutputConfig(
            transport="fs",
            format=fmt,
            compression=compression,
            directory=root / "served",
        ),
        command_observability=CommandObservability(visuals=False),
    )
    results = run_profiles(request)
    assert len(results) == 1
    result = results[0]
    assert result.paths == request.serve_run_plans[0].paths
    assert result.preview is None
    metadata = json.loads(result.paths.metadata_path.read_text())
    assert metadata["run_id"] == result.paths.run_id
    assert metadata["status"] == "success"
    assert (result.paths.serve_root / "latest").resolve() == result.paths.run_root
    expected_names = (
        [f"dataset.{output_id}{suffix}" for output_id in request.jobs[0].output_ids]
        if request.jobs[0].output_ids
        else [f"dataset{suffix}"]
    )
    assert [path.name for path in result.outputs] == expected_names
    assert set(result.outputs) == set(result.paths.dataset_dir.iterdir())
    assert all(path.is_file() for path in result.outputs)


@pytest.mark.parametrize("shared_directory", [True, False])
def test_run_results_group_profiles_in_plan_order(copy_fixture, shared_directory):
    root = copy_fixture("regression_project")
    profiles = root / "profiles"
    for index, name in enumerate(("dataset", "second")):
        directory = "output" if shared_directory else f"output-{name}"
        (profiles / f"serve.{name}.yaml").write_text(
            f"operation: dataset\norder: {index}\n"
            f"output: {{transport: fs, format: jsonl, directory: {directory}}}\n"
        )
    request = build_runtime_run_request(
        "serve",
        str(root / "project.yaml"),
        command_observability=CommandObservability(visuals=False),
    )
    results = run_profiles(request)
    assert [result.paths for result in results] == [
        plan.paths for plan in request.serve_run_plans
    ]
    assert [tuple(path.name for path in result.outputs) for result in results] == (
        [("dataset.jsonl", "second.jsonl")]
        if shared_directory
        else [("dataset.jsonl",), ("second.jsonl",)]
    )
    assert all(path.is_file() for result in results for path in result.outputs)


def test_result_excludes_planned_files_when_operation_returns_no_output(
    copy_fixture, monkeypatch
):
    root = copy_fixture("regression_project")
    request = build_runtime_run_request("serve", str(root / "project.yaml"))
    monkeypatch.setattr(
        "jerrythomas.profiles.execution.run_runtime_operation", lambda _job: None
    )
    results = run_profiles(request)
    assert len(results) == 1
    assert results[0].outputs == ()
    assert list(results[0].paths.dataset_dir.iterdir()) == []


def test_stdout_has_no_managed_run_result(copy_fixture, capsys):
    root = copy_fixture("regression_project")
    request = build_runtime_run_request(
        "serve",
        str(root / "project.yaml"),
        limit=1,
        cli_output=ServeOutputConfig(transport="stdout", format="jsonl"),
        command_observability=CommandObservability(visuals=False),
    )
    assert run_profiles(request) == ()
    assert json.loads(capsys.readouterr().out)["features"]


def test_preview_result_does_not_publish_latest(copy_fixture):
    root = copy_fixture("regression_project")
    request = build_runtime_run_request(
        "serve",
        str(root / "project.yaml"),
        preview="samples",
        limit=1,
        command_observability=CommandObservability(visuals=False),
    )
    (result,) = run_profiles(request)
    assert result.preview == "samples"
    assert result.outputs == (result.paths.dataset_dir / "dataset.jsonl",)
    assert not (result.paths.serve_root / "latest").exists()


@pytest.mark.parametrize("failure_point", ["persist_runtime_result", "set_latest_run"])
def test_failed_execution_or_publication_returns_no_result(
    copy_fixture, monkeypatch, failure_point
):
    root = copy_fixture("regression_project")
    request = build_runtime_run_request(
        "serve",
        str(root / "project.yaml"),
        command_observability=CommandObservability(visuals=False),
    )

    def fail(*_args, **_kwargs):
        raise OSError("output failed")

    module = (
        "execution" if failure_point == "persist_runtime_result" else "orchestration"
    )
    monkeypatch.setattr(f"jerrythomas.profiles.{module}.{failure_point}", fail)
    with pytest.raises(OSError, match="output failed"):
        run_profiles(request)
    assert not (root / "output" / "latest").exists()
