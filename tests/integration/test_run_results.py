import gzip
import json
import os
import shutil
from pathlib import Path

import pytest

from jerrythomas.config.profiles.output import ServeOutputConfig
from jerrythomas.execution.settings import CommandObservability
from jerrythomas.io.runs import load_run
from jerrythomas.profiles.orchestration import run_profiles
from jerrythomas.profiles.request_builder import build_runtime_run_request


@pytest.mark.parametrize("fixture", ["regression_project", "walk_forward_project"])
@pytest.mark.parametrize("limit", [None, 1])
@pytest.mark.parametrize(
    ("fmt", "compression", "suffix"),
    [
        ("jsonl", None, ".jsonl"),
        ("jsonl", "gzip", ".jsonl.gz"),
        ("parquet", None, ".parquet"),
    ],
)
def test_run_results_identify_committed_files(
    copy_fixture, fixture, fmt, compression, suffix, limit
):
    root = copy_fixture(fixture)
    request = build_runtime_run_request(
        "serve",
        str(root / "project.yaml"),
        profile_name="dataset",
        limit=limit,
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
    saved = load_run(result.paths.run_root)
    assert saved.metadata.schema_version == 1
    assert saved.metadata.split == request.definition.dataset.split
    assert len(saved.metadata.outputs) == len(result.outputs)
    for output, path in zip(saved.metadata.outputs, result.outputs):
        assert output.profile == "dataset"
        assert output.operation == "dataset"
        assert output.format == fmt
        assert output.compression == compression
        assert output.view == ("flat" if fmt == "parquet" else "raw")
        assert saved.output_path("dataset", output.output_id) == path
        if fmt == "parquet":
            import pyarrow.parquet as pq

            count = pq.read_metadata(path).num_rows
        else:
            opener = gzip.open if compression else open
            with opener(path, "rt") as rows:
                count = sum(1 for _row in rows)
        assert output.row_count == count
        if limit is not None:
            assert count <= limit
        if request.definition.dataset.split is None:
            assert output.fold is None
            assert output.output_id is None
        else:
            assert output.fold is not None
            assert output.output_id == f"{output.fold.id}.{output.fold.role}"
            configured_fold = next(
                fold
                for fold in request.definition.dataset.split.folds
                if fold.id == output.fold.id
            )
            assert output.fold.labels == tuple(
                getattr(configured_fold, output.fold.role)
            )


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
    for result in results:
        saved = load_run(result.paths.run_root)
        for output, path in zip(saved.metadata.outputs, result.outputs):
            assert saved.output_path(output.profile) == path
            assert output.profile == path.stem


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
    assert load_run(results[0].paths.run_root).metadata.outputs == ()


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
    saved = load_run(result.paths.run_root)
    assert saved.metadata.preview == "samples"
    assert saved.output("dataset").row_count == 1
    assert saved.output("dataset").fold is None


def test_saved_fold_selection_survives_project_removal(copy_fixture, tmp_path):
    root = copy_fixture("walk_forward_project")
    request = build_runtime_run_request(
        "serve",
        str(root / "project.yaml"),
        profile_name="dataset",
        command_observability=CommandObservability(visuals=False),
    )
    (result,) = run_profiles(request)
    archive = tmp_path / "archive"
    shutil.copytree(result.paths.run_root, archive)
    manifest = json.loads((archive / "run.json").read_text())
    train = next(
        output for output in manifest["outputs"] if output["fold"]["role"] == "train"
    )
    for output in manifest["outputs"]:
        if output["fold"]["role"] == "test":
            (archive / output["path"]).unlink()
    shutil.rmtree(root)

    saved = load_run(archive)
    assert saved.output_path("dataset", train["output_id"]) == archive / train["path"]
    assert saved.output("dataset", train["output_id"]).row_count > 0
    with pytest.raises(KeyError, match="Run has no output"):
        saved.output("dataset")


def test_saved_stream_previews_have_output_ids_without_fold_identity(copy_fixture):
    root = copy_fixture("walk_forward_project")
    request = build_runtime_run_request(
        "serve",
        str(root / "project.yaml"),
        profile_name="dataset",
        preview="records",
        limit=1,
        command_observability=CommandObservability(visuals=False),
    )
    (result,) = run_profiles(request)
    saved = load_run(result.paths.run_root)
    assert saved.metadata.preview == "records"
    assert saved.metadata.split == request.definition.dataset.split
    assert (
        tuple(output.output_id for output in saved.metadata.outputs)
        == request.jobs[0].output_ids
    )
    assert len(saved.metadata.outputs) > 1
    for output in saved.metadata.outputs:
        assert output.fold is None
        assert output.row_count == 1
        assert saved.output_path("dataset", output.output_id).is_file()


def test_saved_split_uses_execution_rules_after_project_changes(
    copy_fixture, monkeypatch
):
    from jerrythomas.pipelines.dataset.split import build_labeler
    from jerrythomas.profiles import orchestration
    from jerrythomas.services.project_definition import load_project_definition

    root = copy_fixture("walk_forward_project")
    project_path = root / "project.yaml"
    request = build_runtime_run_request(
        "serve",
        str(project_path),
        profile_name="dataset",
        command_observability=CommandObservability(visuals=False),
    )
    original_split = request.definition.dataset.split.model_dump(mode="json")
    execute = orchestration.execute_runtime_job

    def check_started_manifest(command, definition, plan):
        manifest = json.loads(plan.job.output.run.metadata_path.read_text())
        assert manifest["status"] == "running"
        assert manifest["split"] == original_split
        assert manifest["outputs"] == []
        return execute(command, definition, plan)

    monkeypatch.setattr(orchestration, "execute_runtime_job", check_started_manifest)
    (result,) = run_profiles(request)
    dataset_path = root / "dataset.yaml"
    dataset_path.write_text(
        dataset_path.read_text().replace("2024-01-03", "2024-01-02")
    )
    changed_split = load_project_definition(project_path).dataset.split
    saved = load_run(result.paths.run_root)
    assert saved.metadata.split.model_dump(mode="json") == original_split
    key = "2024-01-02T12:00:00Z"
    assert build_labeler(saved.metadata.split).label(key) == "train_0"
    assert build_labeler(changed_split).label(key) == "purge_0"


@pytest.mark.parametrize(
    "failure_point", ["persist_runtime_result", "finish_run_success", "set_latest_run"]
)
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


def test_failed_manifest_commit_preserves_previous_latest(copy_fixture, monkeypatch):
    root = copy_fixture("regression_project")
    previous_request = build_runtime_run_request(
        "serve",
        str(root / "project.yaml"),
        command_observability=CommandObservability(visuals=False),
    )
    (previous,) = run_profiles(previous_request)
    request = build_runtime_run_request(
        "serve",
        str(root / "project.yaml"),
        command_observability=CommandObservability(visuals=False),
    )
    paths = request.serve_run_plans[0].paths
    replace = os.replace

    def reject_completed_manifest(source, destination):
        if Path(destination) == paths.metadata_path:
            manifest = json.loads(Path(source).read_text(encoding="utf-8"))
            if manifest["status"] == "success":
                raise OSError("manifest commit failed")
        return replace(source, destination)

    monkeypatch.setattr(
        "jerrythomas.io.sinks.files.os.replace", reject_completed_manifest
    )
    with pytest.raises(OSError, match="manifest commit failed"):
        run_profiles(request)

    latest = load_run(paths.serve_root / "latest")
    assert latest.directory == previous.paths.run_root
    assert latest.output_path("dataset") == previous.outputs[0]
    with pytest.raises(ValueError, match="successfully finished"):
        load_run(paths.run_root)
    manifest = json.loads(paths.metadata_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "running"
    assert manifest["outputs"] == []
    assert set(paths.run_root.iterdir()) == {paths.dataset_dir, paths.metadata_path}
