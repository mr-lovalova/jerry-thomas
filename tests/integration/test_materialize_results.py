import gzip
import json
import os
import shutil
from datetime import datetime

import pytest

from jerrythomas.cli.command_router import execute_command
from jerrythomas.cli.parser_builder import build_parser
from jerrythomas.execution.settings import CommandObservability
from jerrythomas.profiles.errors import ProfileCommandError
from jerrythomas.io.runs import SavedRun, load_run
from jerrythomas.services.execution_lock import output_lock_path
from jerrythomas.profiles.orchestration import run_profiles
from jerrythomas.profiles.request_builder import build_materialize_run_request


def _profiles(root):
    (root / "operations" / "raw-linear.yaml").write_text(
        "kind: output\nentrypoint: core.records\nstream: metrics.linear\n"
    )
    for order, name, suffix in [(2, "first", ".jsonl"), (1, "second", ".jsonl.gz")]:
        (root / "profiles" / f"materialize.{name}.yaml").write_text(
            f"operation: raw-linear\norder: {order}\noutput: exports/{name}{suffix}\n"
        )


def _execute(root, *extra, result_json=True):
    args = build_parser().parse_args(
        [
            "materialize",
            "--project",
            str(root / "project.yaml"),
            "--no-visuals",
            *(["--result-json"] if result_json else []),
            *extra,
        ]
    )
    execute_command(
        args=args,
        plugin_root=None,
        workspace_context=None,
        cli_level_arg=None,
        cli_log_outputs=[],
    )


def test_materialize_returns_completed_invocation_in_profile_order(copy_fixture):
    root = copy_fixture("regression_project")
    _profiles(root)
    request = build_materialize_run_request(
        str(root / "project.yaml"),
        profile_name=None,
        overwrite=None,
        output=None,
        artifact_mode=None,
        command_observability=CommandObservability(visuals=False),
    )
    results = run_profiles(request)
    assert len(results) == 2
    assert results[0].metadata.run_id == results[1].metadata.run_id
    assert [result.metadata.outputs[0].profile for result in results] == [
        "second",
        "first",
    ]
    for result in results:
        assert isinstance(result, SavedRun)
        assert load_run(result.metadata_path) == result
        assert datetime.fromisoformat(
            result.metadata.started_at
        ) <= datetime.fromisoformat(result.metadata.finished_at)
        (output,) = result.metadata.outputs
        opener = gzip.open if output.compression else open
        with opener(result.output_path(output.profile), "rt") as rows:
            count = sum(1 for _row in rows)
        assert output.row_count == count > 0
        assert output.stream == "metrics.linear"
        assert output.format == "jsonl"
        assert output.view == "raw"
        assert output.encoding == "utf-8"
    assert set((root / "exports").iterdir()) == {
        path
        for result in results
        for path in (
            result.outputs[0],
            result.metadata_path,
            result.recipe_path,
            output_lock_path(result.outputs[0]),
        )
    }


def test_materialize_cli_result_json(copy_fixture, capsys):
    root = copy_fixture("regression_project")
    _profiles(root)
    _execute(root)

    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == 3
    assert len(payload["runs"]) == 2
    for run in payload["runs"]:
        receipt = run.pop("receipt")
        saved = load_run(receipt)
        assert run == saved.metadata.model_dump(mode="json")
        assert run["command"] == "materialize"
        assert run["status"] == "success"
    assert payload["runs"][0]["outputs"][0]["path"] == "second.jsonl.gz"


def test_materialize_cli_empty_selection(copy_fixture, capsys):
    root = copy_fixture("regression_project")
    (root / "operations" / "raw-linear.yaml").write_text(
        "kind: output\nentrypoint: core.records\nstream: metrics.linear\n"
    )
    (root / "profiles" / "materialize.disabled.yaml").write_text(
        "operation: raw-linear\noutput: exports/disabled.jsonl\nenabled: false\n"
    )
    _execute(root)
    assert json.loads(capsys.readouterr().out) == {"schema_version": 3, "runs": []}


@pytest.mark.parametrize(
    "config", ["materialize.defaults.yaml", "materialize.first.yaml"]
)
def test_materialize_cli_rejects_stdout_logs_before_writing(
    copy_fixture, capsys, config
):
    root = copy_fixture("regression_project")
    _profiles(root)
    path = root / "profiles" / config
    with path.open("a") as f:
        f.write("observability:\n  logging:\n    outputs: [{transport: stdout}]\n")

    with pytest.raises(ProfileCommandError, match="cannot use stdout logging"):
        _execute(root)
    assert capsys.readouterr().out == ""
    assert not (root / "exports").exists()


def test_materialize_cli_does_not_report_success_after_partial_failure(
    copy_fixture, capsys, monkeypatch
):
    from jerrythomas.profiles import orchestration

    root = copy_fixture("regression_project")
    _profiles(root)
    execute = orchestration.execute_materialize_job

    def fail_second(job, runtime, **kwargs):
        if job.name == "first":
            raise OSError("write failed")
        return execute(job, runtime, **kwargs)

    monkeypatch.setattr(orchestration, "execute_materialize_job", fail_second)
    with pytest.raises(OSError, match="write failed"):
        _execute(root)
    assert capsys.readouterr().out == ""
    assert (root / "exports" / "second.jsonl.gz").is_file()
    assert not (root / "exports" / "first.jsonl").exists()
    saved = load_run(root / "exports" / "second.jsonl.gz.run.json")
    assert saved.metadata.status == "success"


def test_materialize_saves_receipt_without_result_json(copy_fixture, capsys, tmp_path):
    root = copy_fixture("regression_project")
    _profiles(root)
    _execute(root, "--profile", "first", result_json=False)
    assert capsys.readouterr().out == ""
    receipt = root / "exports" / "first.jsonl.run.json"
    saved = load_run(receipt)
    archive = tmp_path / "copied"
    archive.mkdir()
    shutil.copy(receipt, archive)
    shutil.copy(saved.recipe_path, archive)
    shutil.copy(saved.outputs[0], archive)
    shutil.rmtree(root)
    copied = load_run(archive / receipt.name)
    assert copied.output_path("first").read_bytes()
    assert copied.metadata == saved.metadata
    assert copied.load_recipe().command == "materialize"


def test_materialize_empty_output_has_successful_zero_row_receipt(
    copy_fixture, monkeypatch
):
    root = copy_fixture("regression_project")
    _profiles(root)
    monkeypatch.setattr(
        "jerrythomas.operations.runtime.records.run_stream_pipeline",
        lambda *_args: iter(()),
    )
    _execute(root, "--profile", "first", result_json=False)
    saved = load_run(root / "exports" / "first.jsonl.run.json")
    assert saved.output("first").row_count == 0
    assert saved.output_path("first").read_bytes() == b""


def test_materialize_overwrite_replaces_receipt_and_invalidates_loaded_result(
    copy_fixture, capsys
):
    root = copy_fixture("regression_project")
    _profiles(root)
    _execute(root, "--profile", "first")
    receipt = root / "exports" / "first.jsonl.run.json"
    previous = load_run(receipt)
    _execute(root, "--profile", "first", "--overwrite")
    current = load_run(receipt)
    assert current.metadata.run_id != previous.metadata.run_id
    assert current.output_path("first").is_file()
    with pytest.raises(ValueError, match="receipt changed"):
        previous.output_path("first")


@pytest.mark.parametrize("failure", ["start", "data", "finish"])
def test_materialize_overwrite_failure_never_leaves_old_success_with_new_data(
    copy_fixture, capsys, monkeypatch, failure
):
    from jerrythomas.profiles import materialize

    root = copy_fixture("regression_project")
    _profiles(root)
    _execute(root, "--profile", "first")
    capsys.readouterr()
    output = root / "exports" / "first.jsonl"
    receipt = output.with_name(output.name + ".run.json")
    old_data, old_receipt = output.read_bytes(), receipt.read_bytes()
    function = {
        "start": "start_run",
        "data": "materialize_stream",
        "finish": "finish_run_success",
    }[failure]

    def fail(*_args, **_kwargs):
        raise OSError("publication failed")

    monkeypatch.setattr(materialize, function, fail)
    with pytest.raises(OSError, match="publication failed"):
        _execute(root, "--profile", "first", "--overwrite")
    assert capsys.readouterr().out == ""
    if failure == "start":
        assert receipt.read_bytes() == old_receipt
        assert output.read_bytes() == old_data
    else:
        assert json.loads(receipt.read_text())["status"] == "failed"
        with pytest.raises(ValueError, match="successfully finished"):
            load_run(receipt)
        if failure == "data":
            assert output.read_bytes() == old_data


def test_materialize_no_overwrite_preserves_data_and_receipt(copy_fixture, capsys):
    root = copy_fixture("regression_project")
    _profiles(root)
    _execute(root, "--profile", "first")
    capsys.readouterr()
    output = root / "exports" / "first.jsonl"
    receipt = output.with_name(output.name + ".run.json")
    original = (output.read_bytes(), receipt.read_bytes())
    with pytest.raises(SystemExit) as error:
        _execute(root, "--profile", "first")
    assert error.value.code == 2
    assert (output.read_bytes(), receipt.read_bytes()) == original
    assert capsys.readouterr().out == ""


def test_materialize_receipt_commit_failure_leaves_running_receipt(
    copy_fixture, capsys, monkeypatch
):
    root = copy_fixture("regression_project")
    _profiles(root)
    _execute(root, "--profile", "first")
    capsys.readouterr()
    output = root / "exports" / "first.jsonl"
    receipt = output.with_name(output.name + ".run.json")
    previous = load_run(receipt)
    replace = os.replace

    def reject_final_receipt(source, destination):
        if (
            destination == receipt
            and json.loads(source.read_text())["status"] != "running"
        ):
            raise OSError("receipt commit failed")
        return replace(source, destination)

    monkeypatch.setattr("jerrythomas.io.sinks.files.os.replace", reject_final_receipt)
    with pytest.raises(OSError, match="receipt commit failed"):
        _execute(root, "--profile", "first", "--overwrite")
    assert capsys.readouterr().out == ""
    assert output.is_file()
    assert json.loads(receipt.read_text())["status"] == "running"
    with pytest.raises(ValueError, match="successfully finished"):
        load_run(receipt)
    with pytest.raises(ValueError, match="receipt changed"):
        previous.output_path("first")


@pytest.mark.parametrize("reserved", ["first.jsonl.run.json", ".first.jsonl.lock"])
def test_materialize_rejects_log_conflicts_with_receipt_and_lock(
    copy_fixture, reserved
):
    root = copy_fixture("regression_project")
    _profiles(root)
    with (root / "profiles" / "materialize.defaults.yaml").open("w") as f:
        f.write(
            f"observability:\n  logging:\n    outputs: [{{transport: fs, path: exports/{reserved}}}]\n"
        )
    with pytest.raises(ProfileCommandError, match="same path"):
        _execute(root)
    assert not (root / "exports").exists()


@pytest.mark.parametrize("reserved", ["first.jsonl.run.json", ".first.jsonl.lock"])
def test_materialize_rejects_symlink_receipts_and_locks_before_writing(
    copy_fixture, tmp_path, reserved
):
    root = copy_fixture("regression_project")
    _profiles(root)
    outside = tmp_path / "outside"
    outside.write_text("keep")
    exports = root / "exports"
    exports.mkdir()
    (exports / reserved).symlink_to(outside)
    with pytest.raises(SystemExit) as error:
        _execute(root, "--overwrite")
    assert error.value.code == 2
    assert outside.read_text() == "keep"
    assert not (exports / "second.jsonl.gz").exists()
