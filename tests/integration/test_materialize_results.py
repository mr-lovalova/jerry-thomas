import gzip
import json
from datetime import datetime

import pytest

from jerrythomas.cli.command_router import execute_command
from jerrythomas.cli.parser_builder import build_parser
from jerrythomas.execution.settings import CommandObservability
from jerrythomas.profiles.errors import ProfileCommandError
from jerrythomas.profiles.models import MaterializeRunResult
from jerrythomas.profiles.orchestration import run_profiles
from jerrythomas.profiles.request_builder import build_materialize_run_request


def _profiles(root):
    for order, name, suffix in [(2, "first", ".jsonl"), (1, "second", ".jsonl.gz")]:
        (root / "profiles" / f"materialize.{name}.yaml").write_text(
            f"stream: metrics.linear\norder: {order}\noutput: exports/{name}{suffix}\n"
        )


def _execute(root, *extra):
    args = build_parser().parse_args(
        [
            "materialize",
            "--project",
            str(root / "project.yaml"),
            "--no-visuals",
            "--result-json",
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
    (result,) = run_profiles(request)

    assert isinstance(result, MaterializeRunResult)
    assert result.run_id
    assert datetime.fromisoformat(result.started_at) <= datetime.fromisoformat(
        result.finished_at
    )
    assert [output.profile for output in result.outputs] == ["second", "first"]
    for output in result.outputs:
        opener = gzip.open if output.compression else open
        with opener(output.path, "rt") as rows:
            count = sum(1 for _row in rows)
        assert output.row_count == count > 0
        assert output.stream == "metrics.linear"
        assert output.format == "jsonl"
        assert output.view == "raw"
        assert output.encoding == "utf-8"
        assert output.path.is_absolute()
    assert set((root / "exports").iterdir()) == {
        output.path for output in result.outputs
    }


def test_materialize_cli_result_json(copy_fixture, capsys):
    root = copy_fixture("regression_project")
    _profiles(root)
    _execute(root)

    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == 1
    (run,) = payload["runs"]
    assert run["command"] == "materialize"
    assert run["status"] == "success"
    assert run["run_id"]
    assert run["started_at"] <= run["finished_at"]
    assert run["outputs"][0] == {
        "profile": "second",
        "stream": "metrics.linear",
        "path": str(root / "exports" / "second.jsonl.gz"),
        "format": "jsonl",
        "view": "raw",
        "encoding": "utf-8",
        "compression": "gzip",
        "row_count": run["outputs"][1]["row_count"],
    }


def test_materialize_cli_empty_selection(copy_fixture, capsys):
    root = copy_fixture("regression_project")
    (root / "profiles" / "materialize.disabled.yaml").write_text(
        "stream: metrics.linear\noutput: exports/disabled.jsonl\nenabled: false\n"
    )
    _execute(root)
    assert json.loads(capsys.readouterr().out) == {"schema_version": 1, "runs": []}


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

    def fail_second(job, runtime):
        if job.name == "first":
            raise OSError("write failed")
        return execute(job, runtime)

    monkeypatch.setattr(orchestration, "execute_materialize_job", fail_second)
    with pytest.raises(OSError, match="write failed"):
        _execute(root)
    assert capsys.readouterr().out == ""
    assert (root / "exports" / "second.jsonl.gz").is_file()
    assert not (root / "exports" / "first.jsonl").exists()
