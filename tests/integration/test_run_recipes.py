import hashlib
import json
import shutil
import subprocess

import pytest

from jerrythomas.execution.settings import CommandObservability
from jerrythomas.io.runs import load_run
from jerrythomas.profiles.orchestration import run_profiles
from jerrythomas.profiles.request_builder import (
    build_materialize_run_request,
    build_runtime_run_request,
)


def _materialize(root, stream="metrics.linear", name="linear"):
    (root / "profiles" / f"materialize.{name}.yaml").write_text(
        f"stream: {stream}\noutput: exports/{name}.jsonl\n"
    )
    return build_materialize_run_request(
        str(root / "project.yaml"),
        profile_name=name,
        overwrite=None,
        output=None,
        artifact_mode=None,
        command_observability=CommandObservability(visuals=False),
    )


def test_materialize_recipe_preserves_resolved_external_config_not_unused_catalog(
    copy_fixture,
    tmp_path,
):
    root = copy_fixture("regression_project")
    shared = tmp_path / "shared"
    shared.mkdir()
    shutil.move(root / "sources" / "metrics.linear.yaml", shared)
    project = root / "project.yaml"
    project.write_text(
        project.read_text()
        .replace("sources: sources", "sources: [sources, ../shared]")
        .replace("globals:", "globals:\n  unused_setting: do-not-preserve")
    )
    request = _materialize(root)
    # The execution request already owns resolved configuration. Later edits
    # must not leak into either execution or its saved recipe.
    (shared / "metrics.linear.yaml").write_text("now: invalid")
    (root / "streams" / "metrics.linear.yaml").write_text("now: invalid")
    project.write_text("now: invalid")
    (saved,) = run_profiles(request)
    recipe = saved.load_recipe()
    config = recipe.configuration
    assert set(config["sources"]) == {"regression.linear"}
    assert set(config["streams"]) == {"metrics.linear"}
    assert config["dataset"] is None
    assert config["artifacts"] == {}
    assert config["streams"]["metrics.linear"]["preprocess"][0]["comparand"].startswith(
        "2024-03-01"
    )
    assert config["sources"]["regression.linear"]["loader"]["path"] == str(
        root / "data" / "linear_hourly.jsonl"
    )
    assert "do-not-preserve" not in saved.recipe_path.read_text()
    assert "${" not in saved.recipe_path.read_text()
    assert saved.output("linear").row_count == 6
    archive = tmp_path / "archive"
    archive.mkdir()
    for path in (saved.metadata_path, saved.recipe_path, *saved.outputs):
        shutil.copy(path, archive)
    shutil.rmtree(root)
    shutil.rmtree(shared)
    assert load_run(archive / saved.metadata_path.name).load_recipe() == recipe


def test_serve_recipe_includes_overrides_and_selected_artifact_identities(copy_fixture):
    root = copy_fixture("regression_project")
    request = build_runtime_run_request(
        "serve",
        str(root / "project.yaml"),
        limit=2,
        command_observability=CommandObservability(visuals=False),
    )
    (saved,) = run_profiles(request)
    recipe = saved.load_recipe()
    assert recipe.configuration["jobs"][0]["limit"] == 2
    assert recipe.configuration["dataset"] == request.definition.dataset.model_dump(
        mode="json"
    )
    assert set(recipe.artifacts) == {"series", "metadata", "scaler"}
    for key, artifact in recipe.artifacts.items():
        assert artifact[
            "expected_config_hash"
        ] == request.definition.artifact_hashes.for_artifact(key)
        assert artifact["identity"]["artifact_hash"] == artifact["expected_config_hash"]
        assert artifact["identity"]["files"][0]["sha256"]
        assert artifact["producer_recipe"] is None
    output = saved.output("dataset")
    assert (
        output.sha256
        == hashlib.sha256(saved.output_path("dataset").read_bytes()).hexdigest()
    )
    assert output.size_bytes == saved.output_path("dataset").stat().st_size
    assert (
        saved.metadata.recipe.sha256
        == hashlib.sha256(saved.recipe_path.read_bytes()).hexdigest()
    )
    assert "jerry-thomas" in recipe.implementation["packages"]


def test_shared_serve_root_saves_only_its_own_jobs(copy_fixture):
    root = copy_fixture("regression_project")
    for name, directory in (("second", "output"), ("third", "elsewhere")):
        (root / "profiles" / f"serve.{name}.yaml").write_text(
            f"operation: dataset\noutput: {{transport: fs, format: jsonl, directory: {directory}}}\n"
        )
    results = run_profiles(
        build_runtime_run_request("serve", str(root / "project.yaml"))
    )
    assert len(results) == 2
    for saved in results:
        assert {
            job["profile"] for job in saved.load_recipe().configuration["jobs"]
        } == {output.profile for output in saved.metadata.outputs}


def test_recipe_checksum_detects_tampering(copy_fixture):
    root = copy_fixture("regression_project")
    (saved,) = run_profiles(_materialize(root))
    saved.recipe_path.write_text("{}")
    with pytest.raises(ValueError, match="checksum mismatch"):
        saved.load_recipe()


def test_recipe_saves_the_null_argument_actually_passed_to_a_plugin(copy_fixture):
    root = copy_fixture("regression_project")
    project = root / "project.yaml"
    project.write_text(
        project.read_text().replace("globals:", "globals:\n  frequency: null")
    )
    (root / "sources" / "metrics.linear.yaml").write_text(
        "id: regression.linear\nparser: {entrypoint: core.temporal_record}\n"
        "freshness: opaque\nloader:\n  entrypoint: core.synthetic.ticks\n"
        "  args: {start: '${start_time}', end: '${end_time}', frequency: '${frequency}'}\n"
    )
    (saved,) = run_profiles(_materialize(root))
    loader = saved.load_recipe().configuration["sources"]["regression.linear"]["loader"]
    assert loader["args"]["frequency"] is None
    assert saved.output("linear").row_count == 6


def test_git_unavailable_does_not_block_recipe_or_execution(copy_fixture, monkeypatch):
    def unavailable(*args, **kwargs):
        raise FileNotFoundError("no git")

    monkeypatch.setattr("jerrythomas.profiles.recipes.subprocess.run", unavailable)
    root = copy_fixture("regression_project")
    (saved,) = run_profiles(_materialize(root))
    assert all(
        value is None
        for value in saved.load_recipe().implementation["repositories"].values()
    )


@pytest.mark.skipif(shutil.which("git") is None, reason="optional Git is unavailable")
def test_git_records_commit_and_dirty_state_without_requiring_a_clean_tree(
    copy_fixture,
):
    root = copy_fixture("regression_project")
    request = _materialize(root)

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    git("init", "-q")
    git("add", ".")
    git(
        "-c",
        "user.name=Jerry tests",
        "-c",
        "user.email=jerry@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "-c",
        "core.hooksPath=/dev/null",
        "commit",
        "-qm",
        "Test fixture",
    )
    commit = git("rev-parse", "HEAD")
    (root / "unrelated.txt").write_text("uncommitted exploration")
    (saved,) = run_profiles(request)
    identity = saved.load_recipe().implementation["repositories"][str(root)]
    assert identity == {"root": str(root), "commit": commit, "dirty": True}


def test_changed_input_does_not_publish_successful_run(copy_fixture, monkeypatch):
    from jerrythomas.profiles import materialize

    root = copy_fixture("regression_project")
    request = _materialize(root)
    execute = materialize.materialize_stream

    def change_input(**kwargs):
        output = execute(**kwargs)
        with (root / "data" / "linear_hourly.jsonl").open("a") as source:
            source.write("\n")
        return output

    monkeypatch.setattr(materialize, "materialize_stream", change_input)
    with pytest.raises(RuntimeError, match="inputs changed"):
        run_profiles(request)
    receipt = root / "exports" / "linear.jsonl.run.json"
    assert json.loads(receipt.read_text())["status"] == "failed"
    assert receipt.with_name("linear.jsonl.recipe.json").exists()


def test_materialized_input_records_upstream_receipt_reference(copy_fixture):
    root = copy_fixture("regression_project")
    (upstream,) = run_profiles(_materialize(root))
    (root / "sources" / "metrics.linear.yaml").write_text(
        "id: regression.linear\nparser: {entrypoint: core.temporal_record}\n"
        "loader: {transport: fs, path: exports/linear.jsonl, reader: {format: jsonl}}\n"
    )
    (downstream,) = run_profiles(_materialize(root, name="second"))
    snapshot = downstream.load_recipe().inputs["regression.linear"]
    (file,) = next(iter(snapshot["patterns"].values()))
    assert file["upstream_receipt"]["path"] == str(upstream.metadata_path)
    assert (
        file["upstream_receipt"]["sha256"]
        == hashlib.sha256(upstream.metadata_path.read_bytes()).hexdigest()
    )


def test_materialize_batch_captures_inputs_after_previous_job_finishes(copy_fixture):
    root = copy_fixture("regression_project")
    _materialize(root)
    (root / "profiles" / "materialize.linear.yaml").write_text(
        "stream: metrics.linear\norder: 1\noutput: exports/linear.jsonl\n"
    )
    (root / "profiles" / "materialize.second.yaml").write_text(
        "stream: imported\norder: 2\noutput: exports/second.jsonl\n"
    )
    (root / "sources" / "imported.yaml").write_text(
        "id: imported\nparser: {entrypoint: core.temporal_record}\n"
        "loader: {transport: fs, path: exports/linear.jsonl, reader: {format: jsonl}}\n"
    )
    (root / "streams" / "imported.yaml").write_text(
        "id: imported\nfrom: {source: imported}\nmap: {entrypoint: identity}\n"
    )
    request = build_materialize_run_request(
        str(root / "project.yaml"),
        profile_name=None,
        overwrite=None,
        output=None,
        artifact_mode=None,
        command_observability=CommandObservability(visuals=False),
    )
    first, second = run_profiles(request)
    assert second.output("second").sha256 == first.output("linear").sha256
    patterns = second.load_recipe().inputs["imported"]["patterns"]
    (file,) = next(iter(patterns.values()))
    assert file["upstream_receipt"]["path"] == str(first.metadata_path)


@pytest.mark.parametrize("command", ["serve", "materialize"])
def test_recipe_write_failure_never_publishes_success(
    copy_fixture, monkeypatch, command
):
    root = copy_fixture("regression_project")
    request = (
        _materialize(root)
        if command == "materialize"
        else build_runtime_run_request("serve", str(root / "project.yaml"))
    )

    def fail(*args, **kwargs):
        raise OSError("recipe write failed")

    monkeypatch.setattr("jerrythomas.io.runs.save_recipe", fail)
    with pytest.raises(OSError, match="recipe write failed"):
        run_profiles(request)
    receipt = (
        (root / "exports" / "linear.jsonl.run.json")
        if command == "materialize"
        else request.serve_run_plans[0].paths.metadata_path
    )
    with pytest.raises(ValueError, match="successfully finished"):
        load_run(receipt)
