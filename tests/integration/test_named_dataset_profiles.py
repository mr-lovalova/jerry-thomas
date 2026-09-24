import json

import pytest
import yaml

import jerrythomas.profiles.request_builder as request_builder
from jerrythomas.artifacts.scaler import load_scaler_artifact
from jerrythomas.execution.settings import CommandObservability
from jerrythomas.profiles.errors import ProfileCommandError
from jerrythomas.profiles.orchestration import run_profiles
from jerrythomas.profiles.request_builder import (
    build_build_run_request,
    build_materialize_run_request,
    build_runtime_run_request,
)
from jerrythomas.services.saved_runs import load_dataset_run

from tests.named_dataset_helpers import (
    create_project as _project,
    write_config as _write,
)


QUIET = CommandObservability(visuals=False, log_level="CRITICAL")


def test_profiles_bind_distinct_datasets_from_one_definition_snapshot(
    tmp_path, monkeypatch
):
    project = _project(tmp_path)
    calls = []
    load = request_builder.load_project_definition

    def counted(path):
        calls.append(path)
        return load(path)

    monkeypatch.setattr(request_builder, "load_project_definition", counted)
    request = build_runtime_run_request(
        "serve", str(project), command_observability=QUIET
    )
    assert request is not None
    assert len(calls) == 1
    assert [job.runtime.dataset_id for job in request.jobs] == ["alpha", "beta"]
    # The request owns the resolved configuration; editing a YAML file afterward
    # must not silently select a different dataset during the artifact phase.
    _write(tmp_path / "datasets" / "beta.yaml", {"changed_after_request": True})
    runs = run_profiles(request)
    assert len(calls) == 1
    assert {
        (run.metadata.dataset_id, run.metadata.dataset_version) for run in runs
    } == {("alpha", "v1"), ("beta", "v2")}
    alpha, beta = runs
    assert alpha.metadata.split is not None
    assert beta.metadata.split is None
    assert [output.output_id for output in alpha.metadata.outputs] == [
        "holdout.train",
        "holdout.test",
    ]
    assert [output.output_id for output in beta.metadata.outputs] == [None]
    for run in runs:
        recipe = run.load_recipe()
        assert recipe.dataset_id == run.metadata.dataset_id
        assert recipe.dataset_version == run.metadata.dataset_version
        assert set(recipe.configuration["datasets"]) == {run.metadata.dataset_id}
        assert all(
            run.metadata.dataset_id in key for key in recipe.configuration["artifacts"]
        )
    beta_rows = [json.loads(line) for line in beta.outputs[0].read_text().splitlines()]
    assert [row["features"]["values"]["x"] for row in beta_rows] == pytest.approx(
        [-1.224744871391589, 0, 1.224744871391589]
    )


def test_different_datasets_cannot_share_one_saved_run_directory(tmp_path):
    project = _project(tmp_path)
    _write(
        tmp_path / "profiles" / "serve.defaults.yaml",
        {
            "output": {
                "transport": "fs",
                "format": "jsonl",
                "directory": "outputs/shared",
            },
        },
    )
    with pytest.raises(ProfileCommandError, match="same dataset"):
        build_runtime_run_request("serve", str(project), command_observability=QUIET)
    assert not (tmp_path / "outputs").exists()


def test_build_profiles_fit_only_their_bound_dataset(tmp_path):
    project = _project(tmp_path)
    request = build_build_run_request(
        str(project), profile_name="beta", command_observability=QUIET
    )
    assert request is not None
    run_profiles(request)
    assert not (tmp_path / "artifacts" / "datasets" / "alpha").exists()
    artifact = load_scaler_artifact(
        tmp_path / "artifacts" / "datasets" / "beta" / "scaler.json"
    )
    assert artifact.scalers["x"].statistics.mean == 20.0


def test_stream_operations_serve_and_materialize_without_any_dataset(tmp_path):
    project = _project(tmp_path, datasets=False)
    _write(
        tmp_path / "profiles" / "serve.raw.yaml",
        {
            "operation": "raw",
            "limit": 1,
            "output": {
                "transport": "fs",
                "format": "jsonl",
                "directory": "stream-runs",
            },
        },
    )
    serve = build_runtime_run_request(
        "serve", str(project), command_observability=QUIET
    )
    assert serve is not None
    (run,) = run_profiles(serve)
    assert run.metadata.dataset_id is None
    assert run.metadata.outputs[0].stream == "rows"
    assert run.metadata.outputs[0].row_count == 1
    request = build_materialize_run_request(str(project), None, None, None, None, QUIET)
    assert request is not None
    (saved,) = run_profiles(request)
    assert saved.metadata.dataset_id is None
    assert saved.metadata.outputs[0].operation == "raw"
    assert saved.metadata.outputs[0].stream == "rows"
    assert saved.metadata.outputs[0].row_count == 3
    assert saved.load_recipe().configuration["datasets"] == {}


def test_materialize_rejects_dataset_operations(tmp_path):
    project = _project(tmp_path)
    _write(
        tmp_path / "profiles" / "materialize.raw.yaml",
        {
            "operation": "serve-alpha",
            "output": "exports/rows.jsonl",
        },
    )
    with pytest.raises(ProfileCommandError, match="must reference a stream operation"):
        build_materialize_run_request(str(project), None, None, None, None, QUIET)
    assert not (tmp_path / "exports").exists()


def test_dataset_run_lookup_uses_requested_archived_version_and_checks_identity(
    tmp_path,
):
    project = _project(tmp_path)
    request = build_runtime_run_request(
        "serve", str(project), profile_name="beta", command_observability=QUIET
    )
    assert request is not None
    (saved,) = run_profiles(request)
    path = tmp_path / "datasets" / "beta.yaml"
    dataset = yaml.safe_load(path.read_text())
    dataset["version"] = "v3"
    _write(path, dataset)
    found = load_dataset_run(project, "beta", "v2", saved.metadata.run_id)
    assert found.metadata_path == saved.metadata_path
    with pytest.raises(FileNotFoundError, match="No saved run"):
        load_dataset_run(project, "beta", "v3", saved.metadata.run_id)
    # A destination override cannot cause a beta receipt to masquerade as alpha.
    _write(
        tmp_path / "profiles" / "serve.alpha.yaml",
        {
            "operation": "serve-alpha",
            "output": {
                "transport": "fs",
                "format": "jsonl",
                "directory": "outputs/beta/v2",
            },
        },
    )
    with pytest.raises(ValueError, match="identity does not match"):
        load_dataset_run(project, "alpha", "v1", saved.metadata.run_id)


def test_materialize_rejects_explicitly_required_inactive_artifact(tmp_path):
    project = _project(tmp_path)
    dataset_path = tmp_path / "datasets" / "beta.yaml"
    dataset = yaml.safe_load(dataset_path.read_text())
    dataset["features"][0]["scale"] = False
    _write(dataset_path, dataset)
    _write(
        tmp_path / "operations" / "raw.yaml",
        {
            "kind": "runtime",
            "entrypoint": "core.runtime.stream",
            "stream": "rows",
            "requires": ["dataset.beta.scaler"],
        },
    )
    request = build_materialize_run_request(str(project), None, None, None, None, QUIET)
    assert request is not None

    with pytest.raises(ProfileCommandError, match="inactive for this dataset"):
        run_profiles(request)
    assert not (tmp_path / "exports").exists()
