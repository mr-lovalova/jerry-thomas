from pathlib import Path

import pytest
import yaml

from jerrythomas.artifacts.registry import SERIES_SPEC
from jerrythomas.config.dataset.dataset import DatasetConfig, SampleConfig
from jerrythomas.config.profiles.materialize import MaterializeProfile
from jerrythomas.config.tasks.base import ArtifactTask, RuntimeTask
from jerrythomas.config.tasks.stream import StreamTask
from jerrythomas.profiles.errors import ProfileCommandError
from jerrythomas.profiles.request_builder import build_runtime_run_request
from jerrythomas.services.dataset import load_datasets
from jerrythomas.services.operations import (
    artifact_kind,
    operation_documents,
    operations_from_documents,
)
from jerrythomas.services.project import load_project
from jerrythomas.services.project_definition import load_project_definition
from jerrythomas.services.runtime_compiler import (
    compile_runtime,
    compile_runtime_snapshot,
    snapshot_runtime,
)


def _write_yaml(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _project(tmp_path: Path, datasets: tuple[str, ...] = ()) -> Path:
    for name in ("sources", "streams", "operations", "profiles"):
        (tmp_path / name).mkdir()
    paths = {
        "sources": "sources",
        "streams": "streams",
        "operations": "operations",
        "profiles": "profiles",
        "artifacts": "artifacts",
    }
    if datasets:
        paths["datasets"] = "datasets"
        for dataset_id in datasets:
            _write_yaml(
                tmp_path / "datasets" / f"{dataset_id}.yaml",
                {"version": "v1", "sample": {"cadence": "1d", "rounding": "exact"}},
            )
    path = tmp_path / "project.yaml"
    _write_yaml(path, {"schema_version": 7, "artifact_revision": 1, "paths": paths})
    return path


def _operations(project_yaml: Path):
    project = load_project(project_yaml)
    return operations_from_documents(
        project, operation_documents(project), load_datasets(project)
    )


def test_stream_only_project_needs_no_dataset_or_implicit_operations(tmp_path):
    definition = load_project_definition(_project(tmp_path))
    assert definition.datasets == {}
    assert definition.project.dataset_dirs == ()
    assert definition.runtime_operations == ()
    assert definition.artifact_graph.tasks_by_id == {}
    runtime = compile_runtime(definition)
    assert runtime.dataset_id is None
    assert runtime.dataset is None
    assert runtime.artifact_aliases == {}
    with pytest.raises(ValueError, match="selected dataset"):
        runtime.require_dataset()


def test_dataset_catalog_owns_identity_and_generates_separate_artifacts(tmp_path):
    definition = load_project_definition(_project(tmp_path, ("alpha", "beta")))
    assert set(definition.datasets) == {"alpha", "beta"}
    assert definition.runtime_operations == ()
    tasks = tuple(definition.artifact_graph.tasks_by_id.values())
    assert len(tasks) == 8
    assert len({task.output for task in tasks}) == 8
    for task in tasks:
        kind = artifact_kind(task)
        assert task.id == f"dataset.{task.dataset}.{kind}"
        assert task.output.startswith(f"datasets/{task.dataset}/")


def test_compiling_selected_datasets_is_explicit_and_keeps_snapshots_isolated(tmp_path):
    definition = load_project_definition(_project(tmp_path, ("alpha", "beta")))
    alpha = compile_runtime(definition, "alpha")
    beta = compile_runtime(definition, "beta")
    assert alpha.require_dataset() == definition.require_dataset("alpha")
    assert alpha.require_dataset() is not definition.require_dataset("alpha")
    assert alpha.require_dataset() is not beta.require_dataset()
    assert alpha.artifact_aliases["series"] == "dataset.alpha.series"
    assert beta.artifact_aliases["series"] == "dataset.beta.series"
    alpha.artifacts.register(
        "dataset.alpha.series", "datasets/alpha/series/manifest.json"
    )
    restored = compile_runtime_snapshot(snapshot_runtime(alpha, ()))
    assert restored.dataset_id == "alpha"
    assert restored.dataset == alpha.dataset
    assert restored.dataset is not alpha.dataset
    assert restored.artifact_aliases == alpha.artifact_aliases
    assert restored.artifact_aliases is not alpha.artifact_aliases
    assert restored.artifacts.resolve_path(SERIES_SPEC) == alpha.artifacts.resolve_path(
        SERIES_SPEC
    )
    with pytest.raises(ValueError, match="Unknown dataset 'missing'"):
        compile_runtime(definition, "missing")
    assert compile_runtime(definition).dataset is None


def test_stream_only_snapshot_stays_dataset_independent(tmp_path):
    runtime = compile_runtime(load_project_definition(_project(tmp_path)))
    restored = compile_runtime_snapshot(snapshot_runtime(runtime, ()))
    assert restored.dataset is None
    assert restored.dataset_id is None
    assert restored.artifact_aliases == {}


def test_explicit_core_producer_replaces_only_its_dataset_and_kind(tmp_path):
    project_yaml = _project(tmp_path, ("alpha", "beta"))
    _write_yaml(
        tmp_path / "operations" / "alpha-stats.yaml",
        {"kind": "artifact", "entrypoint": "core.artifact.scaler", "dataset": "alpha"},
    )
    operations = _operations(project_yaml)
    scalers = {
        task.dataset: task for task in operations if artifact_kind(task) == "scaler"
    }
    assert scalers["alpha"].id == "alpha-stats"
    assert scalers["alpha"].output == "datasets/alpha/scaler.json"
    assert scalers["beta"].id == "dataset.beta.scaler"
    assert len(operations) == 8


def test_multiple_core_producers_for_one_dataset_are_rejected(tmp_path):
    project_yaml = _project(tmp_path, ("alpha",))
    for name in ("first", "second"):
        _write_yaml(
            tmp_path / "operations" / f"{name}.yaml",
            {
                "kind": "artifact",
                "entrypoint": "core.artifact.series",
                "dataset": "alpha",
            },
        )
    with pytest.raises(ValueError, match="multiple series producers"):
        _operations(project_yaml)


@pytest.mark.parametrize(
    "kind,entrypoint",
    [
        ("runtime", "core.runtime.dataset"),
        ("runtime", "core.runtime.matrix"),
        ("runtime", "core.runtime.coverage"),
        ("artifact", "core.artifact.scaler"),
        ("artifact", "core.artifact.series"),
        ("artifact", "core.artifact.metadata"),
        ("artifact", "core.artifact.coverage_stats"),
    ],
)
def test_dataset_operations_require_explicit_known_bindings(tmp_path, kind, entrypoint):
    project_yaml = _project(tmp_path, ("alpha",))
    operation_path = tmp_path / "operations" / "selected.yaml"
    operation = {"kind": kind, "entrypoint": entrypoint}
    _write_yaml(operation_path, operation)
    with pytest.raises(ValueError, match="must bind a dataset"):
        _operations(project_yaml)
    _write_yaml(operation_path, {**operation, "dataset": "missing"})
    with pytest.raises(ValueError, match="unknown dataset 'missing'"):
        _operations(project_yaml)


def test_runtime_operations_are_explicit_and_stream_operations_have_no_dataset(
    tmp_path,
):
    project_yaml = _project(tmp_path, ("alpha",))
    _write_yaml(
        tmp_path / "operations" / "samples.yaml",
        {"kind": "runtime", "entrypoint": "core.runtime.dataset", "dataset": "alpha"},
    )
    _write_yaml(
        tmp_path / "operations" / "prices.yaml",
        {"kind": "runtime", "entrypoint": "core.runtime.stream", "stream": "prices"},
    )
    runtime_tasks = [
        task for task in _operations(project_yaml) if isinstance(task, RuntimeTask)
    ]
    assert {task.id for task in runtime_tasks} == {"samples", "prices"}
    stream = next(task for task in runtime_tasks if isinstance(task, StreamTask))
    assert stream.stream == "prices"
    assert stream.dataset is None
    with pytest.raises(ValueError, match="must not bind a dataset"):
        StreamTask(id="invalid", stream="prices", dataset="alpha")


def test_scaling_policy_belongs_to_dataset_not_artifact_operation(tmp_path):
    project_yaml = _project(tmp_path, ("alpha",))
    _write_yaml(
        tmp_path / "datasets" / "alpha.yaml",
        {
            "version": "v2",
            "sample": {"cadence": "1d", "rounding": "exact"},
            "scaling": {"with_mean": False, "with_std": True, "epsilon": 0.5},
        },
    )
    dataset = load_datasets(load_project(project_yaml))["alpha"]
    assert dataset.scaling.with_mean is False
    assert dataset.scaling.epsilon == 0.5
    _write_yaml(
        tmp_path / "operations" / "scaler.yaml",
        {
            "kind": "artifact",
            "entrypoint": "core.artifact.scaler",
            "dataset": "alpha",
            "with_mean": False,
        },
    )
    with pytest.raises(ValueError, match="with_mean"):
        _operations(project_yaml)


def test_shared_artifacts_are_preserved_without_dataset_binding(tmp_path):
    project_yaml = _project(tmp_path)
    _write_yaml(
        tmp_path / "operations" / "calendar.yaml",
        {
            "kind": "artifact",
            "entrypoint": "core.artifact.schedule",
            "stream": "sessions",
            "partition_by": [],
            "output": "shared/calendar.jsonl",
        },
    )
    _write_yaml(
        tmp_path / "operations" / "custom.yaml",
        {
            "kind": "artifact",
            "entrypoint": "example.custom",
            "output": "shared/custom.json",
        },
    )
    operations = _operations(project_yaml)
    assert {task.id for task in operations} == {"calendar", "custom"}
    assert all(
        isinstance(task, ArtifactTask) and task.dataset is None for task in operations
    )
    assert all(artifact_kind(task) is None for task in operations)


def test_dataset_file_requires_version_and_rejects_duplicate_identity(tmp_path):
    project_yaml = _project(tmp_path, ("alpha",))
    _write_yaml(
        tmp_path / "datasets" / "alpha.yaml",
        {"sample": {"cadence": "1d", "rounding": "exact"}},
    )
    with pytest.raises(ValueError, match="must declare version"):
        load_datasets(load_project(project_yaml))
    _write_yaml(
        tmp_path / "datasets" / "alpha.yaml",
        {"version": "v1", "sample": {"cadence": "1d", "rounding": "exact"}},
    )
    _write_yaml(
        tmp_path / "datasets" / "nested" / "alpha.yaml",
        {"version": "v2", "sample": {"cadence": "1d", "rounding": "exact"}},
    )
    with pytest.raises(ValueError, match="Duplicate dataset ID 'alpha'"):
        load_datasets(load_project(project_yaml))


@pytest.mark.parametrize(
    "version", ["", "..", "../v1", "v1/v2", "v1\\v2", " v1", "v1 "]
)
def test_dataset_version_is_an_unambiguous_path_segment(version):
    with pytest.raises(ValueError):
        DatasetConfig(
            version=version, sample=SampleConfig(cadence="1d", rounding="exact")
        )


def test_dataset_profile_tokens_can_be_deferred_then_bound_explicitly(tmp_path):
    project = load_project(_project(tmp_path))
    deferred = project.resolve_config(
        {"directory": "outputs/${dataset_id}/${dataset_version}"},
        variables={
            "dataset_id": "${dataset_id}",
            "dataset_version": "${dataset_version}",
        },
    )
    assert deferred == {"directory": "outputs/${dataset_id}/${dataset_version}"}
    assert project.resolve_config(
        deferred, variables={"dataset_id": "alpha", "dataset_version": "v2"}
    ) == {"directory": "outputs/alpha/v2"}
    with pytest.raises(ValueError, match="Unknown interpolation variable 'dataset_id'"):
        project.resolve_config(deferred)


@pytest.mark.parametrize("key", ["dataset_id", "dataset_version"])
def test_project_globals_cannot_shadow_selected_dataset_identity(tmp_path, key):
    path = _project(tmp_path)
    data = yaml.safe_load(path.read_text())
    data["globals"] = {key: "hidden-override"}
    _write_yaml(path, data)
    with pytest.raises(ValueError, match="must not redefine reserved variable"):
        load_project(path)


def test_materialize_profile_selects_operation_instead_of_stream():
    profile = MaterializeProfile(
        cmd="materialize",
        name="prices",
        operation="prices",
        output=Path("prices.jsonl"),
    )
    assert profile.operation == "prices"
    with pytest.raises(ValueError):
        MaterializeProfile.model_validate(
            {
                "cmd": "materialize",
                "name": "prices",
                "stream": "prices",
                "output": "prices.jsonl",
            }
        )


@pytest.mark.parametrize("token", ["dataset_id", "dataset_version"])
def test_dataset_log_paths_must_be_configured_on_concrete_profiles(tmp_path, token):
    project = _project(tmp_path, ("alpha",))
    _write_yaml(
        tmp_path / "operations" / "samples.yaml",
        {"kind": "runtime", "entrypoint": "core.runtime.dataset", "dataset": "alpha"},
    )
    profile_path = tmp_path / "profiles" / "serve.alpha.yaml"
    defaults_path = tmp_path / "profiles" / "serve.defaults.yaml"
    observability = {
        "logging": {"outputs": [{"transport": "fs", "path": f"logs/${{{token}}}.log"}]}
    }
    _write_yaml(profile_path, {"operation": "samples"})
    _write_yaml(defaults_path, {"observability": observability})
    with pytest.raises(
        ProfileCommandError,
        match="Shared prerequisite logs have no selected dataset.*concrete profiles",
    ):
        build_runtime_run_request("serve", str(project))

    _write_yaml(defaults_path, {})
    _write_yaml(profile_path, {"operation": "samples", "observability": observability})
    request = build_runtime_run_request("serve", str(project))
    assert request is not None
    expected = "alpha" if token == "dataset_id" else "v1"
    assert (
        request.jobs[0].observability.log_output.outputs[0].destination
        == tmp_path / "logs" / f"{expected}.log"
    )
