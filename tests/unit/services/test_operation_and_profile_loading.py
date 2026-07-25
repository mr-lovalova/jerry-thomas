from pathlib import Path

import pytest

from datapipeline.config.profiles.materialize import MaterializeProfile
from datapipeline.config.tasks.base import ArtifactTask, RuntimeTask
from datapipeline.config.tasks.coverage import CoverageTask
from datapipeline.config.tasks.dataset import DatasetTask
from datapipeline.config.tasks.matrix import MatrixOptions, MatrixTask
from datapipeline.profiles.loader import (
    apply_profile_defaults,
    profile_specs_with_defaults,
)
from datapipeline.services.operations import (
    operation_documents,
    operations_from_documents,
)
from datapipeline.services.project import load_project


def _tasks(project_yaml: Path):
    project = load_project(project_yaml)
    return operations_from_documents(project, operation_documents(project))


def _artifact_tasks(project_yaml: Path):
    return [task for task in _tasks(project_yaml) if isinstance(task, ArtifactTask)]


def _all_tasks(project_yaml: Path):
    return _tasks(project_yaml)


def _serve_profiles(project_yaml: Path):
    return profile_specs_with_defaults(load_project(project_yaml), cmd="serve")[0]


def _build_profiles(project_yaml: Path):
    return profile_specs_with_defaults(load_project(project_yaml), cmd="build")[0]


def _inspect_profiles(project_yaml: Path):
    return profile_specs_with_defaults(load_project(project_yaml), cmd="inspect")[0]


def _materialize_profiles(project_yaml: Path):
    return profile_specs_with_defaults(load_project(project_yaml), cmd="materialize")[0]


def _serve_defaults(project_yaml: Path):
    return profile_specs_with_defaults(load_project(project_yaml), cmd="serve")[1]


def _materialize_defaults(project_yaml: Path):
    return profile_specs_with_defaults(load_project(project_yaml), cmd="materialize")[1]


def _write_project(tmp_path: Path, operations_ref: str | None = None) -> Path:
    project_yaml = tmp_path / "project.yaml"
    lines = [
        "schema_version: 4",
        "artifact_revision: 1",
        "paths:",
        "  streams: streams",
        "  sources: sources",
        "  dataset: dataset.yaml",
        "  artifacts: artifacts",
    ]
    if operations_ref:
        lines.append(f"  operations: {operations_ref}")
    project_yaml.write_text("\n".join(lines), encoding="utf-8")
    return project_yaml


def _operations_dir(project_yaml: Path) -> Path:
    path = project_yaml.parent / "operations"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _profiles_dir(project_yaml: Path) -> Path:
    path = project_yaml.parent / "profiles"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _profile_kind_dir(project_yaml: Path) -> Path:
    path = _profiles_dir(project_yaml)
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_runtime_task_rejects_non_serializable_options() -> None:
    with pytest.raises(ValueError, match="JSON-serializable"):
        RuntimeTask(
            id="plugin",
            entrypoint="plugin.runtime",
            options={"value": object()},
        )


def test_artifact_tasks_load_configs(tmp_path):
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    config_dir = _operations_dir(project_yaml)
    (config_dir / "schema.yaml").write_text(
        ("kind: artifact\nentrypoint: plugin.artifact.schema\noutput: schema.json\n"),
        encoding="utf-8",
    )
    (config_dir / "scaler.yaml").write_text(
        "output: stats.pkl\n",
        encoding="utf-8",
    )

    tasks = _artifact_tasks(project_yaml)

    assert {task.id for task in tasks} == {
        "scaler",
        "series",
        "metadata",
        "coverage_stats",
        "schema",
    }
    schema = next(task for task in tasks if task.id == "schema")
    assert type(schema) is ArtifactTask
    assert schema.entrypoint == "plugin.artifact.schema"
    assert schema.output == "schema.json"
    scaler = next(task for task in tasks if task.id == "scaler")
    assert scaler.output == "stats.pkl"


def test_core_operations_load_without_configuration(tmp_path):
    project_yaml = _write_project(tmp_path)

    tasks = _tasks(project_yaml)
    artifact_tasks = [task for task in tasks if isinstance(task, ArtifactTask)]
    runtime_tasks = [task for task in tasks if isinstance(task, RuntimeTask)]

    assert [task.id for task in artifact_tasks] == [
        "scaler",
        "series",
        "metadata",
        "coverage_stats",
    ]
    assert [task.id for task in runtime_tasks] == [
        "dataset",
        "coverage",
        "matrix",
    ]
    task = next(task for task in artifact_tasks if task.id == "series")
    assert task.id == "series"
    assert task.entrypoint == "core.artifact.series"
    assert task.output == "build/series/manifest.json"
    coverage_stats = next(
        task for task in artifact_tasks if task.id == "coverage_stats"
    )
    assert coverage_stats.entrypoint == "core.artifact.coverage_stats"
    assert coverage_stats.output == "build/coverage_stats.json"


@pytest.mark.parametrize("operation_id", ["vector_inputs", "variable_records"])
def test_legacy_series_override_is_not_a_core_operation(
    tmp_path,
    operation_id: str,
) -> None:
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    config_dir = _operations_dir(project_yaml)
    (config_dir / f"{operation_id}.yaml").write_text(
        f"output: build/{operation_id}/manifest.json\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=rf"Custom operation '{operation_id}' must set kind to artifact or runtime",
    ):
        _tasks(project_yaml)


def test_ticks_artifact_task_loads_arbitrary_id(tmp_path):
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    config_dir = _operations_dir(project_yaml)
    (config_dir / "dataset_ticks.yaml").write_text(
        (
            "kind: artifact\n"
            "entrypoint: core.artifact.ticks\n"
            "stream: reference.stream\n"
            "output: build/dataset_ticks.jsonl\n"
        ),
        encoding="utf-8",
    )

    tasks = _artifact_tasks(project_yaml)

    task = next(task for task in tasks if task.id == "dataset_ticks")
    assert task.id == "dataset_ticks"
    assert task.entrypoint == "core.artifact.ticks"
    assert task.stream == "reference.stream"
    assert task.grid_by == []
    assert task.output == "build/dataset_ticks.jsonl"


def test_ticks_artifact_task_loads_grid_by(tmp_path):
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    config_dir = _operations_dir(project_yaml)
    (config_dir / "model_grid.yaml").write_text(
        (
            "kind: artifact\n"
            "entrypoint: core.artifact.ticks\n"
            "stream: reference.stream\n"
            "grid_by: [security_id]\n"
            "output: build/model_grid.jsonl\n"
        ),
        encoding="utf-8",
    )

    tasks = _artifact_tasks(project_yaml)

    task = next(task for task in tasks if task.id == "model_grid")
    assert task.grid_by == ["security_id"]


@pytest.mark.parametrize(
    "body",
    [
        "stream: '   '\n",
        "stream: reference.stream\ngrid_by: ['']\n",
        "stream: reference.stream\ngrid_by: [security_id, security_id]\n",
    ],
)
def test_ticks_artifact_rejects_invalid_identity_fields(
    tmp_path: Path,
    body: str,
) -> None:
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    config_dir = _operations_dir(project_yaml)
    (config_dir / "model_grid.yaml").write_text(
        (
            "kind: artifact\n"
            "entrypoint: core.artifact.ticks\n"
            f"{body}"
            "output: build/model_grid.jsonl\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        _artifact_tasks(project_yaml)


@pytest.mark.parametrize("output", ["", "   ", ".", "./"])
def test_artifact_operation_rejects_empty_output(
    tmp_path: Path,
    output: str,
) -> None:
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    config_dir = _operations_dir(project_yaml)
    (config_dir / "snapshot.yaml").write_text(
        (f"kind: artifact\nentrypoint: plugin.artifact.snapshot\noutput: {output!r}\n"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="output must name a file"):
        _artifact_tasks(project_yaml)


def test_core_operation_rejects_entrypoint_override(tmp_path):
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    config_dir = _operations_dir(project_yaml)
    (config_dir / "metadata.yaml").write_text(
        (
            "entrypoint: core.artifact.ticks\n"
            "stream: reference.stream\n"
            "output: build/metadata_ticks.jsonl\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cannot replace its kind or entrypoint"):
        _artifact_tasks(project_yaml)


def test_coverage_stats_task_loads_configs(tmp_path):
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    config_dir = _operations_dir(project_yaml)
    (config_dir / "coverage_stats.yaml").write_text(
        "output: build/custom-coverage-stats.json\nstage: assembled\n",
        encoding="utf-8",
    )

    tasks = _artifact_tasks(project_yaml)
    coverage_stats = next(task for task in tasks if task.id == "coverage_stats")
    assert coverage_stats.output == "build/custom-coverage-stats.json"
    assert coverage_stats.stage == "assembled"


def test_coverage_stats_task_defaults_to_postprocessed_stage(tmp_path):
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    config_dir = _operations_dir(project_yaml)
    (config_dir / "coverage_stats.yaml").write_text(
        "output: build/custom-coverage-stats.json\n",
        encoding="utf-8",
    )

    coverage_stats = next(
        task for task in _artifact_tasks(project_yaml) if task.id == "coverage_stats"
    )

    assert coverage_stats.stage == "postprocessed"


def test_coverage_stats_task_rejects_unknown_fields(tmp_path):
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    config_dir = _operations_dir(project_yaml)
    (config_dir / "coverage_stats.yaml").write_text(
        "unexpected: true\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        _artifact_tasks(project_yaml)


def test_serve_tasks_respect_name_and_enabled(tmp_path):
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    config_dir = _profile_kind_dir(project_yaml)
    (config_dir / "serve.train.yaml").write_text(
        "operation: dataset\n", encoding="utf-8"
    )
    (config_dir / "serve.val.yaml").write_text(
        "operation: dataset\nenabled: false\n",
        encoding="utf-8",
    )

    tasks = _serve_profiles(project_yaml)

    assert [task.name for task in tasks] == ["train", "val"]
    assert [task.name for task in tasks if task.enabled] == ["train"]


def test_serve_profiles_load_include_outputs(tmp_path):
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    config_dir = _profile_kind_dir(project_yaml)
    (config_dir / "serve.dataset.yaml").write_text(
        "operation: dataset\ninclude_outputs: [train, val]\n",
        encoding="utf-8",
    )

    tasks = _serve_profiles(project_yaml)

    assert tasks[0].include_outputs == ["train", "val"]


def test_serve_defaults_supply_include_outputs_unless_profile_overrides(tmp_path):
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    profiles_dir = _profile_kind_dir(project_yaml)
    (profiles_dir / "serve.defaults.yaml").write_text(
        "include_outputs: [train, val]\n",
        encoding="utf-8",
    )
    (profiles_dir / "serve.all.yaml").write_text(
        "operation: dataset\n",
        encoding="utf-8",
    )
    (profiles_dir / "serve.test.yaml").write_text(
        "operation: dataset\ninclude_outputs: [test]\n",
        encoding="utf-8",
    )

    profiles = _serve_profiles(project_yaml)
    defaults = _serve_defaults(project_yaml)
    merged = [apply_profile_defaults(profile, defaults) for profile in profiles]

    assert [profile.include_outputs for profile in merged] == [
        ["train", "val"],
        ["test"],
    ]


def test_serve_artifact_mode_is_defaults_only(tmp_path):
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    profiles_dir = _profile_kind_dir(project_yaml)
    (profiles_dir / "serve.dataset.yaml").write_text(
        "operation: dataset\nartifact_mode: AUTO\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        _serve_profiles(project_yaml)


def test_inspect_artifact_mode_is_defaults_only(tmp_path):
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    profiles_dir = _profile_kind_dir(project_yaml)
    (profiles_dir / "inspect.coverage.yaml").write_text(
        "operation: coverage\nartifact_mode: AUTO\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        _inspect_profiles(project_yaml)


def test_serve_defaults_keep_command_wide_artifact_mode_out_of_profile(tmp_path):
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    profiles_dir = _profile_kind_dir(project_yaml)
    (profiles_dir / "serve.defaults.yaml").write_text(
        "artifact_mode: FORCE\n",
        encoding="utf-8",
    )
    (profiles_dir / "serve.dataset.yaml").write_text(
        "operation: dataset\n",
        encoding="utf-8",
    )

    profile = _serve_profiles(project_yaml)[0]
    defaults = _serve_defaults(project_yaml)
    merged = apply_profile_defaults(profile, defaults)

    assert defaults.artifact_mode == "FORCE"
    assert not hasattr(merged, "artifact_mode")


@pytest.mark.parametrize("value", [".inf", ".nan"])
def test_serve_profile_rejects_non_finite_throttle(tmp_path, value):
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    profiles_dir = _profile_kind_dir(project_yaml)
    (profiles_dir / "serve.dataset.yaml").write_text(
        f"operation: dataset\nthrottle_ms: {value}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="finite number"):
        _serve_profiles(project_yaml)


@pytest.mark.parametrize("value", [".inf", ".nan"])
def test_serve_defaults_reject_non_finite_throttle(tmp_path, value):
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    profiles_dir = _profile_kind_dir(project_yaml)
    (profiles_dir / "serve.defaults.yaml").write_text(
        f"throttle_ms: {value}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="finite number"):
        _serve_defaults(project_yaml)


@pytest.mark.parametrize("value", [".inf", ".nan"])
def test_profile_rejects_non_finite_heartbeat(tmp_path, value):
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    profiles_dir = _profile_kind_dir(project_yaml)
    (profiles_dir / "serve.dataset.yaml").write_text(
        f"operation: dataset\nobservability:\n  heartbeat_interval_seconds: {value}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="finite number"):
        _serve_profiles(project_yaml)


@pytest.mark.parametrize("value", [".inf", ".nan"])
def test_profile_defaults_reject_non_finite_heartbeat(tmp_path, value):
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    profiles_dir = _profile_kind_dir(project_yaml)
    (profiles_dir / "serve.defaults.yaml").write_text(
        f"observability:\n  heartbeat_interval_seconds: {value}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="finite number"):
        _serve_defaults(project_yaml)


def test_serve_profiles_interpolate_project_globals(tmp_path):
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    project_yaml.write_text(
        "\n".join(
            [
                "schema_version: 4",
                "artifact_revision: 1",
                "name: momentum",
                "variant: price",
                "paths:",
                "  streams: streams",
                "  sources: sources",
                "  dataset: dataset.yaml",
                "  artifacts: artifacts",
                "  operations: operations",
                "  profiles: profiles",
                "globals:",
                "  adv_window_days: 20",
                "  volatility_window_days: 63",
                "  momentum_skip_days: 21",
                "  momentum_lag_days: 189",
                "  forward_return_horizon_days: 126",
                "  dataset_version: ${project_name}_${project_variant}_adv${adv_window_days}_vol${volatility_window_days}_mom${momentum_lag_days}_${momentum_skip_days}_fwd${forward_return_horizon_days}",
            ]
        ),
        encoding="utf-8",
    )
    profiles_dir = _profile_kind_dir(project_yaml)
    (profiles_dir / "serve.processed.yaml").write_text(
        "\n".join(
            [
                "operation: dataset",
                "output:",
                "  transport: fs",
                "  format: jsonl",
                "  view: raw",
                "  directory: ../../data/processed/${dataset_version}",
                "  filename: ${dataset_version}",
            ]
        ),
        encoding="utf-8",
    )

    profile = _serve_profiles(project_yaml)[0]

    expected = "momentum_price_adv20_vol63_mom189_21_fwd126"
    assert profile.output.directory == Path(f"../../data/processed/{expected}")
    assert profile.output.filename == expected


def test_profile_defaults_interpolate_project_globals(tmp_path):
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    project_yaml.write_text(
        "\n".join(
            [
                "schema_version: 4",
                "artifact_revision: 1",
                "name: momentum",
                "variant: price",
                "paths:",
                "  streams: streams",
                "  sources: sources",
                "  dataset: dataset.yaml",
                "  artifacts: artifacts",
                "  operations: operations",
                "  profiles: profiles",
                "globals:",
                "  dataset_version: ${project_name}_${project_variant}",
            ]
        ),
        encoding="utf-8",
    )
    profiles_dir = _profile_kind_dir(project_yaml)
    (profiles_dir / "serve.defaults.yaml").write_text(
        "\n".join(
            [
                "output:",
                "  transport: fs",
                "  format: jsonl",
                "  view: raw",
                "  directory: ../../data/processed/${dataset_version}",
            ]
        ),
        encoding="utf-8",
    )

    defaults = _serve_defaults(project_yaml)

    assert defaults.output.directory == Path("../../data/processed/momentum_price")


def test_profile_identity_comes_from_filename(tmp_path):
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    config_dir = _profile_kind_dir(project_yaml)
    (config_dir / "serve.train.yaml").write_text(
        "operation: dataset\n", encoding="utf-8"
    )

    profile = _serve_profiles(project_yaml)[0]

    assert profile.cmd == "serve"
    assert profile.name == "train"


def test_profile_version_field_is_rejected(tmp_path):
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    profiles_dir = _profile_kind_dir(project_yaml)
    (profiles_dir / "serve.train.yaml").write_text(
        "operation: dataset\nversion: 1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        _serve_profiles(project_yaml)


def test_profile_defaults_version_field_is_rejected(tmp_path):
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    profiles_dir = _profile_kind_dir(project_yaml)
    (profiles_dir / "serve.defaults.yaml").write_text(
        "version: 1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        _serve_profiles(project_yaml)


def test_build_profiles_load_and_respect_enabled(tmp_path):
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    config_dir = _profile_kind_dir(project_yaml)
    (config_dir / "build.fast.yaml").write_text(
        "enabled: true\nmode: auto\noperation: metadata\n",
        encoding="utf-8",
    )
    (config_dir / "build.full.yaml").write_text(
        "enabled: false\nmode: force\noperation: coverage_stats\n",
        encoding="utf-8",
    )

    tasks = _build_profiles(project_yaml)
    assert [task.name for task in tasks] == ["fast", "full"]
    assert tasks[0].operation == "metadata"
    assert tasks[0].enabled is True
    assert tasks[1].enabled is False


def test_inspect_profiles_load_and_respect_enabled(tmp_path):
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    config_dir = _profile_kind_dir(project_yaml)
    (config_dir / "inspect.default.yaml").write_text(
        "enabled: true\noperation: coverage\n",
        encoding="utf-8",
    )
    (config_dir / "inspect.extra.yaml").write_text(
        "enabled: false\noperation: coverage\n",
        encoding="utf-8",
    )

    tasks = _inspect_profiles(project_yaml)
    assert [task.name for task in tasks] == ["default", "extra"]
    assert tasks[0].operation == "coverage"
    assert tasks[0].enabled is True
    assert tasks[1].enabled is False


def test_materialize_profiles_load_and_normalize_fields(tmp_path):
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    profiles_dir = _profile_kind_dir(project_yaml)
    (profiles_dir / "materialize.adv-20.yaml").write_text(
        (
            "order: 20\n"
            "stream: ' adv.20 '\n"
            "output: ' data/features/adv/20.jsonl '\n"
            "overwrite: true\n"
            "observability:\n"
            "  visuals: on\n"
        ),
        encoding="utf-8",
    )

    profiles = _materialize_profiles(project_yaml)

    assert len(profiles) == 1
    profile = profiles[0]
    assert isinstance(profile, MaterializeProfile)
    assert profile.stream == "adv.20"
    assert profile.output == Path("data/features/adv/20.jsonl")
    assert profile.overwrite is True
    assert profile.observability is not None
    assert profile.observability.visuals == "ON"


def test_materialize_profile_accepts_gzip_output(tmp_path):
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    profiles_dir = _profile_kind_dir(project_yaml)
    (profiles_dir / "materialize.adv-20.yaml").write_text(
        "stream: adv.20\noutput: data/features/adv/20.jsonl.gz\n",
        encoding="utf-8",
    )

    profile = _materialize_profiles(project_yaml)[0]

    assert profile.output == Path("data/features/adv/20.jsonl.gz")


def test_materialize_profile_rejects_compression_field(tmp_path):
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    profiles_dir = _profile_kind_dir(project_yaml)
    (profiles_dir / "materialize.adv-20.yaml").write_text(
        "stream: adv.20\noutput: data/features/adv/20.jsonl.gz\ncompression: gzip\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        _materialize_profiles(project_yaml)


@pytest.mark.parametrize("field", ["stream", "output"])
def test_materialize_profile_requires_nonempty_paths(tmp_path, field):
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    profiles_dir = _profile_kind_dir(project_yaml)
    values = {"stream": "adv.20", "output": "adv-20.jsonl"}
    values[field] = "   "
    (profiles_dir / "materialize.adv-20.yaml").write_text(
        (f"stream: '{values['stream']}'\noutput: '{values['output']}'\n"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=f"{field} must be set"):
        _materialize_profiles(project_yaml)


def test_materialize_profile_requires_jsonl_output(tmp_path):
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    profiles_dir = _profile_kind_dir(project_yaml)
    (profiles_dir / "materialize.adv-20.yaml").write_text(
        ("stream: adv.20\noutput: data/features/adv/20.csv\n"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="output must use a .jsonl or .jsonl.gz path"):
        _materialize_profiles(project_yaml)


def test_materialize_profile_requires_boolean_overwrite(tmp_path):
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    profiles_dir = _profile_kind_dir(project_yaml)
    (profiles_dir / "materialize.adv-20.yaml").write_text(
        ("stream: adv.20\noutput: data/features/adv/20.jsonl\noverwrite: 'false'\n"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="valid boolean"):
        _materialize_profiles(project_yaml)


def test_materialize_defaults_apply_overwrite_and_observability(tmp_path):
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    profiles_dir = _profile_kind_dir(project_yaml)
    (profiles_dir / "materialize.defaults.yaml").write_text(
        (
            "artifact_mode: force\n"
            "overwrite: true\n"
            "observability:\n"
            "  heartbeat_interval_seconds: 30\n"
        ),
        encoding="utf-8",
    )
    (profiles_dir / "materialize.adv-20.yaml").write_text(
        (
            "stream: adv.20\n"
            "output: data/features/adv/20.jsonl\n"
            "observability:\n"
            "  visuals: off\n"
        ),
        encoding="utf-8",
    )

    profile = _materialize_profiles(project_yaml)[0]
    defaults = _materialize_defaults(project_yaml)
    merged = apply_profile_defaults(profile, defaults)

    assert defaults.artifact_mode == "FORCE"
    assert isinstance(merged, MaterializeProfile)
    assert not hasattr(merged, "artifact_mode")
    assert merged.overwrite is True
    assert merged.observability is not None
    assert merged.observability.visuals == "OFF"
    assert merged.observability.heartbeat_interval_seconds == 30


def test_materialize_defaults_reject_compression_field(tmp_path):
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    profiles_dir = _profile_kind_dir(project_yaml)
    (profiles_dir / "materialize.defaults.yaml").write_text(
        "compression: gzip\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        _materialize_defaults(project_yaml)


def test_materialize_artifact_mode_is_defaults_only(tmp_path):
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    profiles_dir = _profile_kind_dir(project_yaml)
    (profiles_dir / "materialize.adv-20.yaml").write_text(
        ("stream: adv.20\noutput: data/features/adv/20.jsonl\nartifact_mode: AUTO\n"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        _materialize_profiles(project_yaml)


def test_materialize_defaults_reject_unknown_artifact_mode(tmp_path):
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    profiles_dir = _profile_kind_dir(project_yaml)
    (profiles_dir / "materialize.defaults.yaml").write_text(
        "artifact_mode: SOMETIMES\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="artifact mode must be one of"):
        _materialize_defaults(project_yaml)


def test_profile_order_overrides_file_order(tmp_path):
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    profiles_dir = _profile_kind_dir(project_yaml)
    (profiles_dir / "serve.train.yaml").write_text(
        "operation: dataset\n",
        encoding="utf-8",
    )
    (profiles_dir / "serve.val.yaml").write_text(
        "order: 2\noperation: dataset\n",
        encoding="utf-8",
    )
    (profiles_dir / "serve.test.yaml").write_text(
        "order: 1\noperation: dataset\n",
        encoding="utf-8",
    )

    tasks = _serve_profiles(project_yaml)
    assert [task.name for task in tasks] == ["test", "val", "train"]


def test_serve_defaults_are_loaded_but_not_executable_profiles(tmp_path):
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    profiles_dir = _profile_kind_dir(project_yaml)
    (profiles_dir / "serve.defaults.yaml").write_text(
        ("observability:\n  logging:\n    outputs:\n      - transport: stdout\n"),
        encoding="utf-8",
    )
    (profiles_dir / "serve.train.yaml").write_text(
        "operation: dataset\n",
        encoding="utf-8",
    )

    tasks = _serve_profiles(project_yaml)
    defaults = _serve_defaults(project_yaml)

    assert [task.name for task in tasks] == ["train"]
    assert defaults.cmd == "serve"
    assert defaults.observability is not None


def test_missing_profile_defaults_resolve_to_builtins(tmp_path):
    project_yaml = _write_project(tmp_path, operations_ref="operations")

    defaults = _serve_defaults(project_yaml)

    assert defaults.cmd == "serve"
    assert defaults.execution.sort_buffer_mb == 128


def test_profile_defaults_reject_executable_fields(tmp_path):
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    profiles_dir = _profile_kind_dir(project_yaml)
    (profiles_dir / "serve.defaults.yaml").write_text(
        "name: should-not-exist\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        _serve_profiles(project_yaml)


def test_profile_defaults_reject_body_command(tmp_path):
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    profiles_dir = _profile_kind_dir(project_yaml)
    (profiles_dir / "serve.defaults.yaml").write_text(
        "cmd: serve\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="command comes from the defaults filename"):
        _serve_profiles(project_yaml)


@pytest.mark.parametrize("identity", ["cmd: serve\n", "name: train\n"])
def test_concrete_profile_rejects_body_identity(tmp_path, identity):
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    profiles_dir = _profile_kind_dir(project_yaml)
    (profiles_dir / "serve.train.yaml").write_text(
        identity + "operation: dataset\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="come from the filename"):
        _serve_profiles(project_yaml)


def test_profile_file_requires_one_mapping(tmp_path):
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    profiles_dir = _profile_kind_dir(project_yaml)
    (profiles_dir / "serve.train.yaml").write_text(
        "- operation: dataset\n- operation: dataset\n",
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match="one mapping profile"):
        _serve_profiles(project_yaml)


def test_command_load_ignores_malformed_other_profile_kinds(tmp_path):
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    profiles_dir = _profile_kind_dir(project_yaml)
    (profiles_dir / "serve.train.yaml").write_text(
        "operation: dataset\n",
        encoding="utf-8",
    )
    (profiles_dir / "build.broken.yaml").write_text(
        "- not-a-profile\n",
        encoding="utf-8",
    )

    assert [profile.name for profile in _serve_profiles(project_yaml)] == ["train"]
    with pytest.raises(TypeError, match="one mapping profile"):
        _build_profiles(project_yaml)


def test_execution_policy_is_not_allowed_on_concrete_profiles(tmp_path):
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    profiles_dir = _profile_kind_dir(project_yaml)
    (profiles_dir / "serve.train.yaml").write_text(
        ("operation: dataset\nexecution:\n  sort_buffer_mb: 1\n"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        _serve_profiles(project_yaml)


def test_duplicate_profile_defaults_per_kind_raise(tmp_path):
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    profiles_dir = _profile_kind_dir(project_yaml)
    (profiles_dir / "serve.defaults.yaml").write_text(
        "",
        encoding="utf-8",
    )
    (profiles_dir / "serve.defaults.yml").write_text(
        "",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate serve defaults"):
        _serve_profiles(project_yaml)


def test_artifact_operation_rejects_dependencies_field(tmp_path):
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    config_dir = _operations_dir(project_yaml)
    (config_dir / "metadata.yaml").write_text(
        "dependencies:\n  - schema\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        _artifact_tasks(project_yaml)


def test_dataset_operation_loads(tmp_path):
    project_yaml = _write_project(tmp_path)

    task = next(task for task in _all_tasks(project_yaml) if task.id == "dataset")
    assert isinstance(task, DatasetTask)
    assert task.entrypoint == "core.runtime.dataset"


def test_coverage_operation_options_are_typed(tmp_path):
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    config_dir = _operations_dir(project_yaml)
    (config_dir / "coverage.yaml").write_text(
        "options:\n  threshold: 0.8\n",
        encoding="utf-8",
    )

    operation = next(task for task in _all_tasks(project_yaml) if task.id == "coverage")

    assert isinstance(operation, CoverageTask)
    assert operation.options.threshold == 0.8


@pytest.mark.parametrize(
    ("entrypoint", "model_cls"),
    [
        ("core.runtime.dataset", DatasetTask),
        ("core.runtime.matrix", MatrixTask),
    ],
)
def test_runtime_tasks_without_options_accept_empty_options(
    tmp_path: Path,
    entrypoint: str,
    model_cls: type[RuntimeTask],
) -> None:
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    config_dir = _operations_dir(project_yaml)
    (config_dir / "runtime.yaml").write_text(
        (f"kind: runtime\nentrypoint: {entrypoint}\noptions: {{}}\n"),
        encoding="utf-8",
    )

    operation = next(task for task in _all_tasks(project_yaml) if task.id == "runtime")

    assert isinstance(operation, model_cls)
    if isinstance(operation, MatrixTask):
        assert operation.options == MatrixOptions()
    else:
        assert operation.options == {}


@pytest.mark.parametrize(
    "entrypoint",
    ["core.runtime.pipeline", "core.runtime.typo"],
)
def test_unknown_core_runtime_entrypoint_is_rejected_during_loading(
    tmp_path: Path,
    entrypoint: str,
) -> None:
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    config_dir = _operations_dir(project_yaml)
    (config_dir / "invalid.yaml").write_text(
        f"kind: runtime\nentrypoint: {entrypoint}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported core runtime entrypoint"):
        _all_tasks(project_yaml)


@pytest.mark.parametrize(
    ("entrypoint", "option", "error"),
    [
        (
            "core.runtime.dataset",
            "  sort: missing\n",
            "dataset operation does not accept options",
        ),
        (
            "core.runtime.matrix",
            "  rows: 10\n",
            "Extra inputs are not permitted",
        ),
        (
            "core.runtime.coverage",
            "  sort: typo\n",
            "Extra inputs are not permitted",
        ),
        (
            "core.runtime.coverage",
            "  threshold: 1.1\n",
            "less than or equal to 1",
        ),
        (
            "core.runtime.coverage",
            "  threshold: true\n",
            "Input should be a valid number",
        ),
        (
            "core.runtime.matrix",
            "  max_cells: 0\n",
            "greater than 0",
        ),
        (
            "core.runtime.matrix",
            "  stage: raw\n",
            "Input should be 'assembled' or 'postprocessed'",
        ),
    ],
)
def test_builtin_runtime_tasks_reject_invalid_options(
    tmp_path: Path,
    entrypoint: str,
    option: str,
    error: str,
) -> None:
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    config_dir = _operations_dir(project_yaml)
    (config_dir / "runtime.yaml").write_text(
        (f"kind: runtime\nentrypoint: {entrypoint}\noptions:\n{option}"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=error):
        _all_tasks(project_yaml)


def test_coverage_options_default_to_current_threshold(tmp_path: Path) -> None:
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    config_dir = _operations_dir(project_yaml)
    (config_dir / "runtime.yaml").write_text(
        "kind: runtime\nentrypoint: core.runtime.coverage\n",
        encoding="utf-8",
    )

    operation = next(task for task in _all_tasks(project_yaml) if task.id == "runtime")

    assert isinstance(operation, CoverageTask)
    assert operation.options.threshold == 0.95


def test_plugin_runtime_options_remain_plugin_owned(tmp_path: Path) -> None:
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    config_dir = _operations_dir(project_yaml)
    (config_dir / "custom.yaml").write_text(
        (
            "kind: runtime\n"
            "entrypoint: plugin.runtime.custom\n"
            "requires: [ Custom_Snapshot ]\n"
            "options:\n"
            "  nested:\n"
            "    value: 3\n"
        ),
        encoding="utf-8",
    )

    task = next(task for task in _all_tasks(project_yaml) if task.id == "custom")

    assert type(task) is RuntimeTask
    assert task.requires == ("custom_snapshot",)
    assert task.options == {"nested": {"value": 3}}


@pytest.mark.parametrize(
    "entrypoint",
    [
        "core.artifact.scaler",
        "core.artifact.series",
        "core.artifact.metadata",
        "core.artifact.coverage_stats",
    ],
)
def test_custom_artifact_rejects_reserved_core_entrypoint(
    tmp_path: Path,
    entrypoint: str,
) -> None:
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    config_dir = _operations_dir(project_yaml)
    (config_dir / "custom.yaml").write_text(
        (f"kind: artifact\nentrypoint: {entrypoint}\noutput: build/custom.json\n"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="is reserved for core operation"):
        _all_tasks(project_yaml)


def test_custom_operation_rejects_entrypoint_outer_whitespace(tmp_path: Path) -> None:
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    config_dir = _operations_dir(project_yaml)
    (config_dir / "custom.yaml").write_text(
        (
            "kind: runtime\n"
            "entrypoint: ' core.runtime.coverage '\n"
            "options:\n"
            "  typo: true\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must not contain outer whitespace"):
        _all_tasks(project_yaml)


@pytest.mark.parametrize(
    ("requires", "error"),
    [
        ("snapshot", "requires must be a list"),
        ("[snapshot, SNAPSHOT]", "must not contain duplicate"),
        ("['   ']", "requires item must be set"),
    ],
)
def test_runtime_operation_rejects_invalid_requires(
    tmp_path: Path,
    requires: str,
    error: str,
) -> None:
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    config_dir = _operations_dir(project_yaml)
    (config_dir / "custom.yaml").write_text(
        (f"kind: runtime\nentrypoint: plugin.runtime.custom\nrequires: {requires}\n"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=error):
        _all_tasks(project_yaml)


def test_plugin_runtime_options_must_be_a_mapping(tmp_path: Path) -> None:
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    config_dir = _operations_dir(project_yaml)
    (config_dir / "custom.yaml").write_text(
        ("kind: runtime\nentrypoint: plugin.runtime.custom\noptions: []\n"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="options must be a mapping"):
        _all_tasks(project_yaml)


def test_runtime_operation_rejects_dependencies_field(tmp_path):
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    config_dir = _operations_dir(project_yaml)
    (config_dir / "dataset.yaml").write_text(
        "dependencies:\n  - missing_artifact\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        _all_tasks(project_yaml)


def test_runtime_operation_rejects_output_formats_field(tmp_path):
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    config_dir = _operations_dir(project_yaml)
    (config_dir / "dataset.yaml").write_text(
        "output_formats:\n  - jsonl\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        _all_tasks(project_yaml)


def test_duplicate_operation_filenames_raise(tmp_path):
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    config_dir = _operations_dir(project_yaml)
    (config_dir / "snapshot.yaml").write_text(
        "output: snapshot-a.json\n", encoding="utf-8"
    )
    (config_dir / "snapshot.yml").write_text(
        "output: snapshot-b.json\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Duplicate operation ids"):
        _artifact_tasks(project_yaml)


def test_duplicate_serve_profile_names_raise(tmp_path):
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    config_dir = _profile_kind_dir(project_yaml)
    (config_dir / "serve.train.yaml").write_text(
        "operation: dataset\n", encoding="utf-8"
    )
    (config_dir / "serve.train.yml").write_text(
        "operation: dataset\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Duplicate serve profile names"):
        _serve_profiles(project_yaml)


def test_operation_id_comes_from_filename(tmp_path):
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    config_dir = _operations_dir(project_yaml)
    (config_dir / "custom.yaml").write_text(
        "id: another\nkind: runtime\nentrypoint: plugin.runtime\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="filename supplies 'custom'"):
        _all_tasks(project_yaml)


def test_configured_operations_directory_must_exist(tmp_path):
    project_yaml = _write_project(tmp_path, operations_ref="operations")

    with pytest.raises(FileNotFoundError, match="operations directory not found"):
        _all_tasks(project_yaml)


def test_duplicate_build_profile_names_raise(tmp_path):
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    config_dir = _profile_kind_dir(project_yaml)
    (config_dir / "build.nightly.yaml").write_text(
        "operation: coverage_stats\n",
        encoding="utf-8",
    )
    (config_dir / "build.nightly.yml").write_text(
        "operation: metadata\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate build profile names"):
        _build_profiles(project_yaml)


def test_duplicate_inspect_profile_names_raise(tmp_path):
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    config_dir = _profile_kind_dir(project_yaml)
    (config_dir / "inspect.inspect.yaml").write_text(
        "operation: coverage\n",
        encoding="utf-8",
    )
    (config_dir / "inspect.inspect.yml").write_text(
        "operation: coverage\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate inspect profile names"):
        _inspect_profiles(project_yaml)


def test_legacy_config_directory_is_not_loaded(tmp_path):
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    tasks_root = project_yaml.parent / "tasks"
    tasks_root.mkdir(parents=True, exist_ok=True)
    (tasks_root / "report.yaml").write_text(
        "id: report\nkind: runtime\nentrypoint: core.runtime.dataset\n",
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="operations directory not found"):
        _all_tasks(project_yaml)


def test_nested_profile_files_are_rejected(tmp_path):
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    profiles_root = _profiles_dir(project_yaml) / "serve"
    profiles_root.mkdir(parents=True, exist_ok=True)
    (profiles_root / "train.yaml").write_text(
        "operation: dataset\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="flat under profiles/"):
        _serve_profiles(project_yaml)


def test_missing_profile_directory_uses_empty_profiles_and_defaults(tmp_path):
    project_yaml = _write_project(tmp_path, operations_ref="operations")

    assert _serve_profiles(project_yaml) == []
    assert _serve_defaults(project_yaml).cmd == "serve"


def test_profile_path_must_be_a_directory(tmp_path):
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    profiles_root = project_yaml.parent / "profiles"
    profiles_root.write_text("not a directory\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Profiles path must be a directory"):
        _serve_profiles(project_yaml)


def test_profile_filename_rejects_reserved_path_component(tmp_path):
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    profiles_root = _profiles_dir(project_yaml)
    (profiles_root / "inspect....yaml").write_text(
        "operation: coverage\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="profile name must not be"):
        _inspect_profiles(project_yaml)


def test_profile_command_cannot_override_filename(tmp_path):
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    build_dir = _profile_kind_dir(project_yaml)
    (build_dir / "build.oops.yaml").write_text(
        "cmd: serve\noperation: dataset\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="command and name come from the filename"):
        _build_profiles(project_yaml)


def test_profile_filename_prefix_is_required(tmp_path):
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    profiles_root = _profiles_dir(project_yaml)
    (profiles_root / "oops.yaml").write_text(
        "operation: dataset\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="must use \\{serve,build,inspect,materialize\\}",
    ):
        _serve_profiles(project_yaml)


def test_profile_rejects_unknown_fields(tmp_path):
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    profiles_root = _profiles_dir(project_yaml)
    (profiles_root / "build.metadata.yaml").write_text(
        "operation: metadata\noutput: should-not-exist\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        _build_profiles(project_yaml)


def test_operation_task_rejects_unknown_fields(tmp_path):
    project_yaml = _write_project(tmp_path, operations_ref="operations")
    config_dir = _operations_dir(project_yaml)
    (config_dir / "dataset.yaml").write_text(
        ("runtime_kind: inspect\n"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        _all_tasks(project_yaml)
