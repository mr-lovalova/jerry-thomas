from pathlib import Path

import pytest
from pydantic import ValidationError

from jerrythomas.cli.output_options import build_cli_output_config
from jerrythomas.config.dataset.dataset import DatasetConfig, SampleConfig
from jerrythomas.config.dataset.series import SeriesConfig
from jerrythomas.config.dataset.split import (
    DatasetFold,
    HashSplitConfig,
    TimeInterval,
    TimeSplitConfig,
)
from jerrythomas.config.preview import PreviewStage
from jerrythomas.config.profiles.build import BuildProfile
from jerrythomas.config.profiles.inspect import InspectProfile
from jerrythomas.config.profiles.output import ServeOutputConfig
from jerrythomas.config.profiles.serve import ServeProfile
from jerrythomas.config.streams import SourceStreamConfig, StreamsConfig
from jerrythomas.config.tasks.base import PluginRuntimeTask, RuntimeTask
from jerrythomas.config.tasks.dataset import DatasetTask
from jerrythomas.execution.settings import CommandObservability, LogOutputTarget
from jerrythomas.profiles.runtime_profiles import (
    resolve_inspect_profiles,
    resolve_serve_profiles,
)
from tests.unit.profiles.helpers import project_definition


def _definition(
    project_path: Path,
    split: HashSplitConfig | TimeSplitConfig | None = None,
    runtime_operations: tuple[RuntimeTask, ...] = (),
):
    return project_definition(
        project_path,
        dataset=DatasetConfig(
            sample=SampleConfig(cadence="1h"),
            split=split,
        ),
        runtime_operations=runtime_operations,
    )


def _resolve_serve(
    project_path: Path,
    profiles: list[ServeProfile],
    preview: PreviewStage | None = None,
    limit: int | None = None,
    cli_output: ServeOutputConfig | None = None,
    cli_log_outputs: list[LogOutputTarget] | None = None,
    cli_heartbeat_interval_seconds: float | None = None,
    split: HashSplitConfig | TimeSplitConfig | None = None,
    runtime_operations: tuple[RuntimeTask, ...] = (),
):
    return resolve_serve_profiles(
        definition=_definition(project_path, split, runtime_operations),
        profiles=profiles,
        preview=preview,
        limit=limit,
        cli_output=cli_output,
        command_observability=CommandObservability(
            heartbeat_interval_seconds=cli_heartbeat_interval_seconds,
            log_outputs=tuple(cli_log_outputs or ()),
        ),
    )


def _resolve_inspect(
    project_path: Path,
    profiles: list[InspectProfile],
    limit: int | None = None,
    cli_output: ServeOutputConfig | None = None,
    split: HashSplitConfig | TimeSplitConfig | None = None,
):
    return resolve_inspect_profiles(
        definition=_definition(project_path, split),
        profiles=profiles,
        limit=limit,
        cli_output=cli_output,
    )


def test_serve_profile_accepts_include_outputs_list():
    profile = ServeProfile.model_validate(
        {
            "cmd": "serve",
            "name": "dataset",
            "operation": "serve",
            "include_outputs": ["train", "val"],
        }
    )

    assert profile.include_outputs == ["train", "val"]


def test_serve_profile_rejects_include_outputs_string():
    with pytest.raises(ValidationError, match="include_outputs must be a list"):
        ServeProfile.model_validate(
            {
                "cmd": "serve",
                "name": "dataset",
                "operation": "serve",
                "include_outputs": "train",
            }
        )


@pytest.mark.parametrize("label", [42, " train"])
def test_serve_profile_rejects_invalid_include_output_id(label):
    with pytest.raises(ValidationError, match="include_outputs entries must"):
        ServeProfile.model_validate(
            {
                "cmd": "serve",
                "name": "dataset",
                "operation": "serve",
                "include_outputs": [label],
            }
        )


def test_serve_profile_rejects_numeric_operation():
    with pytest.raises(ValidationError, match="operation must be a string"):
        ServeProfile.model_validate(
            {
                "cmd": "serve",
                "name": "dataset",
                "operation": 42,
            }
        )


def test_serve_profile_rejects_numeric_preview() -> None:
    with pytest.raises(ValidationError, match="preview"):
        ServeProfile.model_validate(
            {
                "cmd": "serve",
                "name": "preview",
                "operation": "serve",
                "preview": 3,
            }
        )


@pytest.mark.parametrize(
    ("profile_type", "command"),
    [(ServeProfile, "serve"), (InspectProfile, "inspect")],
)
def test_runtime_profiles_reject_unknown_fields(profile_type, command):
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        profile_type.model_validate(
            {
                "cmd": command,
                "name": "example",
                "operation": "serve",
                "unexpected": True,
            }
        )


def test_runtime_profiles_resolve_heartbeat_setting(tmp_path):
    profile = ServeProfile.model_validate(
        {
            "cmd": "serve",
            "name": "demo",
            "operation": "serve",
            "observability": {"heartbeat_interval_seconds": 30},
        }
    )
    resolved = _resolve_serve(tmp_path, [profile])[0]
    assert resolved.observability.heartbeat_interval_seconds == 30

    cli_override = _resolve_serve(
        tmp_path,
        [profile],
        cli_heartbeat_interval_seconds=0,
    )[0]
    assert cli_override.observability.heartbeat_interval_seconds == 0


def test_dataset_serve_defaults_to_dataset_output_ids(tmp_path):
    profile = ServeProfile.model_validate(
        {
            "cmd": "serve",
            "name": "dataset",
            "operation": "dataset",
            "output": {
                "transport": "fs",
                "format": "jsonl",
                "directory": "runs",
            },
        }
    )
    split = TimeSplitConfig(
        intervals=[
            TimeInterval(id="train", until="2024-01-01T00:00:00Z"),
            TimeInterval(id="purge", until="2024-02-01T00:00:00Z"),
            TimeInterval(id="val"),
        ],
        folds=[DatasetFold(id="default", train=["train"], validation=["val"])],
    )

    resolved = _resolve_serve(
        tmp_path / "project.yaml",
        [profile],
        split=split,
        runtime_operations=(DatasetTask(id="dataset"),),
    )[0]

    assert resolved.output_ids == ("default.train", "default.validation")


def test_dataset_serve_defaults_to_every_walk_forward_output(tmp_path):
    profile = ServeProfile.model_validate(
        {
            "cmd": "serve",
            "name": "dataset",
            "operation": "dataset",
            "output": {
                "transport": "fs",
                "format": "jsonl",
                "directory": "runs",
            },
        }
    )
    split = TimeSplitConfig(
        intervals=[
            TimeInterval(id="train_0", until="2024-01-01T00:00:00Z"),
            TimeInterval(id="val_0", until="2024-02-01T00:00:00Z"),
            TimeInterval(id="train_1", until="2024-03-01T00:00:00Z"),
            TimeInterval(id="val_1"),
        ],
        folds=[
            DatasetFold(
                id="walk_0",
                train=["train_0"],
                validation=["val_0"],
            ),
            DatasetFold(
                id="walk_1",
                train=["train_0", "val_0", "train_1"],
                validation=["val_1"],
            ),
        ],
    )

    resolved = _resolve_serve(
        tmp_path / "project.yaml",
        [profile],
        split=split,
        runtime_operations=(DatasetTask(id="dataset"),),
    )[0]

    assert resolved.output_ids == (
        "walk_0.train",
        "walk_0.validation",
        "walk_1.train",
        "walk_1.validation",
    )


def test_dataset_serve_without_dataset_split_emits_combined_output(tmp_path):
    profile = ServeProfile.model_validate(
        {"cmd": "serve", "name": "dataset", "operation": "dataset"}
    )

    resolved = _resolve_serve(
        tmp_path / "project.yaml",
        [profile],
        runtime_operations=(DatasetTask(id="dataset"),),
    )[0]

    assert resolved.output_ids == ()


def test_dataset_preview_bypasses_default_routed_outputs(tmp_path):
    profile = ServeProfile.model_validate(
        {
            "cmd": "serve",
            "name": "dataset",
            "operation": "dataset",
            "output": {
                "transport": "fs",
                "format": "jsonl",
                "directory": "runs",
            },
        }
    )

    resolved = _resolve_serve(
        tmp_path / "project.yaml",
        [profile],
        preview="postprocess",
        split=HashSplitConfig(
            ratios={"train": 0.8, "test": 0.2},
            folds=[DatasetFold(id="default", train=["train"], test=["test"])],
        ),
        runtime_operations=(DatasetTask(id="dataset"),),
    )[0]

    assert resolved.preview == "postprocess"
    assert resolved.output_ids == ()


@pytest.mark.parametrize(
    ("preview", "expected_ids"),
    [
        ("input", ("prices", "fundamentals")),
        ("canonical", ("prices", "fundamentals")),
        ("records", ("prices", "fundamentals")),
        ("series", ("close", "open", "pe")),
        ("samples", ()),
        ("postprocess", ()),
    ],
)
def test_dataset_preview_plans_every_output_id(
    tmp_path,
    preview,
    expected_ids,
) -> None:
    streams = StreamsConfig(
        streams={
            stream_id: SourceStreamConfig.model_validate(
                {
                    "id": stream_id,
                    "from": {"source": f"{stream_id}.source"},
                    "map": {"entrypoint": "identity"},
                }
            )
            for stream_id in ("prices", "fundamentals")
        }
    )
    definition = project_definition(
        tmp_path / "project.yaml",
        dataset=DatasetConfig(
            sample=SampleConfig(cadence="1h"),
            features=[
                SeriesConfig(id="close", stream="prices", field="close"),
                SeriesConfig(id="open", stream="prices", field="open"),
                SeriesConfig(id="pe", stream="fundamentals", field="pe"),
            ],
        ),
        streams=streams,
        runtime_operations=(DatasetTask(id="dataset"),),
    )
    profile = ServeProfile.model_validate(
        {"cmd": "serve", "name": "dataset", "operation": "dataset"}
    )

    resolved = resolve_serve_profiles(
        definition,
        [profile],
        preview=preview,
        limit=None,
        cli_output=None,
    )

    assert resolved[0].output_ids == expected_ids


def test_dataset_preview_rejects_explicit_include_outputs(tmp_path):
    profile = ServeProfile.model_validate(
        {
            "cmd": "serve",
            "name": "dataset",
            "operation": "dataset",
            "include_outputs": ["default.train"],
        }
    )

    with pytest.raises(ValueError, match="cannot combine preview with include_outputs"):
        _resolve_serve(
            tmp_path / "project.yaml",
            [profile],
            preview="postprocess",
            split=HashSplitConfig(
                ratios={"train": 1.0},
                folds=[DatasetFold(id="default", train=["train"])],
            ),
            runtime_operations=(DatasetTask(id="dataset"),),
        )


def test_non_dataset_serve_profile_does_not_inherit_routed_outputs(tmp_path):
    profile = ServeProfile.model_validate(
        {"cmd": "serve", "name": "custom", "operation": "custom"}
    )

    resolved = _resolve_serve(
        tmp_path / "project.yaml",
        [profile],
        split=HashSplitConfig(
            ratios={"train": 1.0},
            folds=[DatasetFold(id="default", train=["train"])],
        ),
        runtime_operations=(
            PluginRuntimeTask(id="custom", entrypoint="plugin.runtime.custom"),
        ),
    )

    assert resolved[0].output_ids == ()


def test_inspect_profile_does_not_inherit_routed_outputs(tmp_path):
    profile = InspectProfile.model_validate(
        {"cmd": "inspect", "name": "inspection", "operation": "dataset"}
    )

    resolved = _resolve_inspect(
        tmp_path / "project.yaml",
        [profile],
        split=HashSplitConfig(
            ratios={"train": 1.0},
            folds=[DatasetFold(id="default", train=["train"])],
        ),
    )

    assert resolved[0].output_ids == ()


def test_run_profiles_resolve_include_outputs_for_fs_output(tmp_path):
    profile = ServeProfile.model_validate(
        {
            "cmd": "serve",
            "name": "dataset",
            "operation": "serve",
            "include_outputs": ["default.train", "default.validation"],
            "output": {
                "transport": "fs",
                "format": "jsonl",
                "directory": "runs",
            },
        }
    )
    resolved = _resolve_serve(
        tmp_path / "project.yaml",
        [profile],
        split=HashSplitConfig(
            ratios={"train": 0.8, "val": 0.2},
            folds=[DatasetFold(id="default", train=["train"], validation=["val"])],
        ),
    )[0]

    assert resolved.name == "dataset"
    assert resolved.operation_id == "serve"
    assert resolved.output_ids == ("default.train", "default.validation")
    assert not hasattr(resolved, "runtime")
    assert resolved.output.transport == "fs"


def test_run_profiles_reject_include_outputs_without_dataset_split(tmp_path):
    profile = ServeProfile.model_validate(
        {
            "cmd": "serve",
            "name": "dataset",
            "operation": "serve",
            "include_outputs": ["default.train"],
            "output": {
                "transport": "fs",
                "format": "jsonl",
                "directory": "runs",
            },
        }
    )

    with pytest.raises(ValueError, match="dataset split is not configured"):
        _resolve_serve(tmp_path / "project.yaml", [profile])


def test_run_profiles_reject_unknown_include_outputs(tmp_path):
    profile = ServeProfile.model_validate(
        {
            "cmd": "serve",
            "name": "dataset",
            "operation": "serve",
            "include_outputs": ["default.train", "default.test"],
            "output": {
                "transport": "fs",
                "format": "jsonl",
                "directory": "runs",
            },
        }
    )
    with pytest.raises(
        ValueError,
        match="not published by the dataset: 'default.test'",
    ):
        _resolve_serve(
            tmp_path / "project.yaml",
            [profile],
            split=HashSplitConfig(
                ratios={"train": 0.8, "val": 0.2},
                folds=[DatasetFold(id="default", train=["train"], validation=["val"])],
            ),
        )


def test_include_outputs_cannot_publish_internal_dataset_label(tmp_path):
    profile = ServeProfile.model_validate(
        {
            "cmd": "serve",
            "name": "dataset",
            "operation": "dataset",
            "include_outputs": ["default.purge"],
            "output": {
                "transport": "fs",
                "format": "jsonl",
                "directory": "runs",
            },
        }
    )
    split = TimeSplitConfig(
        intervals=[
            TimeInterval(id="train", until="2024-01-01T00:00:00Z"),
            TimeInterval(id="purge", until="2024-02-01T00:00:00Z"),
            TimeInterval(id="val"),
        ],
        folds=[DatasetFold(id="default", train=["train"], validation=["val"])],
    )

    with pytest.raises(
        ValueError,
        match="not published by the dataset: 'default.purge'",
    ):
        _resolve_serve(
            tmp_path / "project.yaml",
            [profile],
            split=split,
            runtime_operations=(DatasetTask(id="dataset"),),
        )


def test_run_profiles_reject_include_outputs_for_stdout(tmp_path):
    profile = ServeProfile.model_validate(
        {
            "cmd": "serve",
            "name": "dataset",
            "operation": "serve",
            "include_outputs": ["default.train"],
        }
    )
    with pytest.raises(ValueError, match="requires fs output"):
        _resolve_serve(
            tmp_path / "project.yaml",
            [profile],
            split=HashSplitConfig(
                ratios={"train": 1.0},
                folds=[DatasetFold(id="default", train=["train"])],
            ),
        )


def test_default_split_output_rejects_stdout(tmp_path):
    profile = ServeProfile.model_validate(
        {"cmd": "serve", "name": "dataset", "operation": "dataset"}
    )

    with pytest.raises(ValueError, match="requires fs output"):
        _resolve_serve(
            tmp_path / "project.yaml",
            [profile],
            split=HashSplitConfig(
                ratios={"train": 1.0},
                folds=[DatasetFold(id="default", train=["train"])],
            ),
            runtime_operations=(DatasetTask(id="dataset"),),
        )


def test_include_outputs_use_explicit_filename_as_base(tmp_path):
    profile = ServeProfile.model_validate(
        {
            "cmd": "serve",
            "name": "dataset",
            "operation": "serve",
            "include_outputs": ["default.train"],
            "output": {
                "transport": "fs",
                "format": "jsonl",
                "directory": "runs",
                "filename": "vectors",
            },
        }
    )
    resolved = _resolve_serve(
        tmp_path / "project.yaml",
        [profile],
        split=HashSplitConfig(
            ratios={"train": 1.0},
            folds=[DatasetFold(id="default", train=["train"])],
        ),
    )[0]

    assert (
        resolved.output.for_output("default.train").destination.name
        == "vectors.default.train.jsonl"
    )


def test_default_split_output_uses_explicit_filename_as_base(tmp_path):
    profile = ServeProfile.model_validate(
        {
            "cmd": "serve",
            "name": "dataset",
            "operation": "dataset",
            "output": {
                "transport": "fs",
                "format": "jsonl",
                "directory": "runs",
                "filename": "vectors",
            },
        }
    )

    resolved = _resolve_serve(
        tmp_path / "project.yaml",
        [profile],
        split=HashSplitConfig(
            ratios={"train": 1.0},
            folds=[DatasetFold(id="default", train=["train"])],
        ),
        runtime_operations=(DatasetTask(id="dataset"),),
    )[0]

    assert (
        resolved.output.for_output("default.train").destination.name
        == "vectors.default.train.jsonl"
    )


def test_run_profiles_leave_unconfigured_throttle_unset(tmp_path):
    profile = ServeProfile.model_validate(
        {"cmd": "serve", "name": "demo", "operation": "serve"}
    )

    resolved = _resolve_serve(tmp_path, [profile])[0]

    assert resolved.throttle_ms is None


def test_serve_profile_resolves_configured_and_cli_limits(tmp_path):
    profile = ServeProfile.model_validate(
        {
            "cmd": "serve",
            "name": "demo",
            "operation": "serve",
            "limit": 3,
        }
    )

    assert _resolve_serve(tmp_path, [profile])[0].limit == 3
    assert _resolve_serve(tmp_path, [profile], limit=7)[0].limit == 7


def test_inspect_profile_uses_cli_limit(tmp_path):
    profile = InspectProfile.model_validate(
        {"cmd": "inspect", "name": "coverage", "operation": "coverage"}
    )

    assert _resolve_inspect(tmp_path, [profile], limit=7)[0].limit == 7


def test_run_profiles_use_builtin_visuals_defaults(tmp_path):
    profile = ServeProfile.model_validate(
        {"cmd": "serve", "name": "demo", "operation": "serve"}
    )

    resolved = _resolve_serve(tmp_path, [profile])[0]

    assert resolved.observability.visuals is True


def test_run_profiles_run_visuals_override_defaults(tmp_path):
    profile = ServeProfile.model_validate(
        {
            "cmd": "serve",
            "name": "demo",
            "operation": "serve",
            "observability": {"visuals": True},
        }
    )

    resolved = _resolve_serve(tmp_path, [profile])[0]

    assert resolved.observability.visuals is True


def test_run_profiles_resolve_log_output_precedence(tmp_path):
    profile = ServeProfile.model_validate(
        {
            "cmd": "serve",
            "name": "demo",
            "operation": "serve",
            "observability": {"logging": {"outputs": [{"transport": "stdout"}]}},
        }
    )

    resolved = _resolve_serve(tmp_path / "project.yaml", [profile])[0]
    assert resolved.observability.log_output.outputs[0].transport == "stdout"
    assert resolved.observability.log_output.outputs[0].destination is None

    cli_log = tmp_path / "logs" / "cli.log"
    cli_override = _resolve_serve(
        tmp_path / "project.yaml",
        [profile],
        cli_log_outputs=[LogOutputTarget(transport="fs", destination=cli_log)],
    )[0]
    assert cli_override.observability.log_output.outputs[0].transport == "fs"
    assert cli_override.observability.log_output.outputs[0].destination == cli_log


def test_execution_scoped_logs_can_be_resolved_for_inspect_profiles(tmp_path):
    inspect_profile = InspectProfile.model_validate(
        {
            "cmd": "inspect",
            "name": "demo",
            "operation": "coverage",
            "observability": {
                "logging": {
                    "outputs": [
                        {
                            "transport": "fs",
                            "scope": "execution",
                            "path": "logs/run.log",
                        }
                    ]
                }
            },
        }
    )
    cli_output = ServeOutputConfig(
        transport="fs",
        format="jsonl",
        directory=tmp_path / "out",
    )

    resolved_inspect = _resolve_inspect(
        tmp_path / "project.yaml",
        [inspect_profile],
        cli_output=cli_output,
    )[0]
    inspect_log = resolved_inspect.observability.log_output.outputs[0]
    assert inspect_log.scope == "execution"
    assert inspect_log.destination == Path("logs/run.log")

    serve_profile = ServeProfile.model_validate(
        {
            "cmd": "serve",
            "name": "demo",
            "operation": "serve",
            "observability": {
                "logging": {
                    "outputs": [
                        {
                            "transport": "fs",
                            "scope": "execution",
                            "path": "logs/run.log",
                        }
                    ]
                }
            },
        }
    )
    resolved_serve = _resolve_serve(
        tmp_path / "project.yaml",
        [serve_profile],
        cli_output=cli_output,
    )[0]
    serve_log = resolved_serve.observability.log_output.outputs[0]
    assert resolved_serve.output.run is not None
    assert serve_log.scope == "execution"
    assert serve_log.destination == Path("logs/run.log")


def test_execution_scoped_logs_default_to_task_specific_filename(tmp_path):
    profile = ServeProfile.model_validate(
        {
            "cmd": "serve",
            "operation": "serve",
            "name": "val",
            "observability": {
                "logging": {"outputs": [{"transport": "fs", "scope": "execution"}]}
            },
        }
    )
    cli_output = ServeOutputConfig(
        transport="fs",
        format="jsonl",
        directory=tmp_path / "out",
    )

    resolved = _resolve_serve(
        tmp_path / "project.yaml",
        [profile],
        cli_output=cli_output,
    )[0]
    log_output = resolved.observability.log_output.outputs[0]
    assert resolved.output.run is not None
    assert log_output.scope == "execution"
    assert log_output.destination is None


def test_serve_runtime_profiles_share_run_and_namespace_outputs(tmp_path):
    profiles = [
        ServeProfile.model_validate(
            {
                "cmd": "serve",
                "name": name,
                "operation": "serve",
                "include_outputs": include_outputs,
            }
        )
        for name, include_outputs in (
            ("first", ["default.train"]),
            ("second", ["default.train"]),
            ("train", None),
        )
    ]
    cli_output = ServeOutputConfig(
        transport="fs",
        format="jsonl",
        directory=tmp_path / "out",
    )

    resolved = _resolve_serve(
        tmp_path / "project.yaml",
        profiles,
        cli_output=cli_output,
        split=HashSplitConfig(
            ratios={"train": 1.0},
            folds=[DatasetFold(id="default", train=["train"])],
        ),
    )

    run_ids = {
        profile.output.run.run_id
        for profile in resolved
        if profile.output.run is not None
    }
    dataset_dirs = {
        profile.output.run.dataset_dir
        for profile in resolved
        if profile.output.run is not None
    }
    assert len(run_ids) == 1
    assert len(dataset_dirs) == 1

    profiles_by_name = {profile.name: profile for profile in resolved}
    assert {profile.output.destination.name for profile in resolved} == {
        "first.jsonl",
        "second.jsonl",
        "train.jsonl",
    }
    assert {
        profiles_by_name["first"].output.for_output("default.train").destination.name,
        profiles_by_name["second"].output.for_output("default.train").destination.name,
        profiles_by_name["train"].output.destination.name,
    } == {
        "first.default.train.jsonl",
        "second.default.train.jsonl",
        "train.jsonl",
    }
    planned_run = resolved[0].output.run
    assert planned_run is not None
    assert not planned_run.serve_root.exists()
    assert not planned_run.dataset_dir.exists()
    assert not planned_run.metadata_path.exists()


def test_shared_serve_run_rejects_mixed_preview_stages_without_writes(tmp_path):
    profiles = [
        ServeProfile.model_validate(
            {
                "cmd": "serve",
                "name": name,
                "operation": "serve",
                "preview": preview,
            }
        )
        for name, preview in (("first", "input"), ("second", "canonical"))
    ]
    output_root = tmp_path / "out"

    with pytest.raises(ValueError, match="must use the same preview"):
        _resolve_serve(
            tmp_path / "project.yaml",
            profiles,
            cli_output=ServeOutputConfig(
                transport="fs",
                format="jsonl",
                directory=output_root,
            ),
        )

    assert not output_root.exists()


def test_shared_serve_runs_allow_distinct_explicit_output_filenames(tmp_path):
    profiles = [
        ServeProfile.model_validate(
            {
                "cmd": "serve",
                "name": name,
                "operation": "serve",
                "output": {
                    "transport": "fs",
                    "format": "jsonl",
                    "directory": str(tmp_path / "out"),
                    "filename": name,
                },
            }
        )
        for name in ("train", "val")
    ]

    resolved = _resolve_serve(tmp_path / "project.yaml", profiles)

    assert [profile.output.destination.name for profile in resolved] == [
        "train.jsonl",
        "val.jsonl",
    ]


def test_cli_output_directory_resolves_relative_to_workspace(tmp_path):
    overrides = build_cli_output_config(
        "fs",
        "jsonl",
        ".",
        workspace_root=tmp_path,
    )

    assert overrides is not None
    assert overrides["directory"] == tmp_path.resolve()


def test_inspect_profiles_accept_html_output_for_matrix(tmp_path):
    profile = InspectProfile.model_validate(
        {
            "cmd": "inspect",
            "name": "matrix",
            "operation": "matrix",
            "output": {
                "transport": "fs",
                "format": "html",
                "directory": str(tmp_path / "inspect"),
            },
        }
    )

    resolved = _resolve_inspect(tmp_path, [profile])[0]

    assert resolved.output.format == "html"
    assert resolved.output.transport == "fs"


def test_inspect_profile_model_allows_html_output_for_any_operation(tmp_path):
    profile = InspectProfile.model_validate(
        {
            "cmd": "inspect",
            "name": "coverage",
            "operation": "coverage",
            "output": {
                "transport": "fs",
                "format": "html",
                "directory": str(tmp_path / "inspect"),
            },
        }
    )

    assert profile.output is not None
    assert profile.output.format == "html"


def test_inspect_profiles_accept_cli_html_override_for_non_matrix(tmp_path):
    profile = InspectProfile.model_validate(
        {
            "cmd": "inspect",
            "name": "coverage",
            "operation": "coverage",
        }
    )

    resolved = _resolve_inspect(
        tmp_path,
        [profile],
        cli_output=ServeOutputConfig(
            transport="fs",
            format="html",
            directory=tmp_path / "inspect",
        ),
    )[0]

    assert resolved.output.format == "html"


def test_serve_profile_model_allows_html_output(tmp_path):
    profile = ServeProfile.model_validate(
        {
            "cmd": "serve",
            "name": "serve",
            "operation": "serve",
            "output": {
                "transport": "fs",
                "format": "html",
                "directory": str(tmp_path / "serve"),
            },
        }
    )

    assert profile.output is not None
    assert profile.output.format == "html"


def test_serve_profiles_accept_cli_txt_override(tmp_path):
    profile = ServeProfile.model_validate(
        {
            "cmd": "serve",
            "name": "serve",
            "operation": "serve",
        }
    )

    resolved = _resolve_serve(
        tmp_path,
        [profile],
        cli_output=ServeOutputConfig(
            transport="fs",
            format="txt",
            directory=tmp_path / "serve",
        ),
    )[0]

    assert resolved.output.format == "txt"


def test_build_profile_rejects_runtime_output_fields():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        BuildProfile.model_validate(
            {
                "cmd": "build",
                "name": "metadata",
                "operation": "metadata",
                "output": {
                    "transport": "fs",
                    "format": "jsonl",
                    "directory": "out",
                },
            }
        )
