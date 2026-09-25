from pathlib import Path

import pytest

from jerrythomas.artifacts import fingerprints
from jerrythomas.artifacts.fingerprints import calculate_artifact_hashes
from jerrythomas.artifacts.planning import build_artifact_graph
from jerrythomas.config.dataset.dataset import DatasetConfig, SampleConfig
from jerrythomas.config.dataset.series import (
    ScalingConfig,
    SeriesConfig,
    TargetSeriesConfig,
)
from jerrythomas.config.dataset.split import DatasetFold, TimeInterval, TimeSplitConfig
from jerrythomas.config.streams import StreamsConfig
from jerrythomas.config.tasks.base import ArtifactTask
from jerrythomas.config.tasks.coverage_stats import CoverageStatsTask
from jerrythomas.config.tasks.metadata import MetadataTask
from jerrythomas.config.tasks.scaler import ScalerTask
from jerrythomas.config.tasks.series import SeriesTask
from jerrythomas.services import config_inventory
from jerrythomas.services.project_definition import load_project_definition
from jerrythomas.services.runtime_compiler import compile_runtime
from jerrythomas.io import yaml as yaml_loader


def _write_project(root: Path) -> Path:
    for name in ("sources", "streams", "operations", "profiles", "datasets"):
        (root / name).mkdir(parents=True)
    project_yaml = root / "project.yaml"
    project_yaml.write_text(
        """schema_version: 7
artifact_revision: 1
name: snapshot
paths:
  sources: sources
  streams: streams
  operations: operations
  profiles: profiles
  datasets: datasets
  artifacts: artifacts
""",
        encoding="utf-8",
    )
    (root / "datasets" / "default.yaml").write_text(
        "version: v1\nsample: {rounding: ceil, cadence: 1h}\nfeatures: []\ntargets: []\n",
        encoding="utf-8",
    )
    return project_yaml


def test_load_project_definition_parses_each_config_document_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_yaml = _write_project(tmp_path)
    (tmp_path / "profiles" / "serve.broken.yaml").write_text(
        "this profile is intentionally: [invalid",
        encoding="utf-8",
    )
    original_load = yaml_loader.yaml.load
    parsed_documents = 0

    def count_parse(content, *, Loader):
        nonlocal parsed_documents
        parsed_documents += 1
        return original_load(content, Loader=Loader)

    monkeypatch.setattr(yaml_loader.yaml, "load", count_parse)

    definition = load_project_definition(project_yaml)

    assert definition.project.path == project_yaml.resolve()
    assert parsed_documents == 2


def test_project_definition_owns_the_canonical_artifact_graph(
    tmp_path: Path,
) -> None:
    definition = load_project_definition(_write_project(tmp_path))

    assert tuple(definition.artifact_graph.tasks_by_id) == (
        "dataset.default.scaler",
        "dataset.default.series",
        "dataset.default.metadata",
        "dataset.default.coverage_stats",
    )
    assert (
        calculate_artifact_hashes(
            definition.project,
            definition.datasets,
            definition.streams,
            definition.artifact_graph,
        )
        == definition.artifact_hashes
    )


def test_compile_runtime_uses_the_loaded_definition(tmp_path: Path) -> None:
    project_yaml = _write_project(tmp_path)
    definition = load_project_definition(project_yaml)
    project_yaml.unlink()
    (tmp_path / "datasets" / "default.yaml").unlink()

    first = compile_runtime(definition, "default")
    second = compile_runtime(definition, "default")
    first.dataset.features.append(
        SeriesConfig(id="local", stream="local", field="value")
    )

    assert first is not second
    assert first.dataset != definition.require_dataset("default")
    assert second.dataset == definition.require_dataset("default")
    assert first.dataset is not definition.require_dataset("default")
    assert second.dataset is not definition.require_dataset("default")
    assert first.dataset is not second.dataset
    assert first.streams == second.streams == {}
    assert first.streams is not second.streams
    assert definition.require_dataset("default").features == []
    assert second.dataset.features == []


def test_load_project_definition_rejects_legacy_scaler_fold_config(
    tmp_path: Path,
) -> None:
    project_yaml = _write_project(tmp_path)
    (tmp_path / "sources" / "prices.yaml").write_text(
        "id: prices\n"
        "parser: {entrypoint: core.identity}\n"
        "loader: {entrypoint: custom.loader}\n"
        "freshness: opaque\n",
        encoding="utf-8",
    )
    (tmp_path / "streams" / "prices.yaml").write_text(
        "id: prices\nfrom: {source: prices}\nmap: {entrypoint: core.identity}\n",
        encoding="utf-8",
    )
    (tmp_path / "datasets" / "default.yaml").write_text(
        "version: v1\nsample: {rounding: ceil, cadence: 1h}\n"
        "features:\n"
        "  - {id: price, stream: prices, field: value, scale: true}\n"
        "split:\n"
        "  mode: hash\n"
        "  ratios: {train: 0.8, validation: 0.2}\n"
        "  folds:\n"
        "    - {id: holdout, train: [train], validation: [validation]}\n",
        encoding="utf-8",
    )
    (tmp_path / "operations" / "scaler.yaml").write_text(
        "kind: artifact\nentrypoint: core.scaler\ndataset: default\nfolds:\n  - fit: [train]\n    apply: [train, validation]\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="folds"):
        load_project_definition(project_yaml)


def test_project_definition_keeps_resolved_environment_snapshot(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_yaml = _write_project(tmp_path)
    (tmp_path / "sources" / "prices.yaml").write_text(
        """id: prices
parser: {entrypoint: core.identity}
loader:
  transport: fs
  path: ${env:SOURCE_PATH}
  reader:
    format: jsonl
""",
        encoding="utf-8",
    )
    current_environment = {"SOURCE_PATH": "data/first.jsonl"}
    monkeypatch.setattr(
        "jerrythomas.services.project.merged_project_env",
        lambda _project_yaml: dict(current_environment),
    )
    first = load_project_definition(project_yaml)

    source = first.streams.sources["prices"]
    assert source.loader.path == "data/first.jsonl"

    current_environment["SOURCE_PATH"] = "data/second.jsonl"
    second = load_project_definition(project_yaml)

    assert first.streams.sources["prices"].loader.path == "data/first.jsonl"
    assert second.streams.sources["prices"].loader.path == "data/second.jsonl"


def test_load_project_definition_canonicalizes_symlinked_dataset_path(
    tmp_path: Path,
) -> None:
    project_yaml = _write_project(tmp_path)
    dataset = tmp_path / "datasets" / "default.yaml"
    target = tmp_path / "dataset.actual.yaml"
    dataset.rename(target)
    dataset.symlink_to(target)

    definition = load_project_definition(project_yaml)

    assert set(definition.datasets) == {"default"}
    assert definition.require_dataset("default").version == "v1"


def test_load_project_definition_canonicalizes_symlinked_config_root(
    tmp_path: Path,
) -> None:
    project_yaml = _write_project(tmp_path)
    shared = tmp_path / "shared"
    (shared / "sources").mkdir(parents=True)
    (tmp_path / "current").symlink_to(shared, target_is_directory=True)
    project_yaml.write_text(
        project_yaml.read_text(encoding="utf-8").replace(
            "sources: sources",
            "sources: current/sources",
        ),
        encoding="utf-8",
    )

    definition = load_project_definition(project_yaml)

    assert definition.project.source_dirs == ((shared / "sources").resolve(),)


def test_load_project_definition_canonicalizes_symlinked_project_path(
    tmp_path: Path,
) -> None:
    actual = tmp_path / "actual"
    project_yaml = _write_project(actual)
    (tmp_path / "current").symlink_to(actual, target_is_directory=True)

    definition = load_project_definition(tmp_path / "current" / project_yaml.name)

    assert definition.project.path == project_yaml.resolve()


def test_project_definition_keeps_retargeted_yaml_symlink_snapshot(
    tmp_path: Path,
) -> None:
    project_yaml = _write_project(tmp_path)
    first = tmp_path / "first-source.yaml"
    first.write_text(
        "id: linked\nparser: {entrypoint: core.identity}\n"
        "loader: {entrypoint: custom.loader}\n"
        "freshness: opaque\n",
        encoding="utf-8",
    )
    second = tmp_path / "second-source.yaml"
    second.write_text(
        "id: linked\nparser: {entrypoint: core.identity}\n"
        "loader: {entrypoint: other.loader}\n"
        "freshness: opaque\n",
        encoding="utf-8",
    )
    linked = tmp_path / "sources" / "linked.yaml"
    linked.symlink_to(first)

    definition = load_project_definition(project_yaml)
    assert definition.streams.sources["linked"].loader.entrypoint == "custom.loader"

    linked.unlink()
    linked.symlink_to(second)
    reloaded = load_project_definition(project_yaml)

    assert definition.streams.sources["linked"].loader.entrypoint == "custom.loader"
    assert reloaded.streams.sources["linked"].loader.entrypoint == "other.loader"


def test_pipeline_yaml_inventory_surfaces_scan_errors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def denied_walk(root, *, onerror, followlinks):
        assert root == tmp_path
        assert followlinks is False
        onerror(PermissionError("denied"))
        return ()

    monkeypatch.setattr(config_inventory.os, "walk", denied_walk)

    with pytest.raises(PermissionError, match="denied"):
        config_inventory.pipeline_yaml_files(tmp_path)


def test_pipeline_yaml_inventory_rejects_nested_symlink_directories(
    tmp_path: Path,
) -> None:
    shared = tmp_path / "shared"
    shared.mkdir()
    (tmp_path / "linked").symlink_to(shared, target_is_directory=True)

    with pytest.raises(
        ValueError,
        match="configuration directories must not be symlinks",
    ):
        config_inventory.pipeline_yaml_files(tmp_path)


def test_next_project_definition_reloads_project_dotenv(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("SOURCE_PATH", raising=False)
    project_yaml = _write_project(tmp_path)
    (tmp_path / "sources" / "prices.yaml").write_text(
        """id: prices
parser: {entrypoint: core.identity}
loader:
  transport: fs
  path: ${env:SOURCE_PATH}
  reader:
    format: jsonl
""",
        encoding="utf-8",
    )
    dotenv = tmp_path / ".env"
    dotenv.write_text("SOURCE_PATH=data/first.jsonl\n", encoding="utf-8")
    first = load_project_definition(project_yaml)

    dotenv.write_text("SOURCE_PATH=data/second.jsonl\n", encoding="utf-8")

    second = load_project_definition(project_yaml)
    assert first.streams.sources["prices"].loader.path == "data/first.jsonl"
    assert second.streams.sources["prices"].loader.path == "data/second.jsonl"


def test_project_without_name_still_validates_dataset_interpolation(
    tmp_path: Path,
) -> None:
    project_yaml = _write_project(tmp_path)
    project_yaml.write_text(
        project_yaml.read_text(encoding="utf-8").replace(
            "name: snapshot\n",
            "",
        ),
        encoding="utf-8",
    )
    (tmp_path / "datasets" / "default.yaml").write_text(
        'version: v1\nsample: {rounding: ceil, cadence: "${unknown_cadence}"}\n',
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Unknown interpolation variable 'unknown_cadence'",
    ):
        load_project_definition(project_yaml)


def test_project_without_name_still_validates_operation_interpolation(
    tmp_path: Path,
) -> None:
    project_yaml = _write_project(tmp_path)
    project_yaml.write_text(
        project_yaml.read_text(encoding="utf-8").replace(
            "name: snapshot\n",
            "",
        ),
        encoding="utf-8",
    )
    (tmp_path / "operations" / "custom.yaml").write_text(
        'kind: artifact\nentrypoint: plugin.artifact\npath: "${unknown_output}"\n',
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Unknown interpolation variable 'unknown_output'",
    ):
        load_project_definition(project_yaml)


def test_runtime_operation_change_does_not_change_artifact_hashes(
    tmp_path: Path,
) -> None:
    project_yaml = _write_project(tmp_path)
    operation = tmp_path / "operations" / "custom.yaml"
    operation.write_text(
        "kind: output\nentrypoint: plugin.runtime.custom\noptions: {threshold: 1}\n",
        encoding="utf-8",
    )
    first = load_project_definition(project_yaml)

    operation.write_text(
        "kind: output\nentrypoint: plugin.runtime.custom\noptions: {threshold: 2}\n",
        encoding="utf-8",
    )
    second = load_project_definition(project_yaml)

    assert second.runtime_operations != first.runtime_operations
    assert second.artifact_hashes == first.artifact_hashes


def test_artifact_operation_change_changes_artifact_hashes(tmp_path: Path) -> None:
    project_yaml = _write_project(tmp_path)
    operation = tmp_path / "operations" / "custom.yaml"
    operation.write_text(
        "kind: artifact\nentrypoint: plugin.artifact.custom\npath: build/first.json\n",
        encoding="utf-8",
    )
    first = load_project_definition(project_yaml)

    operation.write_text(
        "kind: artifact\nentrypoint: plugin.artifact.custom\npath: build/second.json\n",
        encoding="utf-8",
    )
    second = load_project_definition(project_yaml)

    assert second.artifact_hashes != first.artifact_hashes


def test_custom_artifact_change_does_not_change_core_artifact_hashes(
    tmp_path: Path,
) -> None:
    definition = load_project_definition(_write_project(tmp_path))
    first_task = ArtifactTask(
        id="snapshot",
        entrypoint="plugin.snapshot",
        output="build/first.json",
    )
    second_task = first_task.model_copy(update={"output": "build/second.json"})
    base_tasks = tuple(definition.artifact_graph.tasks_by_id.values())

    first = calculate_artifact_hashes(
        definition.project,
        definition.datasets,
        definition.streams,
        build_artifact_graph(
            (*base_tasks, first_task),
            definition.datasets,
            definition.streams,
        ),
    )
    second = calculate_artifact_hashes(
        definition.project,
        definition.datasets,
        definition.streams,
        build_artifact_graph(
            (*base_tasks, second_task),
            definition.datasets,
            definition.streams,
        ),
    )

    assert first.for_artifact("snapshot") != second.for_artifact("snapshot")
    for key in ("dataset.default.series", "dataset.default.metadata"):
        assert first.for_artifact(key) == second.for_artifact(key)


def test_artifact_hashing_rejects_missing_metadata_dependencies(tmp_path: Path) -> None:
    definition = load_project_definition(_write_project(tmp_path))

    with pytest.raises(
        ValueError,
        match="Required artifact operation 'dataset.default.series' is not declared",
    ):
        calculate_artifact_hashes(
            definition.project,
            definition.datasets,
            definition.streams,
            build_artifact_graph(
                (MetadataTask(id="dataset.default.metadata", dataset="default"),),
                definition.datasets,
                definition.streams,
            ),
        )


def test_artifact_hashing_rejects_missing_active_scaler(tmp_path: Path) -> None:
    project_yaml = _write_project(tmp_path)
    (tmp_path / "sources" / "prices.yaml").write_text(
        "id: prices\n"
        "parser: {entrypoint: core.identity}\n"
        "loader: {entrypoint: custom.loader}\n"
        "freshness: opaque\n",
        encoding="utf-8",
    )
    (tmp_path / "streams" / "prices.yaml").write_text(
        "id: prices\nfrom: {source: prices}\nmap: {entrypoint: core.identity}\n",
        encoding="utf-8",
    )
    definition = load_project_definition(project_yaml)
    dataset = DatasetConfig(
        sample=SampleConfig(rounding="ceil", cadence="1h"),
        features=[
            SeriesConfig(
                id="price",
                stream="prices",
                field="close",
                scale=True,
            )
        ],
    )

    with pytest.raises(
        ValueError,
        match="Required artifact operation 'dataset.default.scaler' is not declared",
    ):
        calculate_artifact_hashes(
            definition.project,
            {"default": dataset},
            definition.streams,
            build_artifact_graph(
                (SeriesTask(id="dataset.default.series", dataset="default"),),
                {"default": dataset},
                definition.streams,
            ),
        )


def test_rounding_policy_invalidates_series_scaler_and_metadata(tmp_path: Path) -> None:
    definition = load_project_definition(_write_project(tmp_path))
    streams = _single_stream_catalog()
    operations = (
        ScalerTask(id="dataset.default.scaler", dataset="default"),
        SeriesTask(id="dataset.default.series", dataset="default"),
        MetadataTask(id="dataset.default.metadata", dataset="default"),
    )
    hashes = []
    for rounding in ("floor", "ceil", "exact"):
        dataset = DatasetConfig(
            sample=SampleConfig(rounding=rounding, cadence="1h"),
            features=[
                SeriesConfig(id="price", stream="prices", field="close", scale=True)
            ],
        )
        hashes.append(
            calculate_artifact_hashes(
                definition.project,
                {"default": dataset},
                streams,
                build_artifact_graph(operations, {"default": dataset}, streams),
            )
        )
    for artifact in (
        "dataset.default.series",
        "dataset.default.scaler",
        "dataset.default.metadata",
    ):
        assert len({result.for_artifact(artifact) for result in hashes}) == 3


def test_scaling_policy_does_not_invalidate_unscaled_series(
    tmp_path: Path,
) -> None:
    definition = load_project_definition(_write_project(tmp_path))
    streams = _single_stream_catalog()
    operations = (
        ScalerTask(id="dataset.default.scaler", dataset="default"),
        SeriesTask(id="dataset.default.series", dataset="default"),
    )
    unscaled = DatasetConfig(
        sample=SampleConfig(rounding="ceil", cadence="1h"),
        features=[SeriesConfig(id="price", stream="prices", field="close")],
    )
    scaled = unscaled.model_copy(
        update={
            "features": [
                unscaled.features[0].model_copy(update={"scale": ScalingConfig()})
            ]
        }
    )

    unscaled_hashes = calculate_artifact_hashes(
        definition.project,
        {"default": unscaled},
        streams,
        build_artifact_graph(operations, {"default": unscaled}, streams),
    )
    scaled_hashes = calculate_artifact_hashes(
        definition.project,
        {"default": scaled},
        streams,
        build_artifact_graph(operations, {"default": scaled}, streams),
    )

    assert scaled_hashes.for_artifact("dataset.default.series") == (
        unscaled_hashes.for_artifact("dataset.default.series")
    )
    assert scaled_hashes.for_artifact("dataset.default.scaler") != (
        unscaled_hashes.for_artifact("dataset.default.scaler")
    )


@pytest.mark.parametrize("role", ["features", "targets"])
@pytest.mark.parametrize(
    "settings",
    [{"with_mean": False}, {"with_std": False}, {"epsilon": 0.25}],
)
def test_feature_and_target_scaling_settings_only_invalidate_scaler(
    tmp_path: Path, role: str, settings: dict[str, object]
) -> None:
    definition = load_project_definition(_write_project(tmp_path))
    streams = _single_stream_catalog()
    operations = (
        ScalerTask(id="dataset.default.scaler", dataset="default"),
        SeriesTask(id="dataset.default.series", dataset="default"),
        MetadataTask(id="dataset.default.metadata", dataset="default"),
    )
    dataset = DatasetConfig.model_validate(
        {
            "sample": {"rounding": "ceil", "cadence": "1h"},
            "features": [
                {"id": "price", "stream": "prices", "field": "close", "scale": True},
            ],
            "targets": [
                {
                    "id": "target",
                    "stream": "prices",
                    "field": "close",
                    "horizon": "0h",
                    "scale": True,
                },
            ],
        }
    )
    changed_config = dataset.model_dump(mode="json")
    changed_config[role][0]["scale"] = settings
    changed = DatasetConfig.model_validate(changed_config)
    hashes = [
        calculate_artifact_hashes(
            definition.project,
            {"default": selected},
            streams,
            build_artifact_graph(operations, {"default": selected}, streams),
        )
        for selected in (dataset, changed)
    ]
    for key in ("series", "metadata"):
        assert hashes[0].for_artifact(f"dataset.default.{key}") == (
            hashes[1].for_artifact(f"dataset.default.{key}")
        )
    assert hashes[0].for_artifact("dataset.default.scaler") != (
        hashes[1].for_artifact("dataset.default.scaler")
    )


def test_collection_policy_invalidates_series_but_not_scaler(
    tmp_path: Path,
) -> None:
    definition = load_project_definition(_write_project(tmp_path))
    streams = _single_stream_catalog()
    operations = (
        ScalerTask(id="dataset.default.scaler", dataset="default"),
        SeriesTask(id="dataset.default.series", dataset="default"),
        MetadataTask(id="dataset.default.metadata", dataset="default"),
        CoverageStatsTask(id="dataset.default.coverage_stats", dataset="default"),
    )
    feature = SeriesConfig(
        id="price",
        stream="prices",
        field="close",
        scale=True,
    )
    scalar = DatasetConfig(
        sample=SampleConfig(rounding="ceil", cadence="1h"),
        features=[feature],
    )
    collected = scalar.model_copy(
        update={
            "features": [
                feature.model_copy(update={"collect": 2}),
            ]
        }
    )

    scalar_hashes = calculate_artifact_hashes(
        definition.project,
        {"default": scalar},
        streams,
        build_artifact_graph(operations, {"default": scalar}, streams),
    )
    collected_hashes = calculate_artifact_hashes(
        definition.project,
        {"default": collected},
        streams,
        build_artifact_graph(operations, {"default": collected}, streams),
    )

    assert collected_hashes.for_artifact("dataset.default.scaler") == (
        scalar_hashes.for_artifact("dataset.default.scaler")
    )
    for artifact in (
        "dataset.default.series",
        "dataset.default.metadata",
        "dataset.default.coverage_stats",
    ):
        assert collected_hashes.for_artifact(artifact) != (
            scalar_hashes.for_artifact(artifact)
        )


def test_scaler_and_metadata_hashes_track_every_fold_role(tmp_path: Path) -> None:
    definition = load_project_definition(_write_project(tmp_path))
    streams = _wide_stream_catalog()
    feature = SeriesConfig(
        id="price",
        stream="prices",
        field="close",
        scale=True,
    )
    split = TimeSplitConfig(
        intervals=[
            TimeInterval(id="train", until="2024-02-01T00:00:00Z"),
            TimeInterval(id="validation", until="2024-03-01T00:00:00Z"),
            TimeInterval(id="test"),
        ],
        folds=[
            DatasetFold(
                id="holdout",
                train=["train"],
            )
        ],
    )
    baseline = DatasetConfig(
        sample=SampleConfig(rounding="ceil", cadence="1h"),
        features=[feature],
        split=split,
    )
    validation_changed = baseline.model_copy(
        update={
            "split": split.model_copy(
                update={
                    "folds": [
                        DatasetFold(
                            id="holdout",
                            train=["train"],
                            validation=["validation"],
                        )
                    ]
                }
            )
        }
    )
    test_changed = baseline.model_copy(
        update={
            "split": split.model_copy(
                update={
                    "folds": [
                        DatasetFold(
                            id="holdout",
                            train=["train"],
                            test=["test"],
                        )
                    ]
                }
            )
        }
    )
    training_changed = baseline.model_copy(
        update={
            "split": split.model_copy(
                update={
                    "folds": [
                        DatasetFold(
                            id="holdout",
                            train=["train", "validation"],
                            test=["test"],
                        )
                    ]
                }
            )
        }
    )

    def artifact_hashes(dataset: DatasetConfig):
        graph = build_artifact_graph(
            (
                ScalerTask(id="dataset.default.scaler", dataset="default"),
                SeriesTask(id="dataset.default.series", dataset="default"),
                MetadataTask(id="dataset.default.metadata", dataset="default"),
            ),
            {"default": dataset},
            streams,
        )
        return calculate_artifact_hashes(
            definition.project,
            {"default": dataset},
            streams,
            graph,
        )

    baseline_hashes = artifact_hashes(baseline)
    for changed in (validation_changed, test_changed, training_changed):
        changed_hashes = artifact_hashes(changed)
        assert changed_hashes.for_artifact("dataset.default.scaler") != (
            baseline_hashes.for_artifact("dataset.default.scaler")
        )
        assert changed_hashes.for_artifact("dataset.default.metadata") != (
            baseline_hashes.for_artifact("dataset.default.metadata")
        )
        assert changed_hashes.for_artifact(
            "dataset.default.series"
        ) == baseline_hashes.for_artifact("dataset.default.series")


def test_target_horizon_changes_folded_scaler_and_metadata_but_not_series(
    tmp_path: Path,
) -> None:
    definition = load_project_definition(_write_project(tmp_path))
    streams = _wide_stream_catalog()
    split = TimeSplitConfig(
        intervals=[
            TimeInterval(id="train", until="2024-02-01T00:00:00Z"),
            TimeInterval(id="validation"),
        ],
        folds=[
            DatasetFold(
                id="holdout",
                train=["train"],
                validation=["validation"],
            )
        ],
    )
    feature = SeriesConfig(
        id="price",
        stream="prices",
        field="close",
        scale=True,
    )
    baseline_target = TargetSeriesConfig(
        id="return",
        stream="prices",
        field="future_return",
        horizon="1d",
    )
    baseline = DatasetConfig(
        sample=SampleConfig(rounding="ceil", cadence="1h"),
        features=[feature],
        targets=[baseline_target],
        split=split,
    )
    changed = baseline.model_copy(
        update={
            "targets": [
                baseline_target.model_copy(update={"horizon": "2d"}),
            ]
        }
    )
    equivalent = baseline.model_copy(
        update={
            "targets": [
                baseline_target.model_copy(update={"horizon": "24h"}),
            ]
        }
    )
    tasks = (
        ScalerTask(id="dataset.default.scaler", dataset="default"),
        SeriesTask(id="dataset.default.series", dataset="default"),
        MetadataTask(id="dataset.default.metadata", dataset="default"),
    )

    baseline_hashes = calculate_artifact_hashes(
        definition.project,
        {"default": baseline},
        streams,
        build_artifact_graph(tasks, {"default": baseline}, streams),
    )
    changed_hashes = calculate_artifact_hashes(
        definition.project,
        {"default": changed},
        streams,
        build_artifact_graph(tasks, {"default": changed}, streams),
    )
    equivalent_hashes = calculate_artifact_hashes(
        definition.project,
        {"default": equivalent},
        streams,
        build_artifact_graph(tasks, {"default": equivalent}, streams),
    )

    assert changed_hashes.for_artifact("dataset.default.scaler") != (
        baseline_hashes.for_artifact("dataset.default.scaler")
    )
    assert changed_hashes.for_artifact(
        "dataset.default.series"
    ) == baseline_hashes.for_artifact("dataset.default.series")
    assert changed_hashes.for_artifact("dataset.default.metadata") != (
        baseline_hashes.for_artifact("dataset.default.metadata")
    )
    assert equivalent_hashes == baseline_hashes


def test_window_mode_rebuilds_metadata_dependents_but_not_series_or_scaler(
    tmp_path: Path,
) -> None:
    definition = load_project_definition(_write_project(tmp_path))
    streams = _single_stream_catalog()
    baseline = DatasetConfig(
        sample=SampleConfig(rounding="ceil", cadence="1h"),
        features=[
            SeriesConfig(
                id="price",
                stream="prices",
                field="close",
                scale=True,
            )
        ],
    )
    changed = baseline.model_copy(
        update={"sample": baseline.sample.model_copy(update={"window_mode": "union"})}
    )
    tasks = (
        ScalerTask(id="dataset.default.scaler", dataset="default"),
        SeriesTask(id="dataset.default.series", dataset="default"),
        MetadataTask(id="dataset.default.metadata", dataset="default"),
        CoverageStatsTask(id="dataset.default.coverage_stats", dataset="default"),
    )

    baseline_hashes = calculate_artifact_hashes(
        definition.project,
        {"default": baseline},
        streams,
        build_artifact_graph(tasks, {"default": baseline}, streams),
    )
    changed_hashes = calculate_artifact_hashes(
        definition.project,
        {"default": changed},
        streams,
        build_artifact_graph(tasks, {"default": changed}, streams),
    )

    assert changed_hashes.for_artifact("dataset.default.scaler") == (
        baseline_hashes.for_artifact("dataset.default.scaler")
    )
    assert changed_hashes.for_artifact(
        "dataset.default.series"
    ) == baseline_hashes.for_artifact("dataset.default.series")
    assert changed_hashes.for_artifact("dataset.default.metadata") != (
        baseline_hashes.for_artifact("dataset.default.metadata")
    )
    assert changed_hashes.for_artifact("dataset.default.coverage_stats") != (
        baseline_hashes.for_artifact("dataset.default.coverage_stats")
    )


def test_target_horizon_does_not_change_standard_scaler_fingerprint(
    tmp_path: Path,
) -> None:
    definition = load_project_definition(_write_project(tmp_path))
    streams = _single_stream_catalog()
    target = TargetSeriesConfig(
        id="return",
        stream="prices",
        field="future_return",
        scale=True,
        horizon="1d",
    )
    baseline = DatasetConfig(
        sample=SampleConfig(rounding="ceil", cadence="1h"),
        features=[SeriesConfig(id="price", stream="prices", field="close")],
        targets=[target],
    )
    changed = baseline.model_copy(
        update={
            "targets": [
                target.model_copy(update={"horizon": "2d"}),
            ]
        }
    )

    baseline_hash = calculate_artifact_hashes(
        definition.project,
        {"default": baseline},
        streams,
        build_artifact_graph(
            (ScalerTask(id="dataset.default.scaler", dataset="default"),),
            {"default": baseline},
            streams,
        ),
    ).for_artifact("dataset.default.scaler")
    changed_hash = calculate_artifact_hashes(
        definition.project,
        {"default": changed},
        streams,
        build_artifact_graph(
            (ScalerTask(id="dataset.default.scaler", dataset="default"),),
            {"default": changed},
            streams,
        ),
    ).for_artifact("dataset.default.scaler")

    assert changed_hash == baseline_hash


def _single_stream_catalog() -> StreamsConfig:
    return StreamsConfig.model_validate(
        {
            "sources": {
                "raw": {
                    "id": "raw",
                    "parser": {"entrypoint": "parse"},
                    "loader": {"entrypoint": "custom.loader"},
                    "freshness": "opaque",
                }
            },
            "streams": {
                "prices": {
                    "id": "prices",
                    "from": {"source": "raw"},
                    "map": {"entrypoint": "map"},
                }
            },
        }
    )


def _wide_stream_catalog() -> StreamsConfig:
    return StreamsConfig.model_validate(
        {
            "sources": {
                "raw": {
                    "id": "raw",
                    "parser": {"entrypoint": "parse"},
                    "loader": {"entrypoint": "custom.loader"},
                    "freshness": "opaque",
                }
            },
            "streams": {
                "prices": {
                    "id": "prices",
                    "from": {"source": "raw"},
                    "map": {"entrypoint": "map"},
                    "partition_by": ["metric"],
                }
            },
        }
    )


@pytest.mark.parametrize(
    ("source_format", "suffix"),
    [("jsonl", ".jsonl"), ("parquet", ".parquet")],
)
def test_core_artifact_hashes_track_only_referenced_source_closure(
    tmp_path: Path,
    source_format: str,
    suffix: str,
) -> None:
    definition = load_project_definition(_write_project(tmp_path))
    data = tmp_path / "data"
    data.mkdir()
    used = data / f"used{suffix}"
    unused = data / f"unused{suffix}"
    used.write_text("{}\n", encoding="utf-8")
    unused.write_text("{}\n", encoding="utf-8")
    dataset = DatasetConfig(
        sample=SampleConfig(rounding="ceil", cadence="1h"),
        features=[SeriesConfig(id="price", stream="used", field="close")],
    )
    streams = StreamsConfig.model_validate(
        {
            "sources": {
                source_id: {
                    "id": source_id,
                    "parser": {"entrypoint": "parse"},
                    "loader": {
                        "transport": "fs",
                        "path": f"data/{source_id}{suffix}",
                        "reader": {"format": source_format},
                    },
                }
                for source_id in ("used", "unused")
            },
            "streams": {
                source_id: {
                    "id": source_id,
                    "from": {"source": source_id},
                    "map": {"entrypoint": "map"},
                }
                for source_id in ("used", "unused")
            },
        }
    )
    graph = build_artifact_graph(
        definition.artifact_graph.tasks_by_id.values(),
        {"default": dataset},
        streams,
    )

    baseline = calculate_artifact_hashes(
        definition.project,
        {"default": dataset},
        streams,
        graph,
    )
    unused.write_text("unused changed\n", encoding="utf-8")
    after_unused_change = calculate_artifact_hashes(
        definition.project,
        {"default": dataset},
        streams,
        graph,
    )
    assert after_unused_change.for_artifact(
        "dataset.default.series"
    ) == baseline.for_artifact("dataset.default.series")

    used.write_text("used changed\n", encoding="utf-8")
    after_used_change = calculate_artifact_hashes(
        definition.project,
        {"default": dataset},
        streams,
        graph,
    )
    for key in ("dataset.default.series", "dataset.default.metadata"):
        assert after_used_change.for_artifact(key) != baseline.for_artifact(key)


def test_cross_section_config_changes_series_artifact_hash(tmp_path: Path) -> None:
    project_yaml = _write_project(tmp_path)
    (tmp_path / "sources" / "signals.yaml").write_text(
        "id: signals\n"
        "parser: {entrypoint: core.identity}\n"
        "loader: {entrypoint: custom.loader}\n"
        "freshness: opaque\n",
        encoding="utf-8",
    )
    (tmp_path / "streams" / "signals.yaml").write_text(
        "id: signals\n"
        "from: {source: signals}\n"
        "map: {entrypoint: core.identity}\n"
        "partition_by: [ticker]\n",
        encoding="utf-8",
    )
    ranked_stream = tmp_path / "streams" / "ranked.yaml"
    ranked_stream.write_text(
        "id: ranked\n"
        "from: {stream: signals}\n"
        "cross_section:\n"
        "  - {operation: rank_score, field: value, to: rank, min_samples: 2}\n",
        encoding="utf-8",
    )
    (tmp_path / "datasets" / "default.yaml").write_text(
        "version: v1\nsample: {rounding: ceil, cadence: 1h, keys: [ticker]}\n"
        "features:\n"
        "  - {id: rank, stream: ranked, field: rank}\n",
        encoding="utf-8",
    )
    first = load_project_definition(project_yaml)

    ranked_stream.write_text(
        ranked_stream.read_text(encoding="utf-8").replace(
            "min_samples: 2",
            "min_samples: 3",
        ),
        encoding="utf-8",
    )
    second = load_project_definition(project_yaml)

    assert first.artifact_hashes.for_artifact("dataset.default.series") != (
        second.artifact_hashes.for_artifact("dataset.default.series")
    )


def test_artifact_operation_comment_does_not_change_artifact_hashes(
    tmp_path: Path,
) -> None:
    project_yaml = _write_project(tmp_path)
    operation = tmp_path / "operations" / "custom.yaml"
    operation.write_text(
        "kind: artifact\nentrypoint: plugin.artifact.custom\npath: build/custom.json\n",
        encoding="utf-8",
    )
    first = load_project_definition(project_yaml)

    operation.write_text(
        operation.read_text(encoding="utf-8") + "# Documentation only.\n",
        encoding="utf-8",
    )
    second = load_project_definition(project_yaml)

    assert second.artifact_hashes == first.artifact_hashes


def test_artifact_revision_change_changes_artifact_hashes(tmp_path: Path) -> None:
    project_yaml = _write_project(tmp_path)
    first = load_project_definition(project_yaml)

    project_yaml.write_text(
        project_yaml.read_text(encoding="utf-8").replace(
            "artifact_revision: 1\n",
            "artifact_revision: 2\n",
        ),
        encoding="utf-8",
    )
    second = load_project_definition(project_yaml)

    assert second.artifact_hashes != first.artifact_hashes


def test_hash_split_ratio_order_does_not_change_artifact_hash(tmp_path: Path) -> None:
    project_yaml = _write_project(tmp_path)
    dataset = tmp_path / "datasets" / "default.yaml"
    dataset.write_text(
        """\
version: v1
sample: {rounding: ceil, cadence: 1h}
features: []
targets: []
split:
  mode: hash
  ratios: {train: 0.8, test: 0.2}
  folds:
    - {id: holdout, train: [train], test: [test]}
""",
        encoding="utf-8",
    )
    first = load_project_definition(project_yaml)

    dataset.write_text(
        """\
version: v1
sample: {rounding: ceil, cadence: 1h}
features: []
targets: []
split:
  mode: hash
  ratios: {test: 0.2, train: 0.8}
  folds:
    - {id: holdout, train: [train], test: [test]}
""",
        encoding="utf-8",
    )
    second = load_project_definition(project_yaml)

    assert first.artifact_hashes == second.artifact_hashes


def test_artifact_cache_version_changes_artifact_hash(
    tmp_path: Path,
    monkeypatch,
) -> None:
    definition = load_project_definition(_write_project(tmp_path))
    monkeypatch.setattr(
        fingerprints,
        "ARTIFACT_CACHE_VERSION",
        fingerprints.ARTIFACT_CACHE_VERSION + 1,
    )
    changed_artifact_hashes = calculate_artifact_hashes(
        definition.project,
        definition.datasets,
        definition.streams,
        definition.artifact_graph,
    )

    assert changed_artifact_hashes != definition.artifact_hashes


def test_metadata_format_version_invalidates_only_metadata_and_dependents(
    tmp_path: Path,
    monkeypatch,
) -> None:
    definition = load_project_definition(_write_project(tmp_path))
    tasks = (
        SeriesTask(id="dataset.default.series", dataset="default"),
        MetadataTask(id="dataset.default.metadata", dataset="default"),
    )
    graph = build_artifact_graph(tasks, definition.datasets, definition.streams)
    current = calculate_artifact_hashes(
        definition.project,
        definition.datasets,
        definition.streams,
        graph,
    )

    monkeypatch.setattr(
        fingerprints,
        "VECTOR_METADATA_VERSION",
        fingerprints.VECTOR_METADATA_VERSION + 1,
    )
    changed = calculate_artifact_hashes(
        definition.project,
        definition.datasets,
        definition.streams,
        graph,
    )

    assert changed.for_artifact("dataset.default.series") == current.for_artifact(
        "dataset.default.series"
    )
    assert changed.for_artifact("dataset.default.metadata") != current.for_artifact(
        "dataset.default.metadata"
    )


def test_scaler_format_version_invalidates_only_scaler(
    tmp_path: Path,
    monkeypatch,
) -> None:
    definition = load_project_definition(_write_project(tmp_path))
    dataset = DatasetConfig(
        sample=SampleConfig(rounding="ceil", cadence="1h"),
        features=[
            SeriesConfig(
                id="price",
                stream="prices",
                field="close",
                scale=True,
            )
        ],
    )
    tasks = (
        ScalerTask(id="dataset.default.scaler", dataset="default"),
        SeriesTask(id="dataset.default.series", dataset="default"),
    )
    streams = _single_stream_catalog()
    graph = build_artifact_graph(tasks, {"default": dataset}, streams)
    current = calculate_artifact_hashes(
        definition.project,
        {"default": dataset},
        streams,
        graph,
    )

    monkeypatch.setattr(
        fingerprints,
        "SCALER_ARTIFACT_VERSION",
        fingerprints.SCALER_ARTIFACT_VERSION + 1,
    )
    changed = calculate_artifact_hashes(
        definition.project,
        {"default": dataset},
        streams,
        graph,
    )

    assert changed.for_artifact("dataset.default.scaler") != current.for_artifact(
        "dataset.default.scaler"
    )
    assert changed.for_artifact("dataset.default.series") == current.for_artifact(
        "dataset.default.series"
    )
