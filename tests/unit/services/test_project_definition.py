from datetime import datetime, timezone
from pathlib import Path

import pytest

from datapipeline.artifacts import fingerprints
from datapipeline.artifacts.fingerprints import calculate_artifact_hashes
from datapipeline.artifacts.specs import (
    SCALER_STATISTICS,
    SERIES,
    VECTOR_METADATA,
    COVERAGE_STATS,
)
from datapipeline.config.dataset.dataset import DatasetConfig, SampleConfig
from datapipeline.config.dataset.series import SeriesConfig, TargetSeriesConfig
from datapipeline.config.dataset.split import DatasetFold, TimeInterval, TimeSplitConfig
from datapipeline.config.streams import StreamsConfig
from datapipeline.config.tasks.base import ArtifactTask
from datapipeline.config.tasks.coverage_stats import CoverageStatsTask
from datapipeline.config.tasks.metadata import MetadataTask
from datapipeline.config.tasks.scaler import ScalerTask
from datapipeline.config.tasks.series import SeriesTask
from datapipeline.services import config_inventory
from datapipeline.services.project_definition import load_project_definition
from datapipeline.services.runtime_compiler import compile_runtime
from datapipeline.utils import load as yaml_loader


def _write_project(root: Path) -> Path:
    for name in ("sources", "streams", "operations", "profiles"):
        (root / name).mkdir(parents=True)
    project_yaml = root / "project.yaml"
    project_yaml.write_text(
        """schema_version: 4
artifact_revision: 1
name: snapshot
paths:
  sources: sources
  streams: streams
  operations: operations
  profiles: profiles
  dataset: dataset.yaml
  artifacts: artifacts
""",
        encoding="utf-8",
    )
    (root / "dataset.yaml").write_text(
        "sample: {cadence: 1h}\nfeatures: []\ntargets: []\n",
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


def test_compile_runtime_uses_the_loaded_definition(tmp_path: Path) -> None:
    project_yaml = _write_project(tmp_path)
    definition = load_project_definition(project_yaml)
    project_yaml.unlink()
    (tmp_path / "dataset.yaml").unlink()

    first = compile_runtime(definition)
    second = compile_runtime(definition)
    first.window_bounds = (
        datetime(2020, 1, 1, tzinfo=timezone.utc),
        datetime(2021, 1, 1, tzinfo=timezone.utc),
    )
    first.dataset.features.append(
        SeriesConfig(id="local", stream="local", field="value")
    )

    assert first is not second
    assert first.dataset != definition.dataset
    assert second.dataset == definition.dataset
    assert first.dataset is not definition.dataset
    assert second.dataset is not definition.dataset
    assert first.dataset is not second.dataset
    assert first.streams == second.streams == {}
    assert first.streams is not second.streams
    assert definition.dataset.features == []
    assert second.dataset.features == []
    assert second.window_bounds is None


def test_load_project_definition_rejects_legacy_scaler_fold_config(
    tmp_path: Path,
) -> None:
    project_yaml = _write_project(tmp_path)
    (tmp_path / "sources" / "prices.yaml").write_text(
        "id: prices\n"
        "parser: {entrypoint: identity}\n"
        "loader: {entrypoint: custom.loader}\n",
        encoding="utf-8",
    )
    (tmp_path / "streams" / "prices.yaml").write_text(
        "id: prices\nfrom: {source: prices}\nmap: {entrypoint: identity}\n",
        encoding="utf-8",
    )
    (tmp_path / "dataset.yaml").write_text(
        "sample: {cadence: 1h}\n"
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
        "folds:\n  - fit: [train]\n    apply: [train, validation]\n",
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
parser: {entrypoint: identity}
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
        "datapipeline.services.project.merged_project_env",
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
    dataset = tmp_path / "dataset.yaml"
    target = tmp_path / "dataset.actual.yaml"
    dataset.rename(target)
    dataset.symlink_to(target)

    definition = load_project_definition(project_yaml)

    assert definition.project.dataset_path == target.resolve()


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
        "id: linked\nparser: {entrypoint: identity}\n"
        "loader: {entrypoint: custom.loader}\n",
        encoding="utf-8",
    )
    second = tmp_path / "second-source.yaml"
    second.write_text(
        "id: linked\nparser: {entrypoint: identity}\n"
        "loader: {entrypoint: other.loader}\n",
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
parser: {entrypoint: identity}
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
    (tmp_path / "dataset.yaml").write_text(
        'sample: {cadence: "${unknown_cadence}"}\n',
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
        'kind: artifact\nentrypoint: plugin.artifact\noutput: "${unknown_output}"\n',
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
        "kind: runtime\nentrypoint: plugin.runtime.custom\noptions: {threshold: 1}\n",
        encoding="utf-8",
    )
    first = load_project_definition(project_yaml)

    operation.write_text(
        "kind: runtime\nentrypoint: plugin.runtime.custom\noptions: {threshold: 2}\n",
        encoding="utf-8",
    )
    second = load_project_definition(project_yaml)

    assert second.runtime_operations != first.runtime_operations
    assert second.artifact_hashes == first.artifact_hashes


def test_artifact_operation_change_changes_artifact_hashes(tmp_path: Path) -> None:
    project_yaml = _write_project(tmp_path)
    operation = tmp_path / "operations" / "custom.yaml"
    operation.write_text(
        "kind: artifact\n"
        "entrypoint: plugin.artifact.custom\n"
        "output: build/first.json\n",
        encoding="utf-8",
    )
    first = load_project_definition(project_yaml)

    operation.write_text(
        "kind: artifact\n"
        "entrypoint: plugin.artifact.custom\n"
        "output: build/second.json\n",
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

    first = calculate_artifact_hashes(
        definition.project,
        definition.dataset,
        definition.streams,
        (*definition.artifact_operations, first_task),
    )
    second = calculate_artifact_hashes(
        definition.project,
        definition.dataset,
        definition.streams,
        (*definition.artifact_operations, second_task),
    )

    assert first.for_artifact("snapshot") != second.for_artifact("snapshot")
    for key in (SERIES, VECTOR_METADATA):
        assert first.for_artifact(key) == second.for_artifact(key)


def test_artifact_hashing_rejects_missing_metadata_dependencies(tmp_path: Path) -> None:
    definition = load_project_definition(_write_project(tmp_path))

    with pytest.raises(
        ValueError,
        match="Required artifact operation 'series' is not declared",
    ):
        calculate_artifact_hashes(
            definition.project,
            definition.dataset,
            definition.streams,
            (MetadataTask(),),
        )


def test_artifact_hashing_rejects_missing_active_scaler(tmp_path: Path) -> None:
    project_yaml = _write_project(tmp_path)
    (tmp_path / "sources" / "prices.yaml").write_text(
        "id: prices\n"
        "parser: {entrypoint: identity}\n"
        "loader: {entrypoint: custom.loader}\n",
        encoding="utf-8",
    )
    (tmp_path / "streams" / "prices.yaml").write_text(
        "id: prices\nfrom: {source: prices}\nmap: {entrypoint: identity}\n",
        encoding="utf-8",
    )
    definition = load_project_definition(project_yaml)
    dataset = DatasetConfig(
        sample=SampleConfig(cadence="1h"),
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
        match="Required artifact operation 'scaler' is not declared",
    ):
        calculate_artifact_hashes(
            definition.project,
            dataset,
            definition.streams,
            (SeriesTask(),),
        )


def test_scaling_policy_does_not_invalidate_unscaled_series(
    tmp_path: Path,
) -> None:
    definition = load_project_definition(_write_project(tmp_path))
    streams = _single_stream_catalog()
    operations = (ScalerTask(), SeriesTask())
    unscaled = DatasetConfig(
        sample=SampleConfig(cadence="1h"),
        features=[SeriesConfig(id="price", stream="prices", field="close")],
    )
    scaled = unscaled.model_copy(
        update={"features": [unscaled.features[0].model_copy(update={"scale": True})]}
    )

    unscaled_hashes = calculate_artifact_hashes(
        definition.project,
        unscaled,
        streams,
        operations,
    )
    scaled_hashes = calculate_artifact_hashes(
        definition.project,
        scaled,
        streams,
        operations,
    )

    assert scaled_hashes.for_artifact(SERIES) == (unscaled_hashes.for_artifact(SERIES))
    assert scaled_hashes.for_artifact(SCALER_STATISTICS) != (
        unscaled_hashes.for_artifact(SCALER_STATISTICS)
    )


def test_collection_policy_invalidates_series_but_not_scaler(
    tmp_path: Path,
) -> None:
    definition = load_project_definition(_write_project(tmp_path))
    streams = _single_stream_catalog()
    operations = (
        ScalerTask(),
        SeriesTask(),
        MetadataTask(),
        CoverageStatsTask(),
    )
    feature = SeriesConfig(
        id="price",
        stream="prices",
        field="close",
        scale=True,
    )
    scalar = DatasetConfig(
        sample=SampleConfig(cadence="1h"),
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
        scalar,
        streams,
        operations,
    )
    collected_hashes = calculate_artifact_hashes(
        definition.project,
        collected,
        streams,
        operations,
    )

    assert collected_hashes.for_artifact(SCALER_STATISTICS) == (
        scalar_hashes.for_artifact(SCALER_STATISTICS)
    )
    for artifact in (SERIES, VECTOR_METADATA, COVERAGE_STATS):
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
        sample=SampleConfig(cadence="1h"),
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
        return calculate_artifact_hashes(
            definition.project,
            dataset,
            streams,
            (ScalerTask(), SeriesTask(), MetadataTask()),
        )

    baseline_hashes = artifact_hashes(baseline)
    for changed in (validation_changed, test_changed, training_changed):
        changed_hashes = artifact_hashes(changed)
        assert changed_hashes.for_artifact(SCALER_STATISTICS) != (
            baseline_hashes.for_artifact(SCALER_STATISTICS)
        )
        assert changed_hashes.for_artifact(VECTOR_METADATA) != (
            baseline_hashes.for_artifact(VECTOR_METADATA)
        )
        assert changed_hashes.for_artifact(SERIES) == baseline_hashes.for_artifact(
            SERIES
        )


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
        sample=SampleConfig(cadence="1h"),
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
    tasks = (ScalerTask(), SeriesTask(), MetadataTask())

    baseline_hashes = calculate_artifact_hashes(
        definition.project,
        baseline,
        streams,
        tasks,
    )
    changed_hashes = calculate_artifact_hashes(
        definition.project,
        changed,
        streams,
        tasks,
    )
    equivalent_hashes = calculate_artifact_hashes(
        definition.project,
        equivalent,
        streams,
        tasks,
    )

    assert changed_hashes.for_artifact(SCALER_STATISTICS) != (
        baseline_hashes.for_artifact(SCALER_STATISTICS)
    )
    assert changed_hashes.for_artifact(SERIES) == baseline_hashes.for_artifact(SERIES)
    assert changed_hashes.for_artifact(VECTOR_METADATA) != (
        baseline_hashes.for_artifact(VECTOR_METADATA)
    )
    assert equivalent_hashes == baseline_hashes


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
        sample=SampleConfig(cadence="1h"),
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
        baseline,
        streams,
        (ScalerTask(),),
    ).for_artifact(SCALER_STATISTICS)
    changed_hash = calculate_artifact_hashes(
        definition.project,
        changed,
        streams,
        (ScalerTask(),),
    ).for_artifact(SCALER_STATISTICS)

    assert changed_hash == baseline_hash


def _single_stream_catalog() -> StreamsConfig:
    return StreamsConfig.model_validate(
        {
            "sources": {
                "raw": {
                    "id": "raw",
                    "parser": {"entrypoint": "parse"},
                    "loader": {"entrypoint": "custom.loader"},
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
        sample=SampleConfig(cadence="1h"),
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

    baseline = calculate_artifact_hashes(
        definition.project,
        dataset,
        streams,
        definition.artifact_operations,
    )
    unused.write_text("unused changed\n", encoding="utf-8")
    after_unused_change = calculate_artifact_hashes(
        definition.project,
        dataset,
        streams,
        definition.artifact_operations,
    )
    assert after_unused_change.for_artifact(SERIES) == baseline.for_artifact(SERIES)

    used.write_text("used changed\n", encoding="utf-8")
    after_used_change = calculate_artifact_hashes(
        definition.project,
        dataset,
        streams,
        definition.artifact_operations,
    )
    for key in (SERIES, VECTOR_METADATA):
        assert after_used_change.for_artifact(key) != baseline.for_artifact(key)


def test_artifact_operation_comment_does_not_change_artifact_hashes(
    tmp_path: Path,
) -> None:
    project_yaml = _write_project(tmp_path)
    operation = tmp_path / "operations" / "custom.yaml"
    operation.write_text(
        "kind: artifact\n"
        "entrypoint: plugin.artifact.custom\n"
        "output: build/custom.json\n",
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
    dataset = tmp_path / "dataset.yaml"
    dataset.write_text(
        """\
sample: {cadence: 1h}
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
sample: {cadence: 1h}
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
        definition.dataset,
        definition.streams,
        definition.artifact_operations,
    )

    assert changed_artifact_hashes != definition.artifact_hashes


def test_metadata_format_version_invalidates_only_metadata_and_dependents(
    tmp_path: Path,
    monkeypatch,
) -> None:
    definition = load_project_definition(_write_project(tmp_path))
    tasks = (SeriesTask(), MetadataTask())
    current = calculate_artifact_hashes(
        definition.project,
        definition.dataset,
        definition.streams,
        tasks,
    )

    monkeypatch.setattr(
        fingerprints,
        "VECTOR_METADATA_VERSION",
        fingerprints.VECTOR_METADATA_VERSION + 1,
    )
    changed = calculate_artifact_hashes(
        definition.project,
        definition.dataset,
        definition.streams,
        tasks,
    )

    assert changed.for_artifact(SERIES) == current.for_artifact(SERIES)
    assert changed.for_artifact(VECTOR_METADATA) != current.for_artifact(
        VECTOR_METADATA
    )


def test_scaler_format_version_invalidates_only_scaler(
    tmp_path: Path,
    monkeypatch,
) -> None:
    definition = load_project_definition(_write_project(tmp_path))
    dataset = DatasetConfig(
        sample=SampleConfig(cadence="1h"),
        features=[
            SeriesConfig(
                id="price",
                stream="prices",
                field="close",
                scale=True,
            )
        ],
    )
    tasks = (ScalerTask(), SeriesTask())
    current = calculate_artifact_hashes(
        definition.project,
        dataset,
        _single_stream_catalog(),
        tasks,
    )

    monkeypatch.setattr(
        fingerprints,
        "SCALER_ARTIFACT_VERSION",
        fingerprints.SCALER_ARTIFACT_VERSION + 1,
    )
    changed = calculate_artifact_hashes(
        definition.project,
        dataset,
        _single_stream_catalog(),
        tasks,
    )

    assert changed.for_artifact(SCALER_STATISTICS) != current.for_artifact(
        SCALER_STATISTICS
    )
    assert changed.for_artifact(SERIES) == current.for_artifact(SERIES)
