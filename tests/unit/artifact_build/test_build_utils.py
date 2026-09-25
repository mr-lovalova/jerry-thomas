import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from jerrythomas.artifacts.registry import VECTOR_METADATA_SPEC
from jerrythomas.artifacts.series import SeriesRow
from jerrythomas.artifacts.specs import SERIES, VECTOR_METADATA
from jerrythomas.config.dataset.dataset import DatasetConfig, SampleConfig
from jerrythomas.config.dataset.series import SeriesConfig, TargetSeriesConfig
from jerrythomas.config.dataset.split import (
    DatasetFold,
    HashSplitConfig,
    TimeInterval,
    TimeSplitConfig,
)
from jerrythomas.config.tasks.metadata import MetadataTask
from jerrythomas.operations.artifacts import metadata as artifact_metadata
from jerrythomas.operations.artifacts.metadata import (
    _window_bounds_from_stats,
    build_metadata_artifact,
)
from jerrythomas.operations.artifacts.utils import (
    VectorMetadataStats,
    metadata_entries_from_stats,
)
from jerrythomas.runtime import DerivedRuntimeStream, Runtime
from jerrythomas.io.yaml import load_yaml


def test_metadata_task_rejects_dataset_window_mode() -> None:
    with pytest.raises(ValidationError, match="window_mode"):
        MetadataTask.model_validate({"window_mode": "union"})


def _hour(hour: int) -> datetime:
    return datetime(2024, 1, 1, hour=hour, tzinfo=timezone.utc)


def _runtime_with_config(tmp_path, dataset: DatasetConfig) -> Runtime:
    project_yaml = tmp_path / "project.yaml"
    project_yaml.write_text(
        "\n".join(
            [
                "schema_version: 7",
                "artifact_revision: 1",
                "paths:",
                "  streams: streams",
                "  sources: sources",
                "  datasets: datasets",
                "  artifacts: build",
                "  operations: operations",
                "",
            ]
        ),
        encoding="utf-8",
    )
    artifacts_root = tmp_path / "build"
    artifacts_root.mkdir()
    return Runtime(
        project_yaml=project_yaml,
        artifacts_root=artifacts_root,
        dataset=dataset,
    )


def _runtime_with_dataset(tmp_path, dataset_text: str) -> Runtime:
    dataset_path = tmp_path / "datasets/default.yaml"
    dataset_path.parent.mkdir(exist_ok=True)
    dataset_path.write_text(dataset_text, encoding="utf-8")
    return _runtime_with_config(
        tmp_path,
        DatasetConfig.model_validate(load_yaml(dataset_path)),
    )


def _runtime_with_stream_partitions(
    tmp_path,
    dataset: DatasetConfig,
    partitions: dict[str, tuple[str, ...]],
) -> Runtime:
    runtime = _runtime_with_config(tmp_path, dataset)
    runtime.streams.update(
        {
            stream_id: DerivedRuntimeStream(
                input_stream="unused",
                partition_by=partition_by,
                transforms=(),
            )
            for stream_id, partition_by in partitions.items()
        }
    )
    return runtime


def _holdout_split() -> TimeSplitConfig:
    return TimeSplitConfig(
        intervals=[
            TimeInterval(id="train", until="2024-01-01T02:00:00Z"),
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


def _mock_series_rows(monkeypatch, runtime: Runtime, rows) -> None:
    runtime.artifacts.register(SERIES, "series.json")
    manifest = SimpleNamespace(
        cadence=runtime.dataset.sample.cadence,
        rounding=runtime.dataset.sample.rounding,
        sample_keys=tuple(runtime.dataset.sample.keys),
    )
    monkeypatch.setattr(
        artifact_metadata,
        "load_series_manifest",
        lambda _path: manifest,
    )
    monkeypatch.setattr(
        artifact_metadata,
        "open_series",
        lambda *_args: iter(rows),
    )


def test_pipeline_context_rejects_invalid_registered_metadata(tmp_path) -> None:
    runtime = Runtime(
        project_yaml=tmp_path / "project.yaml",
        artifacts_root=tmp_path / "artifacts",
        dataset=DatasetConfig(sample=SampleConfig(rounding="ceil", cadence="1h")),
    )
    runtime.artifacts_root.mkdir()
    metadata_path = runtime.artifacts_root / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "catalog": {
                    "features": [
                        {
                            "id": "history",
                            "base_id": "history",
                            "kind": "list",
                            "present_count": 1,
                            "null_count": 0,
                            "length": 0,
                            "observed_elements": 0,
                        }
                    ],
                    "targets": [],
                    "counts": {"feature_vectors": 1, "target_vectors": 0},
                },
                "layout": {"kind": "unsplit"},
            }
        ),
        encoding="utf-8",
    )
    runtime.artifacts.register(VECTOR_METADATA, "metadata.json")

    with pytest.raises(ValueError, match="length"):
        runtime.artifacts.load(VECTOR_METADATA_SPEC)


def test_metadata_materialization_writes_keyed_sample_domain(
    monkeypatch,
    tmp_path,
) -> None:
    runtime = _runtime_with_dataset(
        tmp_path,
        "\n".join(
            [
                "sample:",
                "  rounding: ceil",
                "  cadence: 1h",
                "  keys: [security_id]",
                "features:",
                "  - id: price",
                "    stream: market.prices",
                "    field: close",
                "targets: []",
                "",
            ]
        ),
    )
    start = datetime(2024, 1, 1, 0, tzinfo=timezone.utc)
    end = datetime(2024, 1, 1, 1, tzinfo=timezone.utc)
    _mock_series_rows(
        monkeypatch,
        runtime,
        [
            SeriesRow(start, ("AAPL",), {"price": 1.0}, {}, frozenset()),
            SeriesRow(end, ("AAPL",), {"price": 2.0}, {}, frozenset()),
        ],
    )

    build_metadata_artifact(runtime, MetadataTask(path="metadata.json"))

    payload = json.loads(
        (runtime.artifacts_root / "metadata.json").read_text(encoding="utf-8")
    )
    assert payload["layout"] == {"kind": "unsplit"}
    assert payload["catalog"]["sample"] == {
        "cadence": "1h",
        "keys": ["security_id"],
        "domain": [
            {
                "key": ["AAPL"],
                "start": "2024-01-01T00:00:00Z",
                "end": "2024-01-01T01:00:00Z",
            }
        ],
    }


def test_unsplit_domain_starts_at_first_genuine_observation(
    monkeypatch,
    tmp_path,
) -> None:
    runtime = _runtime_with_dataset(
        tmp_path,
        "\n".join(
            [
                "sample:",
                "  rounding: ceil",
                "  cadence: 1h",
                "  keys: [security_id]",
                "features:",
                "  - id: price",
                "    stream: market.prices",
                "    field: close",
                "",
            ]
        ),
    )
    placeholder = frozenset({"price"})
    _mock_series_rows(
        monkeypatch,
        runtime,
        [
            SeriesRow(_hour(0), ("AAPL",), {"price": None}, {}, placeholder),
            SeriesRow(_hour(1), ("AAPL",), {"price": 1.0}, {}, frozenset()),
            SeriesRow(_hour(2), ("AAPL",), {"price": None}, {}, placeholder),
        ],
    )

    build_metadata_artifact(runtime, MetadataTask(path="metadata.json"))

    payload = json.loads(
        (runtime.artifacts_root / "metadata.json").read_text(encoding="utf-8")
    )
    assert payload["catalog"]["window"] == {
        "start": "2024-01-01T01:00:00Z",
        "end": "2024-01-01T02:00:00Z",
        "mode": "intersection",
        "size": 2,
    }
    assert payload["catalog"]["sample"]["domain"] == [
        {
            "key": ["AAPL"],
            "start": "2024-01-01T01:00:00Z",
            "end": "2024-01-01T02:00:00Z",
        }
    ]


def test_metadata_materialization_preserves_one_timestamp_window(
    monkeypatch,
    tmp_path,
) -> None:
    runtime = _runtime_with_dataset(
        tmp_path,
        "\n".join(
            [
                "sample:",
                "  rounding: ceil",
                "  cadence: 1h",
                "features:",
                "  - id: price",
                "    stream: market.prices",
                "    field: close",
                "targets: []",
                "",
            ]
        ),
    )
    observed_at = _hour(4)
    _mock_series_rows(
        monkeypatch,
        runtime,
        [SeriesRow(observed_at, (), {"price": 1.0}, {}, frozenset())],
    )

    build_metadata_artifact(runtime, MetadataTask(path="metadata.json"))

    path = runtime.artifacts_root / "metadata.json"
    first = path.read_bytes()
    payload = json.loads(first)
    assert payload["layout"] == {"kind": "unsplit"}
    assert payload["catalog"]["window"] == {
        "start": "2024-01-01T04:00:00Z",
        "end": "2024-01-01T04:00:00Z",
        "mode": "intersection",
        "size": 1,
    }
    assert "generated_at" not in payload

    build_metadata_artifact(runtime, MetadataTask(path="metadata.json"))

    assert path.read_bytes() == first


def test_metadata_materialization_scans_features_and_targets_once(
    monkeypatch,
    tmp_path,
) -> None:
    runtime = _runtime_with_dataset(
        tmp_path,
        "\n".join(
            [
                "sample:",
                "  rounding: ceil",
                "  cadence: 1h",
                "features:",
                "  - id: price",
                "    stream: market.prices",
                "    field: close",
                "targets:",
                "  - id: return",
                "    stream: market.prices",
                "    field: return",
                "    horizon: 0s",
                "",
            ]
        ),
    )
    rows = [
        SeriesRow(_hour(0), (), {"price": 1.0}, {"return": 0.1}, frozenset()),
        SeriesRow(_hour(1), (), {"price": 2.0}, {"return": 0.2}, frozenset()),
    ]
    _mock_series_rows(monkeypatch, runtime, rows)
    opens = 0
    trackers: list[tuple[str, str, float]] = []
    advances: list[int] = []

    def open_rows(*_args):
        nonlocal opens
        opens += 1
        return iter(rows)

    class Progress:
        def __init__(self, step: str, unit: str, interval_seconds: float) -> None:
            trackers.append((step, unit, interval_seconds))

        def advance(self, count: int = 1) -> None:
            advances.append(count)

    runtime.heartbeat_interval_seconds = 180
    monkeypatch.setattr(artifact_metadata, "open_series", open_rows)
    monkeypatch.setattr(artifact_metadata, "OperationProgressTracker", Progress)

    build_metadata_artifact(runtime, MetadataTask(path="metadata.json"))

    payload = json.loads(
        (runtime.artifacts_root / "metadata.json").read_text(encoding="utf-8")
    )
    assert payload["layout"] == {"kind": "unsplit"}
    catalog = payload["catalog"]
    assert catalog["counts"] == {
        "feature_vectors": 2,
        "target_vectors": 2,
    }
    assert [entry["id"] for entry in catalog["features"]] == ["price"]
    assert [entry["id"] for entry in catalog["targets"]] == ["return"]
    assert opens == 1
    assert trackers == [("scan_series", "samples", 180)]
    assert advances == [1, 1]


def test_metadata_rejects_wide_feature_missing_from_fold_training(
    monkeypatch,
    tmp_path,
) -> None:
    runtime = _runtime_with_stream_partitions(
        tmp_path,
        DatasetConfig(
            sample=SampleConfig(rounding="ceil", cadence="1h"),
            features=[
                SeriesConfig(
                    id="metric",
                    stream="market.metrics",
                    field="value",
                )
            ],
            split=_holdout_split(),
        ),
        {"market.metrics": ("name",)},
    )
    _mock_series_rows(
        monkeypatch,
        runtime,
        [
            SeriesRow(
                _hour(0),
                (),
                {"metric__@name:known": 1.0},
                {},
                frozenset(),
            ),
            SeriesRow(
                _hour(2),
                (),
                {
                    "metric__@name:known": 2.0,
                    "metric__@name:future": 3.0,
                },
                {},
                frozenset(),
            ),
        ],
    )

    with pytest.raises(
        RuntimeError,
        match=r"fold 'holdout'.*feature series IDs.*metric__@name:future",
    ):
        build_metadata_artifact(runtime, MetadataTask(path="metadata.json"))


def test_metadata_validates_wide_schema_for_each_fold(
    monkeypatch,
    tmp_path,
) -> None:
    runtime = _runtime_with_stream_partitions(
        tmp_path,
        DatasetConfig(
            sample=SampleConfig(rounding="ceil", cadence="1h"),
            features=[
                SeriesConfig(
                    id="metric",
                    stream="market.metrics",
                    field="value",
                ),
                SeriesConfig(
                    id="baseline",
                    stream="market.baseline",
                    field="value",
                ),
            ],
            split=TimeSplitConfig(
                intervals=[
                    TimeInterval(id="train_0", until="2024-01-01T02:00:00Z"),
                    TimeInterval(id="validation_0", until="2024-01-01T04:00:00Z"),
                    TimeInterval(id="train_1", until="2024-01-01T06:00:00Z"),
                    TimeInterval(id="validation_1"),
                ],
                folds=[
                    DatasetFold(
                        id="fold_0",
                        train=["train_0"],
                        validation=["validation_0"],
                    ),
                    DatasetFold(
                        id="fold_1",
                        train=["train_1"],
                        validation=["validation_1"],
                    ),
                ],
            ),
        ),
        {
            "market.metrics": ("bucket",),
            "market.baseline": (),
        },
    )
    _mock_series_rows(
        monkeypatch,
        runtime,
        [
            SeriesRow(
                _hour(0),
                (),
                {
                    "metric__@bucket:legacy": 1.0,
                    "baseline": 10.0,
                },
                {},
                frozenset(),
            ),
            SeriesRow(
                _hour(5),
                (),
                {"baseline": 20.0},
                {},
                frozenset(),
            ),
        ],
    )

    with pytest.raises(
        RuntimeError,
        match=r"fold 'fold_1'.*configured feature series.*metric",
    ):
        build_metadata_artifact(runtime, MetadataTask(path="metadata.json"))


def test_metadata_accepts_wide_feature_present_in_fold_training(
    monkeypatch,
    tmp_path,
) -> None:
    runtime = _runtime_with_stream_partitions(
        tmp_path,
        DatasetConfig(
            sample=SampleConfig(rounding="ceil", cadence="1h"),
            features=[
                SeriesConfig(
                    id="metric",
                    stream="market.metrics",
                    field="value",
                )
            ],
            split=_holdout_split(),
        ),
        {"market.metrics": ("name",)},
    )
    _mock_series_rows(
        monkeypatch,
        runtime,
        [
            SeriesRow(
                _hour(0),
                (),
                {"metric__@name:known": [1.0, 2.0]},
                {},
                frozenset(),
            ),
            SeriesRow(
                _hour(2),
                (),
                {"metric__@name:known": [None, 3.0]},
                {},
                frozenset(),
            ),
        ],
    )

    build_metadata_artifact(runtime, MetadataTask(path="metadata.json"))

    payload = json.loads(
        (runtime.artifacts_root / "metadata.json").read_text(encoding="utf-8")
    )
    assert payload["layout"]["kind"] == "folded"
    training = payload["layout"]["folds"][0]["training_schema"]
    assert [entry["id"] for entry in training["features"]] == ["metric__@name:known"]


def test_metadata_rejects_wide_shape_not_established_by_fold_training(
    monkeypatch,
    tmp_path,
) -> None:
    runtime = _runtime_with_stream_partitions(
        tmp_path,
        DatasetConfig(
            sample=SampleConfig(rounding="ceil", cadence="1h"),
            features=[
                SeriesConfig(
                    id="metric",
                    stream="market.metrics",
                    field="value",
                )
            ],
            split=_holdout_split(),
        ),
        {"market.metrics": ("name",)},
    )
    _mock_series_rows(
        monkeypatch,
        runtime,
        [
            SeriesRow(
                _hour(0),
                (),
                {"metric__@name:known": None},
                {},
                frozenset(),
            ),
            SeriesRow(
                _hour(2),
                (),
                {"metric__@name:known": [1.0, 2.0]},
                {},
                frozenset(),
            ),
        ],
    )

    with pytest.raises(
        RuntimeError,
        match=r"fold 'holdout'.*feature series IDs.*metric__@name:known",
    ):
        build_metadata_artifact(runtime, MetadataTask(path="metadata.json"))


def test_metadata_rejects_static_shape_not_established_by_fold_training(
    monkeypatch,
    tmp_path,
) -> None:
    runtime = _runtime_with_config(
        tmp_path,
        DatasetConfig(
            sample=SampleConfig(rounding="ceil", cadence="1h"),
            features=[
                SeriesConfig(
                    id="metric",
                    stream="market.metrics",
                    field="value",
                )
            ],
            split=_holdout_split(),
        ),
    )
    _mock_series_rows(
        monkeypatch,
        runtime,
        [
            SeriesRow(_hour(0), (), {"metric": None}, {}, frozenset()),
            SeriesRow(
                _hour(2),
                (),
                {"metric": [1.0, 2.0]},
                {},
                frozenset(),
            ),
        ],
    )

    with pytest.raises(
        RuntimeError,
        match=r"fold 'holdout'.*feature series IDs.*metric",
    ):
        build_metadata_artifact(runtime, MetadataTask(path="metadata.json"))


def test_hash_fold_domain_does_not_synthesize_holdout_entity_into_training(
    monkeypatch,
    tmp_path,
) -> None:
    runtime = _runtime_with_config(
        tmp_path,
        DatasetConfig(
            sample=SampleConfig(rounding="ceil", cadence="1h", keys=["entity"]),
            features=[
                SeriesConfig(
                    id="price",
                    stream="market.prices",
                    field="value",
                )
            ],
            split=HashSplitConfig(
                ratios={"train": 0.5, "validation": 0.5},
                folds=[
                    DatasetFold(
                        id="fold",
                        train=["train"],
                        validation=["validation"],
                    )
                ],
                seed=1,
            ),
        ),
    )
    _mock_series_rows(
        monkeypatch,
        runtime,
        [
            SeriesRow(
                _hour(14),
                ("HOLDOUT",),
                {"price": 14.0},
                {},
                frozenset(),
            ),
            SeriesRow(
                _hour(15),
                ("BASE",),
                {"price": 15.0},
                {},
                frozenset(),
            ),
            SeriesRow(
                _hour(16),
                ("BASE",),
                {"price": 16.0},
                {},
                frozenset(),
            ),
            SeriesRow(
                _hour(16),
                ("HOLDOUT",),
                {"price": 16.0},
                {},
                frozenset(),
            ),
        ],
    )

    build_metadata_artifact(runtime, MetadataTask(path="metadata.json"))

    payload = json.loads(
        (runtime.artifacts_root / "metadata.json").read_text(encoding="utf-8")
    )
    fold = payload["layout"]["folds"][0]
    training = next(output for output in fold["outputs"] if output["role"] == "train")
    assert training["sample"]["domain"] == [
        {
            "key": ["BASE"],
            "start": "2024-01-01T15:00:00Z",
            "end": "2024-01-01T16:00:00Z",
        }
    ]


def test_schedule_placeholder_does_not_establish_training_membership(
    monkeypatch,
    tmp_path,
) -> None:
    runtime = _runtime_with_config(
        tmp_path,
        DatasetConfig(
            sample=SampleConfig(rounding="ceil", cadence="1h", keys=["entity"]),
            features=[
                SeriesConfig(
                    id="price",
                    stream="market.prices",
                    field="value",
                )
            ],
            split=TimeSplitConfig(
                intervals=[
                    TimeInterval(id="train", until="2024-01-01T02:00:00Z"),
                    TimeInterval(id="validation"),
                ],
                folds=[
                    DatasetFold(
                        id="fold",
                        train=["train"],
                        validation=["validation"],
                    )
                ],
            ),
        ),
    )
    placeholder = frozenset({"price"})
    _mock_series_rows(
        monkeypatch,
        runtime,
        [
            SeriesRow(
                _hour(0),
                ("BASE",),
                {"price": 1.0},
                {},
                frozenset(),
            ),
            SeriesRow(
                _hour(0),
                ("HOLDOUT",),
                {"price": None},
                {},
                placeholder,
            ),
            SeriesRow(
                _hour(1),
                ("BASE",),
                {"price": None},
                {},
                placeholder,
            ),
            SeriesRow(
                _hour(1),
                ("HOLDOUT",),
                {"price": None},
                {},
                placeholder,
            ),
            SeriesRow(
                _hour(2),
                ("BASE",),
                {"price": None},
                {},
                placeholder,
            ),
            SeriesRow(
                _hour(2),
                ("HOLDOUT",),
                {"price": 2.0},
                {},
                frozenset(),
            ),
        ],
    )

    build_metadata_artifact(runtime, MetadataTask(path="metadata.json"))

    payload = json.loads(
        (runtime.artifacts_root / "metadata.json").read_text(encoding="utf-8")
    )
    fold = payload["layout"]["folds"][0]
    training = next(output for output in fold["outputs"] if output["role"] == "train")
    assert training["sample"]["domain"] == [
        {
            "key": ["BASE"],
            "start": "2024-01-01T00:00:00Z",
            "end": "2024-01-01T01:00:00Z",
        }
    ]


def test_fold_domain_starts_at_first_genuine_observation(
    monkeypatch,
    tmp_path,
) -> None:
    runtime = _runtime_with_config(
        tmp_path,
        DatasetConfig(
            sample=SampleConfig(rounding="ceil", cadence="1h", keys=["entity"]),
            features=[
                SeriesConfig(
                    id="price",
                    stream="market.prices",
                    field="value",
                )
            ],
            split=TimeSplitConfig(
                intervals=[
                    TimeInterval(id="train", until="2024-01-01T03:00:00Z"),
                    TimeInterval(id="validation"),
                ],
                folds=[
                    DatasetFold(
                        id="fold",
                        train=["train"],
                        validation=["validation"],
                    )
                ],
            ),
        ),
    )
    _mock_series_rows(
        monkeypatch,
        runtime,
        [
            SeriesRow(
                _hour(0),
                ("AAPL",),
                {"price": None},
                {},
                frozenset({"price"}),
            ),
            SeriesRow(
                _hour(1),
                ("AAPL",),
                {"price": 1.0},
                {},
                frozenset(),
            ),
            SeriesRow(
                _hour(2),
                ("AAPL",),
                {"price": None},
                {},
                frozenset({"price"}),
            ),
        ],
    )

    build_metadata_artifact(runtime, MetadataTask(path="metadata.json"))

    payload = json.loads(
        (runtime.artifacts_root / "metadata.json").read_text(encoding="utf-8")
    )
    fold = payload["layout"]["folds"][0]
    training = next(output for output in fold["outputs"] if output["role"] == "train")
    assert training["sample"]["domain"] == [
        {
            "key": ["AAPL"],
            "start": "2024-01-01T01:00:00Z",
            "end": "2024-01-01T02:00:00Z",
        }
    ]


def test_fold_domain_continues_after_training_observation(
    monkeypatch,
    tmp_path,
) -> None:
    runtime = _runtime_with_config(
        tmp_path,
        DatasetConfig(
            sample=SampleConfig(rounding="ceil", cadence="1h", keys=["entity"]),
            features=[
                SeriesConfig(
                    id="price",
                    stream="market.prices",
                    field="value",
                )
            ],
            split=TimeSplitConfig(
                intervals=[
                    TimeInterval(id="train", until="2024-01-01T02:00:00Z"),
                    TimeInterval(id="validation"),
                ],
                folds=[
                    DatasetFold(
                        id="fold",
                        train=["train"],
                        validation=["validation"],
                    )
                ],
            ),
        ),
    )
    _mock_series_rows(
        monkeypatch,
        runtime,
        [
            SeriesRow(
                _hour(0),
                ("AAPL",),
                {"price": 1.0},
                {},
                frozenset(),
            ),
            SeriesRow(
                _hour(2),
                ("AAPL",),
                {"price": None},
                {},
                frozenset({"price"}),
            ),
            SeriesRow(
                _hour(3),
                ("AAPL",),
                {"price": None},
                {},
                frozenset({"price"}),
            ),
        ],
    )

    build_metadata_artifact(runtime, MetadataTask(path="metadata.json"))

    payload = json.loads(
        (runtime.artifacts_root / "metadata.json").read_text(encoding="utf-8")
    )
    fold = payload["layout"]["folds"][0]
    validation = next(
        output for output in fold["outputs"] if output["role"] == "validation"
    )
    assert validation["sample"]["domain"] == [
        {
            "key": ["AAPL"],
            "start": "2024-01-01T02:00:00Z",
            "end": "2024-01-01T03:00:00Z",
        }
    ]


@pytest.mark.parametrize(
    ("window_mode", "expected_start"),
    [
        ("union", "2024-01-01T00:00:00Z"),
        ("intersection", "2024-01-01T01:00:00Z"),
    ],
)
def test_fold_window_uses_feature_base_ranges(
    monkeypatch,
    tmp_path,
    window_mode,
    expected_start,
) -> None:
    runtime = _runtime_with_config(
        tmp_path,
        DatasetConfig(
            sample=SampleConfig(rounding="ceil", cadence="1h", window_mode=window_mode),
            features=[
                SeriesConfig(id="price", stream="market", field="price"),
                SeriesConfig(id="volume", stream="market", field="volume"),
            ],
            split=_holdout_split(),
        ),
    )
    _mock_series_rows(
        monkeypatch,
        runtime,
        [
            SeriesRow(
                _hour(0),
                (),
                {"price": 1.0, "volume": None},
                {},
                frozenset({"volume"}),
            ),
            SeriesRow(
                _hour(1),
                (),
                {"price": 2.0, "volume": 10.0},
                {},
                frozenset(),
            ),
        ],
    )

    build_metadata_artifact(runtime, MetadataTask(path="metadata.json"))

    payload = json.loads(
        (runtime.artifacts_root / "metadata.json").read_text(encoding="utf-8")
    )
    training = next(
        output
        for output in payload["layout"]["folds"][0]["outputs"]
        if output["role"] == "train"
    )
    assert training["window"]["start"] == expected_start
    assert training["window"]["end"] == "2024-01-01T01:00:00Z"


def test_fold_strict_window_uses_each_wide_series_id(
    monkeypatch,
    tmp_path,
) -> None:
    runtime = _runtime_with_config(
        tmp_path,
        DatasetConfig(
            sample=SampleConfig(rounding="ceil", cadence="1h", window_mode="strict"),
            features=[
                SeriesConfig(id="metric", stream="market.metrics", field="value")
            ],
            split=_holdout_split(),
        ),
    )
    _mock_series_rows(
        monkeypatch,
        runtime,
        [
            SeriesRow(
                _hour(0),
                (),
                {
                    "metric__@name:a": 1.0,
                    "metric__@name:b": None,
                },
                {},
                frozenset({"metric__@name:b"}),
            ),
            SeriesRow(
                _hour(1),
                (),
                {
                    "metric__@name:a": 2.0,
                    "metric__@name:b": 3.0,
                },
                {},
                frozenset(),
            ),
        ],
    )

    build_metadata_artifact(runtime, MetadataTask(path="metadata.json"))

    payload = json.loads(
        (runtime.artifacts_root / "metadata.json").read_text(encoding="utf-8")
    )
    training = next(
        output
        for output in payload["layout"]["folds"][0]["outputs"]
        if output["role"] == "train"
    )
    assert training["window"] == {
        "start": "2024-01-01T01:00:00Z",
        "end": "2024-01-01T01:00:00Z",
        "mode": "strict",
        "size": 1,
    }


def test_hash_fold_validation_cannot_establish_training_domain(
    monkeypatch,
    tmp_path,
) -> None:
    runtime = _runtime_with_config(
        tmp_path,
        DatasetConfig(
            sample=SampleConfig(rounding="ceil", cadence="1h", keys=["entity"]),
            features=[
                SeriesConfig(
                    id="price",
                    stream="market.prices",
                    field="value",
                )
            ],
            split=HashSplitConfig(
                ratios={"train": 0.5, "validation": 0.5},
                folds=[
                    DatasetFold(
                        id="fold",
                        train=["train"],
                        validation=["validation"],
                    )
                ],
                seed=42,
            ),
        ),
    )
    _mock_series_rows(
        monkeypatch,
        runtime,
        [
            SeriesRow(_hour(0), ("A",), {"price": 10.0}, {}, frozenset()),
            SeriesRow(_hour(0), ("C",), {"price": 20.0}, {}, frozenset()),
            SeriesRow(
                _hour(5),
                ("A",),
                {"price": 10.0},
                {},
                frozenset({"price"}),
            ),
        ],
    )

    build_metadata_artifact(runtime, MetadataTask(path="metadata.json"))

    payload = json.loads(
        (runtime.artifacts_root / "metadata.json").read_text(encoding="utf-8")
    )
    training = next(
        output
        for output in payload["layout"]["folds"][0]["outputs"]
        if output["role"] == "train"
    )
    assert training["sample"]["domain"] == [
        {
            "key": ["C"],
            "start": "2024-01-01T00:00:00Z",
            "end": "2024-01-01T00:00:00Z",
        }
    ]


def test_fold_metadata_normalizes_interval_offsets_to_utc(
    monkeypatch,
    tmp_path,
) -> None:
    runtime = _runtime_with_config(
        tmp_path,
        DatasetConfig(
            sample=SampleConfig(rounding="ceil", cadence="1h"),
            features=[
                SeriesConfig(
                    id="price",
                    stream="market.prices",
                    field="value",
                )
            ],
            split=TimeSplitConfig(
                intervals=[
                    TimeInterval(
                        id="train",
                        until="2024-01-01T02:00:00+01:00",
                    ),
                    TimeInterval(id="validation"),
                ],
                folds=[
                    DatasetFold(
                        id="fold",
                        train=["train"],
                        validation=["validation"],
                    )
                ],
            ),
        ),
    )
    _mock_series_rows(
        monkeypatch,
        runtime,
        [
            SeriesRow(
                _hour(0),
                (),
                {"price": 1.0},
                {},
                frozenset(),
            ),
            SeriesRow(
                _hour(1),
                (),
                {"price": 2.0},
                {},
                frozenset(),
            ),
        ],
    )

    build_metadata_artifact(runtime, MetadataTask(path="metadata.json"))

    payload = json.loads(
        (runtime.artifacts_root / "metadata.json").read_text(encoding="utf-8")
    )
    outputs = payload["layout"]["folds"][0]["outputs"]
    training = next(output for output in outputs if output["role"] == "train")
    validation = next(output for output in outputs if output["role"] == "validation")
    assert training["window"] == {
        "end": "2024-01-01T00:00:00Z",
        "mode": "intersection",
        "size": 1,
        "start": "2024-01-01T00:00:00Z",
    }
    assert validation["window"] == {
        "end": "2024-01-01T01:00:00Z",
        "mode": "intersection",
        "size": 1,
        "start": "2024-01-01T01:00:00Z",
    }


def test_metadata_excludes_target_horizon_boundary_from_wide_training_schema(
    monkeypatch,
    tmp_path,
) -> None:
    runtime = _runtime_with_stream_partitions(
        tmp_path,
        DatasetConfig(
            sample=SampleConfig(rounding="ceil", cadence="1h"),
            features=[
                SeriesConfig(
                    id="metric",
                    stream="market.metrics",
                    field="value",
                )
            ],
            targets=[
                TargetSeriesConfig(
                    id="return",
                    stream="market.returns",
                    field="value",
                    horizon="1h",
                )
            ],
            split=_holdout_split(),
        ),
        {
            "market.metrics": ("name",),
            "market.returns": (),
        },
    )
    _mock_series_rows(
        monkeypatch,
        runtime,
        [
            SeriesRow(
                _hour(0),
                (),
                {"metric__@name:known": 1.0},
                {"return": 0.1},
                frozenset(),
            ),
            SeriesRow(
                _hour(1),
                (),
                {"metric__@name:future": 2.0},
                {"return": 0.2},
                frozenset(),
            ),
            SeriesRow(
                _hour(2),
                (),
                {"metric__@name:future": 3.0},
                {"return": 0.3},
                frozenset(),
            ),
        ],
    )

    with pytest.raises(
        RuntimeError,
        match=r"fold 'holdout'.*feature series IDs.*metric__@name:future",
    ):
        build_metadata_artifact(runtime, MetadataTask(path="metadata.json"))


def test_metadata_rejects_wide_target_missing_from_fold_training(
    monkeypatch,
    tmp_path,
) -> None:
    runtime = _runtime_with_stream_partitions(
        tmp_path,
        DatasetConfig(
            sample=SampleConfig(rounding="ceil", cadence="1h"),
            features=[
                SeriesConfig(
                    id="price",
                    stream="market.prices",
                    field="value",
                )
            ],
            targets=[
                TargetSeriesConfig(
                    id="return",
                    stream="market.returns",
                    field="value",
                    horizon="0s",
                )
            ],
            split=_holdout_split(),
        ),
        {
            "market.prices": (),
            "market.returns": ("name",),
        },
    )
    _mock_series_rows(
        monkeypatch,
        runtime,
        [
            SeriesRow(
                _hour(0),
                (),
                {"price": 1.0},
                {"return__@name:known": 0.1},
                frozenset(),
            ),
            SeriesRow(
                _hour(2),
                (),
                {"price": 2.0},
                {
                    "return__@name:known": 0.2,
                    "return__@name:future": 0.3,
                },
                frozenset(),
            ),
        ],
    )

    with pytest.raises(
        RuntimeError,
        match=r"fold 'holdout'.*target series IDs.*return__@name:future",
    ):
        build_metadata_artifact(runtime, MetadataTask(path="metadata.json"))


def test_metadata_materialization_closes_rows_after_collection_error(
    monkeypatch,
    tmp_path,
) -> None:
    runtime = _runtime_with_dataset(
        tmp_path,
        "\n".join(
            [
                "sample:",
                "  rounding: ceil",
                "  cadence: 1h",
                "features:",
                "  - id: history",
                "    stream: market.prices",
                "    field: close",
                "",
            ]
        ),
    )
    closed = False

    def failing_rows():
        nonlocal closed
        try:
            yield SeriesRow(_hour(0), (), {"history": [1.0]}, {}, frozenset())
            yield SeriesRow(
                _hour(1),
                (),
                {"history": [2.0, 3.0]},
                {},
                frozenset(),
            )
        finally:
            closed = True

    _mock_series_rows(monkeypatch, runtime, failing_rows())

    with pytest.raises(ValueError, match="different lengths"):
        build_metadata_artifact(runtime, MetadataTask(path="metadata.json"))

    assert closed


def test_metadata_entries_include_observation_bounds():
    stats = [
        VectorMetadataStats(
            id="temp",
            base_id="temp",
            kind="scalar",
            present_count=4,
            first_observed=datetime(2024, 1, 1, tzinfo=timezone.utc),
            last_observed=datetime(2024, 1, 2, tzinfo=timezone.utc),
        )
    ]

    entries = metadata_entries_from_stats(stats)

    assert entries[0].first_observed == datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert entries[0].last_observed == datetime(2024, 1, 2, tzinfo=timezone.utc)


def test_metadata_entries_preserve_scalar_and_list_statistics() -> None:
    scalar = VectorMetadataStats(id="price", base_id="price")
    scalar.observe(None, _hour(0))
    scalar.observe(3, _hour(2))
    sequence = VectorMetadataStats(id="history", base_id="history")
    sequence.observe([1.0, None, "stale"], _hour(1))
    sequence.observe([None, 2.0, 3.0], _hour(3))
    null_only = VectorMetadataStats(id="missing", base_id="missing")
    null_only.observe(None, _hour(0))
    null_only.observe(float("nan"), _hour(1))

    entries = metadata_entries_from_stats([scalar, sequence, null_only])

    assert entries[0].model_dump() == {
        "id": "price",
        "base_id": "price",
        "kind": "scalar",
        "present_count": 2,
        "null_count": 1,
        "first_observed": _hour(0),
        "last_observed": _hour(2),
        "value_types": ("int",),
    }
    assert entries[1].model_dump() == {
        "id": "history",
        "base_id": "history",
        "kind": "list",
        "present_count": 2,
        "null_count": 0,
        "first_observed": _hour(1),
        "last_observed": _hour(3),
        "element_types": ("float", "null", "str"),
        "length": 3,
        "observed_elements": 4,
    }
    assert entries[2].model_dump() == {
        "id": "missing",
        "base_id": "missing",
        "kind": "scalar",
        "present_count": 2,
        "null_count": 2,
        "first_observed": _hour(0),
        "last_observed": _hour(1),
        "value_types": (),
    }


def test_window_bounds_modes():
    feature_stats = [
        VectorMetadataStats(
            id="wind__@A",
            base_id="wind",
            first_observed=_hour(0),
            last_observed=_hour(6),
        ),
        VectorMetadataStats(
            id="wind__@B",
            base_id="wind",
            first_observed=_hour(2),
            last_observed=_hour(5),
        ),
        VectorMetadataStats(
            id="temp",
            base_id="temp",
            first_observed=_hour(1),
            last_observed=_hour(7),
        ),
    ]
    target_stats: list[VectorMetadataStats] = []

    start, end = _window_bounds_from_stats(feature_stats, target_stats, mode="union")
    assert start == _hour(0)
    assert end == _hour(7)

    start, end = _window_bounds_from_stats(
        feature_stats, target_stats, mode="intersection"
    )
    assert start == _hour(1)
    assert end == _hour(6)

    start, end = _window_bounds_from_stats(feature_stats, target_stats, mode="strict")
    assert start == _hour(2)
    assert end == _hour(5)


def test_window_bounds_preserve_single_timestamp() -> None:
    stats = [
        VectorMetadataStats(
            id="price",
            base_id="price",
            first_observed=_hour(4),
            last_observed=_hour(4),
        )
    ]

    assert _window_bounds_from_stats(stats, [], mode="union") == (
        _hour(4),
        _hour(4),
    )
    assert _window_bounds_from_stats(stats, [], mode="intersection") == (
        _hour(4),
        _hour(4),
    )
