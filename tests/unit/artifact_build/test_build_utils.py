import json
import math
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from datapipeline.artifacts.registry import VECTOR_METADATA_SPEC
from datapipeline.artifacts.series import SeriesRow
from datapipeline.artifacts.specs import SERIES, VECTOR_METADATA
from datapipeline.config.dataset.dataset import DatasetConfig, SampleConfig
from datapipeline.config.dataset.series import SeriesConfig
from datapipeline.config.tasks import MetadataTask
from datapipeline.execution.context import PipelineContext
from datapipeline.operations.artifacts import metadata as artifact_metadata
from datapipeline.operations.artifacts.metadata import (
    _window_bounds_from_stats,
    materialize_metadata,
)
from datapipeline.operations.artifacts.utils import (
    VectorMetadataCollector,
    VectorMetadataStats,
    metadata_entries_from_stats,
)
from datapipeline.runtime import Runtime
from datapipeline.utils.load import load_yaml


def _hour(hour: int) -> datetime:
    return datetime(2024, 1, 1, hour=hour, tzinfo=timezone.utc)


def test_metadata_task_rejects_removed_relaxed_window_mode() -> None:
    with pytest.raises(ValidationError, match="window_mode"):
        MetadataTask.model_validate({"window_mode": "relaxed"})


def _runtime_with_dataset(tmp_path, dataset_text: str) -> Runtime:
    project_yaml = tmp_path / "project.yaml"
    project_yaml.write_text(
        "\n".join(
            [
                "schema_version: 3",
                "artifact_revision: 1",
                "paths:",
                "  streams: streams",
                "  sources: sources",
                "  dataset: dataset.yaml",
                "  artifacts: build",
                "  operations: operations",
                "",
            ]
        ),
        encoding="utf-8",
    )
    dataset_path = tmp_path / "dataset.yaml"
    dataset_path.write_text(dataset_text, encoding="utf-8")
    artifacts_root = tmp_path / "build"
    artifacts_root.mkdir()
    runtime = Runtime(
        project_yaml=project_yaml,
        artifacts_root=artifacts_root,
        dataset=DatasetConfig.model_validate(load_yaml(dataset_path)),
    )
    return runtime


def _collect_metadata(
    configs: list[SeriesConfig],
    observations,
    sample_keys=(),
):
    collector = VectorMetadataCollector(configs, sample_keys)
    for key, values in observations:
        collector.observe(key, values)
    return collector.result()


def _mock_series_rows(monkeypatch, runtime: Runtime, rows) -> None:
    runtime.artifacts.register(SERIES, "series.json")
    manifest = SimpleNamespace(
        cadence=runtime.dataset.sample.cadence,
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


def test_vector_metadata_collector_counts_nan() -> None:
    config = SeriesConfig(id="wind_speed", stream="met.obs", field="value")
    stats, vector_count, domain = _collect_metadata(
        [config],
        [((_hour(0),), {"wind_speed": math.nan})],
    )

    assert vector_count == 1
    assert domain == {}
    assert stats[0].present_count == 1
    assert stats[0].null_count == 1


def test_vector_metadata_collector_uses_config_order() -> None:
    configs = [
        SeriesConfig(id="history", stream="market.prices", field="value"),
        SeriesConfig(id="price", stream="market.prices", field="value"),
        SeriesConfig(
            id="fundamental",
            stream="market.prices",
            field="value",
        ),
    ]
    stats, _, _ = _collect_metadata(
        configs,
        [
            (
                (_hour(0),),
                {
                    "price": 1.0,
                    "fundamental__@metric:revenue": 10.0,
                    "fundamental__@metric:debt": 5.0,
                },
            ),
            ((_hour(1),), {"history": [1.0, 2.0]}),
        ],
    )

    assert [entry.id for entry in stats] == [
        "history",
        "price",
        "fundamental__@metric:debt",
        "fundamental__@metric:revenue",
    ]


def test_vector_metadata_collector_rejects_missing_configured_ids() -> None:
    configs = [
        SeriesConfig(id="missing", stream="market.prices", field="value"),
        SeriesConfig(id="price", stream="market.prices", field="value"),
    ]

    with pytest.raises(RuntimeError, match="missing"):
        _collect_metadata(
            configs,
            [((_hour(0),), {"price": 1.0})],
        )


def test_vector_metadata_collector_rejects_when_all_ids_are_empty() -> None:
    config = SeriesConfig(id="price", stream="market.prices", field="value")

    with pytest.raises(RuntimeError, match="price"):
        _collect_metadata([config], [])


@pytest.mark.parametrize(
    ("sample_keys", "key"),
    [
        ((), 1),
        ((), ()),
        ((), (0,)),
        ((), (_hour(0), "unexpected")),
        (("ticker",), (_hour(0),)),
    ],
)
def test_vector_metadata_collector_rejects_malformed_sample_keys(
    sample_keys,
    key,
) -> None:
    config = SeriesConfig(id="price", stream="market.prices", field="value")

    with pytest.raises(RuntimeError, match="sample key"):
        _collect_metadata(
            [config],
            [(key, {"price": 1.0})],
            sample_keys=sample_keys,
        )


def test_vector_metadata_collector_rejects_unknown_ids() -> None:
    config = SeriesConfig(id="price", stream="market.prices", field="value")

    with pytest.raises(RuntimeError, match="rogue"):
        _collect_metadata(
            [config],
            [((_hour(0),), {"price": 1.0, "rogue": 2.0})],
        )


def test_vector_metadata_collector_omits_unkeyed_domain() -> None:
    config = SeriesConfig(id="price", stream="market.prices", field="value")

    _, _, domain = _collect_metadata(
        [config],
        [
            ((_hour(0),), {"price": 1.0}),
            ((_hour(1),), {"price": 2.0}),
        ],
    )

    assert domain == {}


def test_vector_metadata_collector_tracks_each_keyed_domain() -> None:
    config = SeriesConfig(id="price", stream="market.prices", field="value")

    _, _, domain = _collect_metadata(
        [config],
        [
            ((_hour(2), "AAPL"), {"price": 2.0}),
            ((_hour(0), "AAPL"), {"price": 1.0}),
            ((_hour(1), "MSFT"), {"price": 3.0}),
        ],
        sample_keys=("ticker",),
    )

    assert domain == {
        ("AAPL",): (_hour(0), _hour(2)),
        ("MSFT",): (_hour(1), _hour(1)),
    }


@pytest.mark.parametrize(
    "values",
    [
        (1.0, [2.0]),
        ([1.0], 2.0),
    ],
)
def test_vector_metadata_collector_rejects_scalar_list_mixtures(values) -> None:
    config = SeriesConfig(id="history", stream="market.prices", field="close")

    with pytest.raises(ValueError, match="both (scalar and list|list and scalar)"):
        _collect_metadata(
            [config],
            [
                ((_hour(0),), {"history": values[0]}),
                ((_hour(1),), {"history": values[1]}),
            ],
        )


def test_vector_metadata_collector_rejects_variable_list_lengths() -> None:
    config = SeriesConfig(id="history", stream="market.prices", field="close")

    with pytest.raises(ValueError, match="different lengths: 1 and 2"):
        _collect_metadata(
            [config],
            [
                ((_hour(0),), {"history": [1.0]}),
                ((_hour(1),), {"history": [2.0, 3.0]}),
            ],
        )


def test_pipeline_context_rejects_invalid_registered_metadata(tmp_path) -> None:
    runtime = Runtime(
        project_yaml=tmp_path / "project.yaml",
        artifacts_root=tmp_path / "artifacts",
        dataset=DatasetConfig(sample=SampleConfig(cadence="1h")),
    )
    runtime.artifacts_root.mkdir()
    metadata_path = runtime.artifacts_root / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": 3,
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
            }
        ),
        encoding="utf-8",
    )
    runtime.artifacts.register(VECTOR_METADATA, "metadata.json")

    with pytest.raises(ValueError, match="length"):
        PipelineContext(runtime).require_artifact(VECTOR_METADATA_SPEC)


def test_metadata_materialization_writes_keyed_sample_domain(
    monkeypatch,
    tmp_path,
) -> None:
    runtime = _runtime_with_dataset(
        tmp_path,
        "\n".join(
            [
                "sample:",
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
            SeriesRow(start, ("AAPL",), {"price": 1.0}, {}),
            SeriesRow(end, ("AAPL",), {"price": 2.0}, {}),
        ],
    )

    materialize_metadata(runtime, MetadataTask(output="metadata.json"))

    payload = json.loads(
        (runtime.artifacts_root / "metadata.json").read_text(encoding="utf-8")
    )
    assert payload["sample"] == {
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


def test_metadata_materialization_preserves_one_timestamp_window(
    monkeypatch,
    tmp_path,
) -> None:
    runtime = _runtime_with_dataset(
        tmp_path,
        "\n".join(
            [
                "sample:",
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
        [SeriesRow(observed_at, (), {"price": 1.0}, {})],
    )

    materialize_metadata(runtime, MetadataTask(output="metadata.json"))

    path = runtime.artifacts_root / "metadata.json"
    first = path.read_bytes()
    payload = json.loads(first)
    assert payload["window"] == {
        "start": "2024-01-01T04:00:00Z",
        "end": "2024-01-01T04:00:00Z",
        "mode": "intersection",
        "size": 1,
    }
    assert "generated_at" not in payload

    materialize_metadata(runtime, MetadataTask(output="metadata.json"))

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
        SeriesRow(_hour(0), (), {"price": 1.0}, {"return": 0.1}),
        SeriesRow(_hour(1), (), {"price": 2.0}, {"return": 0.2}),
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

    materialize_metadata(runtime, MetadataTask(output="metadata.json"))

    payload = json.loads(
        (runtime.artifacts_root / "metadata.json").read_text(encoding="utf-8")
    )
    assert payload["counts"] == {
        "feature_vectors": 2,
        "target_vectors": 2,
    }
    assert [entry["id"] for entry in payload["features"]] == ["price"]
    assert [entry["id"] for entry in payload["targets"]] == ["return"]
    assert opens == 1
    assert trackers == [("scan_series", "samples", 180)]
    assert advances == [1, 1]


def test_metadata_materialization_closes_rows_after_collection_error(
    monkeypatch,
    tmp_path,
) -> None:
    runtime = _runtime_with_dataset(
        tmp_path,
        "\n".join(
            [
                "sample:",
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
            yield SeriesRow(_hour(0), (), {"history": [1.0]}, {})
            yield SeriesRow(_hour(1), (), {"history": [2.0, 3.0]}, {})
        finally:
            closed = True

    _mock_series_rows(monkeypatch, runtime, failing_rows())

    with pytest.raises(ValueError, match="different lengths"):
        materialize_metadata(runtime, MetadataTask(output="metadata.json"))

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
