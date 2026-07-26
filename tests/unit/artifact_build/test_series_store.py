import gzip
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest

import datapipeline.artifacts.series as series_store
from datapipeline.artifacts.series import (
    SERIES_MANIFEST_VERSION,
    SeriesRow,
    load_series_manifest,
    open_series,
    prune_series_cache,
    read_series_rows,
    write_series_rows,
)


def _time(hour: int) -> datetime:
    return datetime(2024, 1, 1, hour=hour, tzinfo=timezone.utc)


def _row(
    hour: int,
    *,
    entity_key: tuple = (),
    features: dict[str, Any] | None = None,
    targets: dict[str, Any] | None = None,
    placeholder_ids: frozenset[str] = frozenset(),
) -> SeriesRow:
    return SeriesRow(
        time=_time(hour),
        entity_key=entity_key,
        features={} if features is None else features,
        targets={} if targets is None else targets,
        placeholder_ids=placeholder_ids,
    )


def _valid_manifest() -> dict[str, Any]:
    return {
        "version": SERIES_MANIFEST_VERSION,
        "format": "jsonl.gz",
        "cadence": "1h",
        "sample_keys": ["security_id"],
        "sample_key_types": ["string"],
        "path": "manifest.data/current/series.jsonl.gz",
        "rows": 1,
        "sha256": "0" * 64,
        "features": [{"id": "price", "samples": 1}],
        "targets": [],
    }


def _write_manifest(
    root: Path,
    rows: list[SeriesRow],
    *,
    features: list[dict[str, object]],
    targets: list[dict[str, object]] | None = None,
    sample_keys: list[str] | None = None,
    sample_key_types: list[str] | None = None,
) -> Path:
    manifest_path = root / "manifest.json"
    data_path = root / "manifest.data/current/series.jsonl.gz"
    result = write_series_rows(data_path, rows)
    payload = {
        "version": SERIES_MANIFEST_VERSION,
        "format": "jsonl.gz",
        "cadence": "1h",
        "sample_keys": [] if sample_keys is None else sample_keys,
        "sample_key_types": ([] if sample_key_types is None else sample_key_types),
        "path": str(data_path.relative_to(root)),
        "rows": result.rows,
        "sha256": result.sha256,
        "features": features,
        "targets": [] if targets is None else targets,
    }
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return manifest_path


def test_series_rows_round_trip_json_native_values(tmp_path: Path) -> None:
    rows = [
        _row(
            0,
            entity_key=("A", 7, "north"),
            features={
                "record": {"values": [None, True, 1, 1.5, "text"]},
                "null_record": None,
            },
            targets={"sequence": [None, False, 2, 2.5, "other"]},
        ),
        _row(
            1,
            entity_key=("B", 8, "south"),
            features={"record": 3.0},
            placeholder_ids=frozenset({"record"}),
        ),
    ]
    destination = tmp_path / "series.jsonl.gz"

    written = write_series_rows(destination, rows)

    assert written.rows == 2
    assert len(written.sha256) == 64
    assert list(read_series_rows(destination, expected_rows=2)) == rows


def test_series_row_preserves_wire_shape(tmp_path: Path) -> None:
    destination = tmp_path / "series.jsonl.gz"
    write_series_rows(
        destination,
        [
            _row(
                0,
                entity_key=("AAPL",),
                features={"price": 1.0},
                targets={"return": [0.1, 0.2]},
            )
        ],
    )

    with gzip.open(destination, "rt", encoding="utf-8") as handle:
        payload = json.loads(handle.read())

    assert list(payload) == [
        "time",
        "entity_key",
        "features",
        "targets",
        "placeholder_ids",
    ]
    assert payload == {
        "time": "2024-01-01T00:00:00Z",
        "entity_key": ["AAPL"],
        "features": {"price": 1.0},
        "targets": {"return": [0.1, 0.2]},
        "placeholder_ids": [],
    }


def test_series_gzip_is_reproducible(tmp_path: Path) -> None:
    rows = [_row(0, features={"price": 1.0})]
    first = tmp_path / "first.jsonl.gz"
    second = tmp_path / "second.jsonl.gz"

    first_result = write_series_rows(first, rows)
    second_result = write_series_rows(second, rows)

    assert first.read_bytes() == second.read_bytes()
    assert first_result == second_result
    assert first_result.rows == 1


def test_series_writer_uses_compression_level_three(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    compression_levels = []
    sink_type = series_store.GzipBinarySink

    def open_sink(
        path: Path,
        compression_level: int,
        overwrite: bool = True,
    ) -> series_store.GzipBinarySink:
        compression_levels.append(compression_level)
        return sink_type(path, compression_level, overwrite)

    monkeypatch.setattr(series_store, "GzipBinarySink", open_sink)

    write_series_rows(
        tmp_path / "series.jsonl.gz",
        [_row(0, features={"price": 1.0})],
    )

    assert compression_levels == [3]


def test_series_writer_aborts_when_atomic_commit_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fail_commit(*_args: object) -> None:
        raise OSError("commit failed")

    monkeypatch.setattr(
        "datapipeline.io.sinks.files._commit_temp_file",
        fail_commit,
    )
    destination = tmp_path / "series.jsonl.gz"

    with pytest.raises(OSError, match="commit failed"):
        write_series_rows(destination, [_row(0, features={"price": 1.0})])

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "value",
    [
        {"nested": [Decimal("1.25")]},
        (1, 2),
        {1: "one"},
    ],
    ids=["unsupported-object", "tuple", "non-string-mapping-key"],
)
def test_series_writer_rejects_lossy_values_atomically(
    tmp_path: Path,
    value: object,
) -> None:
    destination = tmp_path / "series.jsonl.gz"
    destination.write_bytes(b"existing")

    with pytest.raises(TypeError, match="Series"):
        write_series_rows(
            destination,
            [
                _row(0, features={"valid": 1.0}),
                _row(1, features={"invalid": value}),
            ],
        )

    assert destination.read_bytes() == b"existing"
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize(
    ("row", "error", "message"),
    [
        (
            cast(Any, object()),
            TypeError,
            "Expected SeriesRow",
        ),
        (
            SeriesRow(
                time=datetime(2024, 1, 1),
                entity_key=(),
                features={},
                targets={},
                placeholder_ids=frozenset(),
            ),
            ValueError,
            "timezone-aware",
        ),
        (
            SeriesRow(
                time=_time(0),
                entity_key=cast(Any, "AAPL"),
                features={},
                targets={},
                placeholder_ids=frozenset(),
            ),
            TypeError,
            "entity key must be a tuple",
        ),
        (
            SeriesRow(
                time=_time(0),
                entity_key=(),
                features=cast(Any, ()),
                targets={},
                placeholder_ids=frozenset(),
            ),
            TypeError,
            "features and targets must be dictionaries",
        ),
        (
            SeriesRow(
                time=_time(0),
                entity_key=(),
                features={"price": 1.0},
                targets={},
                placeholder_ids=cast(Any, ["price"]),
            ),
            TypeError,
            "placeholder IDs must be a frozenset",
        ),
    ],
    ids=[
        "wrong-type",
        "naive-time",
        "non-tuple-key",
        "non-dict-values",
        "non-frozenset-placeholders",
    ],
)
def test_series_writer_rejects_invalid_row_contract(
    tmp_path: Path,
    row: SeriesRow,
    error: type[Exception],
    message: str,
) -> None:
    destination = tmp_path / "series.jsonl.gz"

    with pytest.raises(error, match=message):
        write_series_rows(destination, [row])

    assert not destination.exists()


def test_series_writer_rejects_non_scalar_entity_key(tmp_path: Path) -> None:
    destination = tmp_path / "series.jsonl.gz"

    with pytest.raises(TypeError, match="Sample key field.*list"):
        write_series_rows(
            destination,
            [_row(0, entity_key=(["A"],), features={"price": 1.0})],
        )

    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_series_writer_rejects_non_finite_values_atomically(
    tmp_path: Path,
    value: float,
) -> None:
    destination = tmp_path / "series.jsonl.gz"
    destination.write_bytes(b"existing")
    row = _row(0, features={"price": {"values": [1.0, value]}})

    with pytest.raises(ValueError, match="Out of range float values"):
        write_series_rows(destination, [row])

    assert row.features["price"] == {"values": [1.0, value]}
    assert destination.read_bytes() == b"existing"
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_series_writer_rejects_non_finite_entity_keys(
    tmp_path: Path,
    value: float,
) -> None:
    destination = tmp_path / "series.jsonl.gz"

    with pytest.raises(ValueError, match="finite float"):
        write_series_rows(
            destination,
            [_row(0, entity_key=(value,), features={"price": 1.0})],
        )

    assert not destination.exists()


def test_series_writer_removes_temp_file_on_interrupt(tmp_path: Path) -> None:
    class InterruptedRows:
        def __init__(self) -> None:
            self.count = 0

        def __iter__(self) -> "InterruptedRows":
            return self

        def __next__(self) -> SeriesRow:
            if self.count == 0:
                self.count += 1
                return _row(0, features={"price": 1.0})
            raise KeyboardInterrupt

    destination = tmp_path / "series.jsonl.gz"

    with pytest.raises(KeyboardInterrupt):
        write_series_rows(destination, InterruptedRows())

    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "version",
    [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 8.0, True],
)
def test_series_manifest_rejects_incompatible_version(
    tmp_path: Path,
    version: object,
) -> None:
    manifest = tmp_path / "manifest.json"
    payload = {} if version is None else {"version": version}
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="FORCE mode"):
        load_series_manifest(manifest)


def test_cache_pruning_rejects_symlinked_manifest_parent(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    artifacts_root.mkdir()
    outside = tmp_path / "outside"
    cache_root = outside / "manifest.data"
    stale = cache_root / "stale"
    stale.mkdir(parents=True)
    sentinel = stale / "keep"
    sentinel.write_text("keep", encoding="utf-8")
    (outside / "manifest.json").write_text(
        json.dumps(_valid_manifest()),
        encoding="utf-8",
    )
    (artifacts_root / "build").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="must stay under artifacts root"):
        prune_series_cache(
            artifacts_root / "build/manifest.json",
            artifacts_root,
        )

    assert sentinel.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize(
    ("change", "value"),
    [
        ("format", "jsonl"),
        ("cadence", "hourly"),
        ("sample_keys", ["security_id", "security_id"]),
        ("sample_key_types", []),
        ("rows", True),
        ("rows", -1),
        ("path", "../outside.jsonl.gz"),
        ("sha256", "not-a-digest"),
        ("samples", True),
        ("samples", -1),
    ],
)
def test_series_manifest_rejects_invalid_contract(
    tmp_path: Path,
    change: str,
    value: object,
) -> None:
    payload = _valid_manifest()
    if change == "samples":
        payload["features"][0]["samples"] = value
    else:
        payload[change] = value
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid series manifest"):
        load_series_manifest(manifest)


@pytest.mark.parametrize("duplicate_role", ["feature", "target"])
def test_series_manifest_rejects_duplicate_ids(
    tmp_path: Path,
    duplicate_role: str,
) -> None:
    payload = _valid_manifest()
    duplicate = {"id": "price", "samples": 1}
    if duplicate_role == "feature":
        payload["features"].append(duplicate)
    else:
        payload["targets"].append(duplicate)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid series manifest"):
        load_series_manifest(manifest)


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (
            {
                "entity_key": [],
                "features": {},
                "targets": {},
            },
            "define 'time'",
        ),
        (
            {
                "time": "2024-01-01T00:00:00Z",
                "features": {},
                "targets": {},
            },
            "list 'entity_key'",
        ),
        (
            {
                "time": "2024-01-01T00:00:00Z",
                "entity_key": None,
                "features": {},
                "targets": {},
            },
            "list 'entity_key'",
        ),
        (
            {
                "time": "2024-01-01T00:00:00Z",
                "entity_key": [["A"]],
                "features": {},
                "targets": {},
            },
            "invalid entity key",
        ),
        (
            {
                "time": "2024-01-01T00:00:00Z",
                "entity_key": [],
                "features": [],
                "targets": {},
            },
            "object 'features'",
        ),
        (
            {
                "time": "2024-01-01T00:00:00Z",
                "entity_key": [],
                "features": {},
                "targets": None,
            },
            "object 'targets'",
        ),
        (
            {
                "time": "2024-01-01T00:00:00Z",
                "entity_key": [],
                "features": {"price": 1.0},
                "targets": {"price": 2.0},
            },
            "both features and targets",
        ),
    ],
    ids=[
        "missing-time",
        "missing-entity-key",
        "null-entity-key",
        "nested-key",
        "invalid-features",
        "invalid-targets",
        "duplicate-role",
    ],
)
def test_series_reader_rejects_malformed_rows(
    tmp_path: Path,
    row: dict[str, object],
    message: str,
) -> None:
    destination = tmp_path / "series.jsonl.gz"
    with gzip.open(destination, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")

    with pytest.raises(ValueError, match=message):
        list(read_series_rows(destination))


@pytest.mark.parametrize(
    ("change", "value", "message"),
    [
        ("placeholder_ids", None, "list 'placeholder_ids'"),
        ("placeholder_ids", "price", "list 'placeholder_ids'"),
        ("placeholder_ids", [""], "invalid placeholder ID"),
        ("placeholder_ids", ["price", "price"], "duplicate placeholder IDs"),
        ("placeholder_ids", ["unknown"], "placeholder IDs without values"),
    ],
)
def test_series_reader_rejects_invalid_placeholder_ids(
    tmp_path: Path,
    change: str,
    value: object,
    message: str,
) -> None:
    row = {
        "time": "2024-01-01T00:00:00Z",
        "entity_key": [],
        "features": {"price": 1.0},
        "targets": {"return": 0.1},
        "placeholder_ids": [],
    }
    row[change] = value
    destination = tmp_path / "series.jsonl.gz"
    with gzip.open(destination, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")

    with pytest.raises(ValueError, match=message):
        list(read_series_rows(destination))


def test_series_writer_rejects_invalid_placeholder_ids(tmp_path: Path) -> None:
    destination = tmp_path / "series.jsonl.gz"

    with pytest.raises(ValueError, match="must reference row values"):
        write_series_rows(
            destination,
            [
                _row(
                    0,
                    features={"price": 1.0},
                    placeholder_ids=frozenset({"unknown"}),
                )
            ],
        )

    assert not destination.exists()


def test_series_reader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    destination = tmp_path / "series.jsonl.gz"
    with gzip.open(destination, "wt", encoding="utf-8") as handle:
        handle.write(
            '{"time":"2024-01-01T00:00:00Z","entity_key":[],'
            '"features":{"price":1,"price":2},"targets":{}}\n'
        )

    with pytest.raises(ValueError, match="Duplicate JSON key 'price'"):
        list(read_series_rows(destination))


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_series_reader_rejects_non_standard_numbers(
    tmp_path: Path,
    value: str,
) -> None:
    destination = tmp_path / "series.jsonl.gz"
    with gzip.open(destination, "wt", encoding="utf-8") as handle:
        handle.write(
            '{"time":"2024-01-01T00:00:00Z","entity_key":[],'
            f'"features":{{"price":{value}}},"targets":{{}}}}\n'
        )

    with pytest.raises(ValueError, match="Non-standard JSON"):
        list(read_series_rows(destination))


def test_series_reader_rejects_fewer_rows_than_declared(tmp_path: Path) -> None:
    destination = tmp_path / "series.jsonl.gz"
    write_series_rows(destination, [_row(0, features={"price": 1.0})])

    with pytest.raises(ValueError, match="declares 2 rows but contains 1"):
        list(read_series_rows(destination, expected_rows=2))


def test_series_reader_rejects_extra_row_before_yielding_it(tmp_path: Path) -> None:
    destination = tmp_path / "series.jsonl.gz"
    rows = [
        _row(0, features={"price": 1.0}),
        _row(1, features={"price": 2.0}),
    ]
    write_series_rows(destination, rows)
    opened = read_series_rows(destination, expected_rows=1)

    assert next(opened) == rows[0]
    with pytest.raises(ValueError, match="more than its declared 1 rows"):
        next(opened)


def test_series_reader_does_not_verify_unread_rows(tmp_path: Path) -> None:
    destination = tmp_path / "series.jsonl.gz"
    row = _row(0, features={"price": 1.0})
    write_series_rows(destination, [row])
    opened = read_series_rows(destination, expected_rows=2)

    assert next(opened) == row
    opened.close()


def test_open_series_validates_order_keys_and_per_series_counts(
    tmp_path: Path,
) -> None:
    rows = [
        _row(
            0,
            entity_key=("A",),
            features={
                "price__@ticker:A": 1.0,
                "price__@ticker:B": 2.0,
            },
            targets={"return": 0.1},
        ),
        _row(
            1,
            entity_key=("A",),
            features={"price__@ticker:A": 3.0},
            targets={"return": 0.2},
        ),
    ]
    manifest_path = _write_manifest(
        tmp_path,
        rows,
        features=[{"id": "price", "samples": 2}],
        targets=[{"id": "return", "samples": 2}],
        sample_keys=["security_id"],
        sample_key_types=["string"],
    )

    assert list(open_series(manifest_path)) == rows


def test_open_series_rejects_unordered_rows(tmp_path: Path) -> None:
    rows = [
        _row(1, features={"price": 2.0}),
        _row(0, features={"price": 1.0}),
    ]
    manifest_path = _write_manifest(
        tmp_path,
        rows,
        features=[{"id": "price", "samples": 2}],
    )

    with pytest.raises(ValueError, match="not strictly ordered"):
        list(open_series(manifest_path))


def test_open_series_rejects_invalid_sample_key_type(tmp_path: Path) -> None:
    manifest_path = _write_manifest(
        tmp_path,
        [_row(0, entity_key=(7,), features={"price": 1.0})],
        features=[{"id": "price", "samples": 1}],
        sample_keys=["security_id"],
        sample_key_types=["string"],
    )

    with pytest.raises(TypeError, match="security_id.*string"):
        list(open_series(manifest_path))


def test_open_series_rejects_unexpected_series_id(tmp_path: Path) -> None:
    manifest_path = _write_manifest(
        tmp_path,
        [_row(0, features={"volume": 1.0})],
        features=[{"id": "price", "samples": 1}],
    )

    with pytest.raises(ValueError, match="unexpected series id 'volume'"):
        list(open_series(manifest_path))


def test_open_series_rejects_changed_wide_value_with_valid_counts(
    tmp_path: Path,
) -> None:
    original = _row(
        0,
        features={
            "price__@ticker:A": 1.0,
            "price__@ticker:B": 2.0,
        },
    )
    manifest_path = _write_manifest(
        tmp_path,
        [original],
        features=[{"id": "price", "samples": 1}],
    )
    manifest = load_series_manifest(manifest_path)
    data_path = manifest_path.parent / manifest.path
    write_series_rows(
        data_path,
        [_row(0, features={"price__@ticker:A": 1.0})],
    )

    with pytest.raises(ValueError, match="content digest"):
        list(open_series(manifest_path, manifest))


@pytest.mark.parametrize(
    ("role", "features", "targets", "message"),
    [
        (
            "feature",
            [{"id": "price", "samples": 2}],
            [],
            "incorrect feature sample counts",
        ),
        (
            "target",
            [],
            [{"id": "return", "samples": 2}],
            "incorrect target sample counts",
        ),
    ],
)
def test_open_series_rejects_incorrect_per_series_count(
    tmp_path: Path,
    role: str,
    features: list[dict[str, object]],
    targets: list[dict[str, object]],
    message: str,
) -> None:
    values = {"price": 1.0} if role == "feature" else {}
    target_values = {"return": 0.1} if role == "target" else {}
    manifest_path = _write_manifest(
        tmp_path,
        [_row(0, features=values, targets=target_values)],
        features=features,
        targets=targets,
    )

    with pytest.raises(ValueError, match=message):
        list(open_series(manifest_path))


def test_prune_series_cache_removes_only_stale_generations(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    manifest_path = artifacts_root / "build/manifest.json"
    current = manifest_path.parent / "manifest.data/current"
    stale = manifest_path.parent / "manifest.data/stale"
    current.mkdir(parents=True)
    stale.mkdir()
    (current / "series.jsonl.gz").write_bytes(b"current")
    (stale / "series.jsonl.gz").write_bytes(b"stale")
    manifest_path.write_text(json.dumps(_valid_manifest()), encoding="utf-8")

    removed = prune_series_cache(manifest_path, artifacts_root)

    assert removed == (stale,)
    assert current.is_dir()
    assert not stale.exists()
