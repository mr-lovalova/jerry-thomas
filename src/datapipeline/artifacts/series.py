import gzip
import hashlib
import json
import shutil
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Final, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from datapipeline.domain.sample_key import (
    SampleKeyContract,
    SampleKeyValueType,
    sample_key_value_type,
)
from datapipeline.domain.series_id import base_id
from datapipeline.io.sinks.files import GzipBinarySink
from datapipeline.services.path_policy import resolve_artifact_output_path
from datapipeline.utils.time import CADENCE_PATTERN, parse_datetime

SERIES_MANIFEST_VERSION: Final = 9
_JSON_SCALAR_TYPES = {type(None), bool, int, float, str}
_NonEmptyString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]
_Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class SeriesEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: _NonEmptyString
    samples: int = Field(strict=True, ge=0)


class SeriesManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal[9] = SERIES_MANIFEST_VERSION
    format: Literal["jsonl.gz"] = "jsonl.gz"
    cadence: str = Field(pattern=CADENCE_PATTERN)
    sample_keys: tuple[_NonEmptyString, ...] = ()
    sample_key_types: tuple[SampleKeyValueType, ...] = ()
    path: _NonEmptyString
    rows: int = Field(strict=True, ge=0)
    sha256: _Sha256
    features: tuple[SeriesEntry, ...] = ()
    targets: tuple[SeriesEntry, ...] = ()

    @field_validator("sample_keys")
    @classmethod
    def validate_unique_sample_keys(cls, keys: tuple[str, ...]) -> tuple[str, ...]:
        if len(keys) != len(set(keys)):
            raise ValueError("sample keys must be unique")
        return keys

    @field_validator("path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("series data path must be relative to the manifest")
        return str(path)

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        if len(self.sample_key_types) != len(self.sample_keys):
            raise ValueError("sample key type count must match sample keys")
        identifiers = [entry.id for entry in (*self.features, *self.targets)]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("series ids must be unique across features and targets")
        return self


@dataclass
class SeriesRow:
    time: datetime
    entity_key: tuple
    features: dict[str, Any]
    targets: dict[str, Any]
    placeholder_ids: frozenset[str]

    @property
    def key(self) -> tuple:
        return self.time, *self.entity_key


@dataclass(frozen=True)
class SeriesWriteResult:
    rows: int
    sha256: str


def _to_iso(value: datetime) -> str:
    text = value.isoformat()
    if text.endswith("+00:00"):
        return text[:-6] + "Z"
    return text


def _require_json_value(value: Any) -> None:
    value_type = type(value)
    if value_type in _JSON_SCALAR_TYPES:
        return
    if value_type is list:
        for item in value:
            _require_json_value(item)
        return
    if value_type is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(
                    f"Series mappings require string keys; got {type(key).__name__}."
                )
            _require_json_value(item)
        return
    raise TypeError(
        f"Series rows require JSON-native values; got {value_type.__name__}."
    )


def write_series_rows(path: Path, rows: Iterable[SeriesRow]) -> SeriesWriteResult:
    sink = GzipBinarySink(path)
    hasher = hashlib.sha256()
    count = 0
    try:
        for row in rows:
            placeholder_ids = _validate_series_row(row)
            payload = {
                "time": _to_iso(row.time),
                "entity_key": list(row.entity_key),
                "features": row.features,
                "targets": row.targets,
                "placeholder_ids": [
                    series_id
                    for series_id in (*row.features, *row.targets)
                    if series_id in placeholder_ids
                ],
            }
            _require_json_value(payload)
            line = (
                json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n"
            ).encode("utf-8")
            hasher.update(line)
            sink.write_bytes(line)
            count += 1
        sink.close()
    except BaseException:
        sink.abort()
        raise
    return SeriesWriteResult(rows=count, sha256=hasher.hexdigest())


def _validate_series_row(
    row: SeriesRow,
) -> frozenset[str]:
    if not isinstance(row, SeriesRow):
        raise TypeError(f"Expected SeriesRow; got {type(row).__name__}.")
    if row.time.tzinfo is None:
        raise ValueError("Series row time must be timezone-aware.")
    if type(row.entity_key) is not tuple:
        raise TypeError("Series row entity key must be a tuple.")
    for index, component in enumerate(row.entity_key):
        sample_key_value_type(f"entity_key[{index}]", component)
    if type(row.features) is not dict or type(row.targets) is not dict:
        raise TypeError("Series row features and targets must be dictionaries.")
    duplicate_ids = row.features.keys() & row.targets.keys()
    if duplicate_ids:
        raise ValueError(
            "Series row contains IDs as both features and targets: "
            + ", ".join(sorted(duplicate_ids))
        )
    if not isinstance(row.placeholder_ids, frozenset):
        raise TypeError("Series row placeholder IDs must be a frozenset.")
    unexpected = row.placeholder_ids - (row.features.keys() | row.targets.keys())
    if unexpected:
        raise ValueError("Series row placeholder IDs must reference row values.")
    return row.placeholder_ids


def load_series_manifest(path: Path) -> SeriesManifest:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected series manifest object in '{path}'.")
    version = payload.get("version")
    if type(version) is not int or version != SERIES_MANIFEST_VERSION:
        raise ValueError(
            f"Unsupported series manifest version {version!r} in '{path}'. "
            "Rebuild series and dependent artifacts in FORCE mode."
        )
    try:
        manifest = SeriesManifest.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(
            f"Invalid series manifest '{path}'. Rebuild series and "
            "dependent artifacts in FORCE mode."
        ) from exc

    root = path.parent.resolve()
    try:
        (root / manifest.path).resolve().relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Series data '{manifest.path}' escapes manifest directory '{root}'."
        ) from exc
    return manifest


def open_series(
    manifest_path: Path,
    manifest: SeriesManifest | None = None,
) -> Iterator[SeriesRow]:
    contract = load_series_manifest(manifest_path) if manifest is None else manifest
    path = manifest_path.parent / contract.path
    feature_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    expected_features = {entry.id: entry.samples for entry in contract.features}
    expected_targets = {entry.id: entry.samples for entry in contract.targets}
    sample_keys = SampleKeyContract(contract.sample_keys, contract.sample_key_types)
    previous_key: tuple | None = None

    for row in read_series_rows(
        path,
        expected_rows=contract.rows,
        expected_sha256=contract.sha256,
    ):
        sample_keys.validate(row.entity_key)
        if previous_key is not None and row.key <= previous_key:
            raise ValueError(
                f"Series rows in '{path}' are not strictly ordered: "
                f"{row.key!r} follows {previous_key!r}."
            )
        previous_key = row.key
        _observe_series_ids(row.features, expected_features, feature_counts, path)
        _observe_series_ids(row.targets, expected_targets, target_counts, path)
        yield row

    feature_mismatches = [
        f"{series_id}={feature_counts.get(series_id, 0)} (declared {count})"
        for series_id, count in expected_features.items()
        if feature_counts.get(series_id, 0) != count
    ]
    if feature_mismatches:
        raise ValueError(
            f"Series data '{path}' has incorrect feature sample counts: "
            + ", ".join(feature_mismatches)
        )

    target_mismatches = [
        f"{series_id}={target_counts.get(series_id, 0)} (declared {count})"
        for series_id, count in expected_targets.items()
        if target_counts.get(series_id, 0) != count
    ]
    if target_mismatches:
        raise ValueError(
            f"Series data '{path}' has incorrect target sample counts: "
            + ", ".join(target_mismatches)
        )


def _observe_series_ids(
    values: Mapping[str, Any],
    expected: Mapping[str, int],
    counts: Counter[str],
    path: Path,
) -> None:
    observed: set[str] = set()
    for series_id in values:
        root_id = base_id(series_id)
        if root_id not in expected:
            raise ValueError(
                f"Series data '{path}' contains unexpected series id '{series_id}'."
            )
        observed.add(root_id)
    counts.update(observed)


def read_series_rows(
    path: Path,
    expected_rows: int | None = None,
    expected_sha256: str | None = None,
) -> Iterator[SeriesRow]:
    hasher = hashlib.sha256()
    rows = 0
    with gzip.open(path, "rb") as fh:
        for line in fh:
            if not line.strip():
                continue
            hasher.update(line)
            payload = json.loads(
                line,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_json_constant,
            )
            if not isinstance(payload, dict):
                raise ValueError(f"Expected series row object in '{path}'.")
            rows += 1
            if expected_rows is not None and rows > expected_rows:
                raise ValueError(
                    f"Series data '{path}' contains more than its "
                    f"declared {expected_rows} rows."
                )
            yield _payload_to_series_row(payload, path)
    if expected_rows is not None and rows != expected_rows:
        raise ValueError(
            f"Series data '{path}' declares {expected_rows} rows but contains {rows}."
        )
    actual_sha256 = hasher.hexdigest()
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(
            f"Series data '{path}' has content digest {actual_sha256}, "
            f"expected {expected_sha256}."
        )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"Duplicate JSON key {key!r} in series data.")
        payload[key] = value
    return payload


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-standard JSON number {value!r} is not supported.")


def _payload_to_series_row(payload: Mapping[str, Any], path: Path) -> SeriesRow:
    raw_time = payload.get("time")
    if not isinstance(raw_time, str) or not raw_time.strip():
        raise ValueError(f"Series row in '{path}' must define 'time'.")
    time_value = parse_datetime(raw_time)
    entity_key = _entity_key(payload, path)
    features = _value_mapping(payload, "features", path)
    targets = _value_mapping(payload, "targets", path)
    duplicate_ids = features.keys() & targets.keys()
    if duplicate_ids:
        raise ValueError(
            f"Series row in '{path}' contains IDs as both features and targets: "
            + ", ".join(sorted(duplicate_ids))
        )
    placeholder_ids = _placeholder_ids(
        payload,
        features.keys() | targets.keys(),
        path,
    )
    return SeriesRow(
        time=time_value,
        entity_key=entity_key,
        features=features,
        targets=targets,
        placeholder_ids=placeholder_ids,
    )


def _value_mapping(
    payload: Mapping[str, Any],
    key: str,
    path: Path,
) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Series row in '{path}' must define object '{key}'.")
    if any(not isinstance(series_id, str) or not series_id for series_id in value):
        raise ValueError(f"Series row in '{path}' has an invalid series id.")
    return value


def _placeholder_ids(
    payload: Mapping[str, Any],
    values: set[str],
    path: Path,
) -> frozenset[str]:
    placeholder_ids = payload.get("placeholder_ids")
    if not isinstance(placeholder_ids, list):
        raise ValueError(f"Series row in '{path}' must define list 'placeholder_ids'.")
    if any(
        not isinstance(series_id, str) or not series_id for series_id in placeholder_ids
    ):
        raise ValueError(f"Series row in '{path}' has an invalid placeholder ID.")
    if len(placeholder_ids) != len(set(placeholder_ids)):
        raise ValueError(f"Series row in '{path}' has duplicate placeholder IDs.")
    unknown = set(placeholder_ids) - values
    if unknown:
        raise ValueError(
            f"Series row in '{path}' has placeholder IDs without values: "
            + ", ".join(sorted(unknown))
        )
    return frozenset(placeholder_ids)


def _entity_key(payload: Mapping[str, Any], path: Path) -> tuple:
    value = payload.get("entity_key")
    if not isinstance(value, list):
        raise ValueError(f"Series row in '{path}' must define list 'entity_key'.")
    for index, component in enumerate(value):
        try:
            sample_key_value_type(f"entity_key[{index}]", component)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Series row in '{path}' has an invalid entity key."
            ) from exc
    return tuple(value)


def prune_series_cache(
    manifest_path: Path,
    artifacts_root: Path,
) -> tuple[Path, ...]:
    """Remove generations unreachable from the current manifest."""

    manifest_path = resolve_artifact_output_path(manifest_path, artifacts_root)
    if not manifest_path.is_file():
        return ()
    manifest = load_series_manifest(manifest_path)
    cache_root = manifest_path.parent / f"{manifest_path.stem}.data"
    if not cache_root.exists():
        return ()
    if cache_root.is_symlink() or not cache_root.is_dir():
        raise RuntimeError(f"Series data path is not a directory: {cache_root}")

    parts = Path(manifest.path).parts
    if len(parts) < 3 or parts[0] != cache_root.name:
        return ()
    retained = parts[1]

    removed: list[Path] = []
    for path in cache_root.iterdir():
        if path.name == retained:
            continue
        if path.is_symlink() or not path.is_dir():
            path.unlink()
        else:
            shutil.rmtree(path)
        removed.append(path)
    return tuple(sorted(removed))
