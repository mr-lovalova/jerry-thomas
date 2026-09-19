from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Generic, TypeVar

from jerrythomas.artifacts.models import (
    VectorMetadata,
    CoverageStatsArtifact,
)
from jerrythomas.artifacts.scaler import ScalerArtifact, load_scaler_artifact
from jerrythomas.artifacts.specs import (
    SCALER_STATISTICS,
    VECTOR_METADATA,
    COVERAGE_STATS,
)
from jerrythomas.io.json_file import read_json_object

ArtifactValue = TypeVar("ArtifactValue")


ArtifactLoader = Callable[[Path], ArtifactValue]


@dataclass(frozen=True)
class ArtifactSpec(Generic[ArtifactValue]):
    key: str
    loader: ArtifactLoader[ArtifactValue]


@dataclass(frozen=True)
class ArtifactRecord:
    relative_path: str
    meta: Mapping[str, Any]

    def resolve(self, root: Path) -> Path:
        return root / self.relative_path


class ArtifactNotRegisteredError(RuntimeError):
    """Raised when attempting to use an artifact that is not registered."""


class ArtifactRegistry:
    """Registered build artifacts available to a runtime."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._records: dict[str, ArtifactRecord] = {}
        self._loaded: dict[str, Any] = {}

    @property
    def root(self) -> Path:
        return self._root

    def register(
        self,
        key: str,
        relative_path: str,
        meta: Mapping[str, Any] | None = None,
    ) -> None:
        self._loaded.pop(key, None)
        self._records[key] = ArtifactRecord(
            relative_path=relative_path,
            meta=MappingProxyType(dict(meta or {})),
        )

    def clear(self) -> None:
        self._records.clear()
        self._loaded.clear()

    def registrations(self) -> dict[str, ArtifactRecord]:
        """Copy registered references without serializing loaded artifacts."""
        return {
            key: ArtifactRecord(record.relative_path, deepcopy(dict(record.meta)))
            for key, record in self._records.items()
        }

    def has(self, key: str) -> bool:
        return key in self._records

    def require(self, key: str) -> ArtifactRecord:
        try:
            return self._records[key]
        except KeyError as exc:
            raise ArtifactNotRegisteredError(
                f"Artifact '{key}' is not registered. "
                "Run `jerry build --project <project.yaml>` first."
            ) from exc

    def optional(self, key: str) -> ArtifactRecord | None:
        return self._records.get(key)

    def resolve_path(self, key: str) -> Path:
        return self.require(key).resolve(self._root)

    def load(self, spec: ArtifactSpec[ArtifactValue]) -> ArtifactValue:
        if spec.key in self._loaded:
            return self._loaded[spec.key]
        path = self.resolve_path(spec.key)
        try:
            value = spec.loader(path)
        except FileNotFoundError as exc:
            message = (
                f"Artifact file not found: {path}. "
                "Run `jerry build --project <project.yaml>` to regenerate it."
            )
            raise RuntimeError(message) from exc
        self._loaded[spec.key] = value
        return value


def _read_vector_metadata(path: Path) -> VectorMetadata:
    return VectorMetadata.model_validate(read_json_object(path))


def _read_coverage_stats(path: Path) -> CoverageStatsArtifact:
    payload = read_json_object(path)
    version = payload.get("schema_version")
    if version != 3:
        raise ValueError(
            f"Unsupported coverage stats schema version {version!r}. "
            "Rebuild coverage_stats in rebuild mode."
        )
    return CoverageStatsArtifact.model_validate(payload)


VECTOR_METADATA_SPEC = ArtifactSpec[VectorMetadata](
    key=VECTOR_METADATA,
    loader=_read_vector_metadata,
)

SCALER_SPEC = ArtifactSpec[ScalerArtifact](
    key=SCALER_STATISTICS,
    loader=load_scaler_artifact,
)

COVERAGE_STATS_SPEC = ArtifactSpec[CoverageStatsArtifact](
    key=COVERAGE_STATS,
    loader=_read_coverage_stats,
)
