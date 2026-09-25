from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_core import PydanticSerializationError


class Task(BaseModel):
    """Concrete executable operation."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["artifact", "output"]
    id: str
    entrypoint: str
    dataset: str | None = None

    @field_validator("id", mode="before")
    @classmethod
    def _normalize_id(cls, value):
        operation_id = str(value).strip().lower() if value is not None else ""
        if not operation_id:
            raise ValueError("id must be set")
        return operation_id

    @field_validator("dataset")
    @classmethod
    def _validate_dataset(cls, value: str | None) -> str | None:
        if value is not None and (not value or value != value.strip()):
            raise ValueError("dataset must be a nonempty dataset ID without whitespace")
        return value

    @field_validator("entrypoint", mode="before")
    @classmethod
    def _normalize_entrypoint(cls, value):
        entrypoint = str(value).strip() if value is not None else ""
        if not entrypoint:
            raise ValueError("entrypoint must be set")
        return entrypoint

    @model_validator(mode="after")
    def _validate_serializable_config(self):
        try:
            self.model_dump(mode="json")
        except PydanticSerializationError as exc:
            raise ValueError(
                "operation configuration must contain only JSON-serializable values"
            ) from exc
        return self


class ArtifactTask(Task):
    kind: Literal["artifact"] = Field(default="artifact")
    # Relative to the project's artifacts root.
    path: str

    @field_validator("path")
    @classmethod
    def _validate_path(cls, path: str) -> str:
        path = path.strip()
        relative = Path(path)
        if not path or relative == Path("."):
            raise ValueError("path must name a file under the artifacts root")
        if relative.is_absolute():
            raise ValueError("path must be a relative path under artifacts root")
        if ".." in relative.parts:
            raise ValueError("path must not traverse outside artifacts root")
        if relative.parts[0].rstrip(" .").casefold() == "_system":
            raise ValueError("path must not use the reserved '_system' directory")
        return path


class DatasetArtifactTask(ArtifactTask):
    """A built-in artifact of one dataset, stored at datasets/<id>/<file> by default."""

    default_filename: ClassVar[str]

    @model_validator(mode="before")
    @classmethod
    def _default_path(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("path") is None:
            dataset = data.get("dataset")
            path = (
                f"datasets/{dataset}/{cls.default_filename}"
                if isinstance(dataset, str) and dataset.strip()
                else cls.default_filename
            )
            return {**data, "path": path}
        return data


class OutputTask(Task):
    kind: Literal["output"] = Field(default="output")
    requires: tuple[str, ...] = ()

    @field_validator("requires", mode="before")
    @classmethod
    def _normalize_requires(cls, value: object) -> tuple[str, ...]:
        if value is None:
            return ()
        if not isinstance(value, (list, tuple)):
            raise ValueError("requires must be a list of artifact operation ids")
        requires: list[str] = []
        for item in value:
            operation_id = str(item).strip().lower() if item is not None else ""
            if not operation_id:
                raise ValueError("requires item must be set")
            requires.append(operation_id)
        if len(requires) != len(set(requires)):
            raise ValueError(
                "requires must not contain duplicate artifact operation ids"
            )
        return tuple(requires)


class PluginOutputTask(OutputTask):
    options: dict[str, object] = Field(default_factory=dict)

    @field_validator("options", mode="before")
    @classmethod
    def _require_options_mapping(cls, value):
        if isinstance(value, Mapping):
            return value
        raise ValueError("options must be a mapping")
