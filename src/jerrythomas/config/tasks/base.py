from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_core import PydanticSerializationError


class Task(BaseModel):
    """Concrete executable operation."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["artifact", "runtime"]
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
    # YAML uses path; Python callers and serialized artifact contracts retain output.
    model_config = ConfigDict(populate_by_name=True)

    kind: Literal["artifact"] = Field(default="artifact")
    output: str = Field(validation_alias="path")

    @field_validator("output")
    @classmethod
    def _validate_output(cls, output: str) -> str:
        output = output.strip()
        output_path = Path(output)
        if not output or output_path == Path("."):
            raise ValueError("path must name a file under the artifacts root")
        if output_path.is_absolute():
            raise ValueError("path must be a relative path under artifacts root")
        if ".." in output_path.parts:
            raise ValueError("path must not traverse outside artifacts root")
        if output_path.parts[0].rstrip(" .").casefold() == "_system":
            raise ValueError("path must not use the reserved '_system' directory")
        return output


class RuntimeTask(Task):
    kind: Literal["runtime"] = Field(default="runtime")
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


class PluginRuntimeTask(RuntimeTask):
    options: dict[str, object] = Field(default_factory=dict)

    @field_validator("options", mode="before")
    @classmethod
    def _require_options_mapping(cls, value):
        if isinstance(value, Mapping):
            return value
        raise ValueError("options must be a mapping")
