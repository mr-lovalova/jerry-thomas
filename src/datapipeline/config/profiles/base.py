from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from datapipeline.config.observability import ObservabilityConfig

ProfileCommand = Literal["serve", "build", "inspect", "materialize"]


class Profile(BaseModel):
    """User-facing run/build bundling and policy."""

    model_config = ConfigDict(extra="forbid")

    cmd: ProfileCommand
    name: str
    order: int | None = Field(default=None, ge=0)
    enabled: bool = Field(default=True)
    observability: ObservabilityConfig | None = None

    @field_validator("name", mode="before")
    @classmethod
    def _normalize_name(cls, value: object) -> str:
        text = str(value).strip() if value is not None else ""
        if not text:
            raise ValueError("profile name must be set")
        if text in {".", ".."}:
            raise ValueError("profile name must not be '.' or '..'")
        return text


class OperationProfile(Profile):
    operation: str

    @field_validator("operation", mode="before")
    @classmethod
    def _normalize_operation(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("operation must be a string")
        operation = value.strip().lower()
        if not operation:
            raise ValueError("operation must be set")
        return operation
