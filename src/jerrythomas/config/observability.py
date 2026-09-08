from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from jerrythomas.config.options import LOG_SCOPE_CHOICES, LOG_TRANSPORT_CHOICES

VALID_LOG_LEVELS = ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG")
VALID_LOG_TRANSPORTS = tuple(value.upper() for value in LOG_TRANSPORT_CHOICES)
VALID_LOG_SCOPES = tuple(value.upper() for value in LOG_SCOPE_CHOICES)


class LogOutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transport: str = Field(..., description="STDERR | STDOUT | FS")
    scope: str = Field(
        default="GLOBAL",
        description="GLOBAL | EXECUTION",
    )
    path: str | None = Field(
        default=None,
        description="Path when transport=FS",
    )

    @field_validator("transport", mode="before")
    @classmethod
    def _normalize_transport(cls, value):
        if value is None:
            return None
        name = str(value).upper()
        if name not in VALID_LOG_TRANSPORTS:
            raise ValueError(
                f"transport must be one of {', '.join(VALID_LOG_TRANSPORTS)}"
            )
        return name

    @field_validator("scope", mode="before")
    @classmethod
    def _normalize_scope(cls, value):
        if value is None:
            return "GLOBAL"
        name = str(value).upper()
        if name not in VALID_LOG_SCOPES:
            raise ValueError(f"scope must be one of {', '.join(VALID_LOG_SCOPES)}")
        return name

    @field_validator("path", mode="before")
    @classmethod
    def _normalize_path(cls, value):
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @model_validator(mode="after")
    def _validate_scope_and_path(self):
        if self.transport == "FS":
            if self.scope == "EXECUTION":
                if self.path is not None and Path(self.path).is_absolute():
                    raise ValueError("path must be relative when scope=EXECUTION")
            elif self.path is None:
                raise ValueError("path must be set when transport=FS and scope=GLOBAL")
            return self
        if self.scope != "GLOBAL":
            raise ValueError("scope=EXECUTION requires transport=FS")
        if self.path is not None:
            raise ValueError("path is only valid when transport=FS")
        return self


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: str | None = Field(
        default=None,
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    outputs: list[LogOutputConfig] | None = Field(
        default=None,
        description="Ordered list of logging outputs.",
    )

    @field_validator("level", mode="before")
    @classmethod
    def _validate_level(cls, value: str | None) -> str | None:
        if value is None:
            return None
        name = str(value).upper()
        if name not in VALID_LOG_LEVELS:
            raise ValueError(f"level must be one of {', '.join(VALID_LOG_LEVELS)}")
        return name


class ObservabilityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    visuals: bool | None = Field(
        default=None,
        strict=True,
        description="Enable terminal visuals.",
    )
    heartbeat_interval_seconds: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
        description="Pipeline heartbeat interval in seconds. Set to 0 to disable.",
    )
    logging: LoggingConfig | None = Field(
        default=None,
        description="Logging settings.",
    )
