import logging
import math
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from jerrythomas.config.observability import LogOutputConfig, ObservabilityConfig
from jerrythomas.config.options import LOG_SCOPE_CHOICES, LOG_TRANSPORT_CHOICES

DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 60.0
LOG_TRANSPORT_SET = set(LOG_TRANSPORT_CHOICES)
LOG_SCOPE_SET = set(LOG_SCOPE_CHOICES)


def _normalize_lower(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    return text.lower() if text else None


def _normalize_upper(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    return text.upper() if text else None


def resolve_visuals(
    cli_visuals: str | None,
    config_visuals: str | None,
    default_visuals: str = "on",
) -> str:
    cli_value = _normalize_lower(cli_visuals)
    if cli_value is not None:
        return cli_value
    config_value = _normalize_lower(config_visuals)
    if config_value is not None:
        return config_value
    return default_visuals


def resolve_heartbeat_interval_seconds(
    value: float | None,
) -> float:
    if value is None:
        return DEFAULT_HEARTBEAT_INTERVAL_SECONDS
    interval = float(value)
    if not math.isfinite(interval):
        raise ValueError("heartbeat_interval_seconds must be finite")
    if interval < 0:
        raise ValueError("heartbeat_interval_seconds must be non-negative")
    if interval > threading.TIMEOUT_MAX:
        raise ValueError(
            f"heartbeat_interval_seconds must not exceed {threading.TIMEOUT_MAX:g}"
        )
    return interval


@dataclass(frozen=True)
class LogLevelDecision:
    name: str
    value: int


def resolve_log_level(
    cli_level: str | None,
    config_level: str | None = None,
    default_level: str = "INFO",
) -> LogLevelDecision:
    if cli_level is not None:
        name = _normalize_upper(cli_level)
    elif config_level is not None:
        name = _normalize_upper(config_level)
    else:
        name = _normalize_upper(default_level)
    if name is None:
        raise ValueError("log level cannot be empty")

    levels = logging.getLevelNamesMapping()
    if name not in levels:
        raise ValueError(f"invalid log level: {name!r}")
    return LogLevelDecision(name=name, value=levels[name])


@dataclass(frozen=True)
class LogOutputTarget:
    transport: str
    destination: Path | None = None
    scope: str = "global"


@dataclass(frozen=True)
class CommandObservability:
    visuals: str | None = None
    heartbeat_interval_seconds: float | None = None
    log_level: str | None = None
    log_outputs: tuple[LogOutputTarget, ...] = ()


@dataclass(frozen=True)
class LogOutputSettings:
    outputs: tuple[LogOutputTarget, ...]


@dataclass(frozen=True)
class ObservabilitySettings:
    visuals: str
    heartbeat_interval_seconds: float
    log_decision: LogLevelDecision
    log_output: LogOutputSettings

    def effective_config(self) -> dict[str, object]:
        return {
            "visuals": self.visuals,
            "heartbeat_interval_seconds": self.heartbeat_interval_seconds,
            "log_level": self.log_decision.name,
            "log_outputs": [
                {
                    "transport": output.transport,
                    "scope": output.scope,
                    "destination": (
                        str(output.destination)
                        if output.destination is not None
                        else None
                    ),
                }
                for output in self.log_output.outputs
            ],
        }


def resolve_project_log_outputs(
    outputs: Sequence[LogOutputConfig] | None,
    project_path: Path,
) -> list[LogOutputTarget]:
    if not outputs:
        return []

    resolved: list[LogOutputTarget] = []
    for output in outputs:
        transport = output.transport.lower()
        scope = output.scope.lower()
        destination: Path | None = None
        if transport == "fs" and output.path is not None:
            destination = (
                (project_path.parent / output.path).resolve()
                if scope == "global"
                else Path(output.path)
            )
        resolved.append(
            LogOutputTarget(
                transport=transport,
                destination=destination,
                scope=scope,
            )
        )
    return resolved


def _normalize_log_outputs(
    outputs: Sequence[LogOutputTarget] | None,
    allow_execution_scope: bool,
) -> tuple[LogOutputTarget, ...]:
    if not outputs:
        return ()

    normalized: list[LogOutputTarget] = []
    for target in outputs:
        transport = _normalize_lower(target.transport)
        if transport is None:
            raise ValueError("log transport cannot be empty")
        if transport not in LOG_TRANSPORT_SET:
            choices = ", ".join(LOG_TRANSPORT_CHOICES)
            raise ValueError(
                f"log transport must be one of {choices}, got {transport!r}"
            )

        scope = _normalize_lower(target.scope)
        if scope is None or scope not in LOG_SCOPE_SET:
            choices = ", ".join(LOG_SCOPE_CHOICES)
            raise ValueError(f"log scope must be one of {choices}, got {scope!r}")

        destination = target.destination
        if scope == "execution":
            if not allow_execution_scope:
                raise ValueError(
                    "log scope 'execution' is only valid for execution-scoped operations"
                )
            if transport != "fs":
                raise ValueError("log scope 'execution' requires transport='fs'")
            if destination is not None and destination.is_absolute():
                raise ValueError("execution-scoped log path must be relative")
            if destination is not None and ".." in destination.parts:
                raise ValueError(
                    "execution-scoped log path must stay inside the execution directory"
                )
            normalized.append(
                LogOutputTarget(
                    transport="fs",
                    destination=destination,
                    scope="execution",
                )
            )
            continue

        if transport == "fs":
            if destination is None:
                raise ValueError("log transport 'fs' requires a log path")
            normalized.append(
                LogOutputTarget(
                    transport="fs",
                    destination=destination,
                    scope="global",
                )
            )
            continue

        normalized.append(LogOutputTarget(transport=transport))
    return tuple(normalized)


def resolve_log_output(
    cli_outputs: Sequence[LogOutputTarget] | None = None,
    config_outputs: Sequence[LogOutputTarget] | None = None,
    default_transport: str = "stderr",
    allow_execution_scope: bool = False,
) -> LogOutputSettings:
    resolved_cli_outputs = _normalize_log_outputs(
        cli_outputs,
        allow_execution_scope=allow_execution_scope,
    )
    if resolved_cli_outputs:
        return LogOutputSettings(outputs=resolved_cli_outputs)

    resolved_config_outputs = _normalize_log_outputs(
        config_outputs,
        allow_execution_scope=allow_execution_scope,
    )
    if resolved_config_outputs:
        return LogOutputSettings(outputs=resolved_config_outputs)

    transport = _normalize_lower(default_transport)
    if transport is None or transport not in LOG_TRANSPORT_SET:
        choices = ", ".join(LOG_TRANSPORT_CHOICES)
        raise ValueError(
            f"default log transport must be one of {choices}, got {transport!r}"
        )
    return LogOutputSettings(outputs=(LogOutputTarget(transport=transport),))


def resolve_observability_settings(
    project_path: Path | None,
    observability: ObservabilityConfig | None,
    command_observability: CommandObservability,
) -> ObservabilitySettings:
    logging_config = observability.logging if observability is not None else None
    configured_output_specs = (
        logging_config.outputs if logging_config is not None else None
    )
    if configured_output_specs and project_path is None:
        raise ValueError("project_path is required for configured log outputs")
    configured_outputs = (
        resolve_project_log_outputs(configured_output_specs, project_path)
        if project_path is not None
        else []
    )
    configured_heartbeat_interval = (
        observability.heartbeat_interval_seconds if observability is not None else None
    )
    heartbeat_interval = (
        command_observability.heartbeat_interval_seconds
        if command_observability.heartbeat_interval_seconds is not None
        else configured_heartbeat_interval
    )

    return ObservabilitySettings(
        visuals=resolve_visuals(
            command_observability.visuals,
            observability.visuals if observability is not None else None,
        ),
        heartbeat_interval_seconds=resolve_heartbeat_interval_seconds(
            heartbeat_interval,
        ),
        log_decision=resolve_log_level(
            command_observability.log_level,
            logging_config.level if logging_config is not None else None,
        ),
        log_output=resolve_log_output(
            cli_outputs=command_observability.log_outputs,
            config_outputs=configured_outputs,
            allow_execution_scope=True,
        ),
    )


def resolve_execution_log_outputs(
    settings: LogOutputSettings,
    execution_dir: Path,
    *,
    default_path: Path,
) -> LogOutputSettings:
    outputs: list[LogOutputTarget] = []
    for target in settings.outputs:
        if target.scope != "execution":
            outputs.append(target)
            continue
        relative_path = target.destination or default_path
        if relative_path.is_absolute():
            raise ValueError("execution-scoped log path must be relative")
        if ".." in relative_path.parts:
            raise ValueError(
                "execution-scoped log path must stay inside the execution directory"
            )
        outputs.append(
            LogOutputTarget(
                transport="fs",
                destination=(execution_dir / relative_path).resolve(),
                scope="execution",
            )
        )
    return LogOutputSettings(outputs=tuple(outputs))
