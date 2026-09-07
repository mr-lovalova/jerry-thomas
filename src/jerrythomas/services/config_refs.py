import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from jerrythomas.config.interpolation import (
    MissingInterpolation,
    is_missing_interpolation,
)


_CONFIG_REF_RE = re.compile(r"\$\{([A-Za-z_][\w-]*):([^}]+)\}")
_INTERPOLATION_RE = re.compile(r"\$\{([^}]+)\}")
_DOTENV_ESCAPES = {"n": "\n", "r": "\r", "t": "\t", '"': '"', "\\": "\\"}
_DOTENV_ESCAPE_RE = re.compile(r"\\(.)")
_PROJECT_VAR_MISSING = object()


class ConfigRefError(ValueError):
    """Raised when an external config reference cannot be resolved."""


def serialize_project_value(value: Any) -> Any:
    """Normalize project global values for interpolation."""
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ConfigRefError("Project datetime globals must be timezone-aware.")
        return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if value is None:
        return None
    return value


def project_vars_from_data(data: Mapping[str, Any]) -> dict[str, Any]:
    vars_: dict[str, Any] = {}
    name = data.get("name")
    if name:
        vars_["project_name"] = str(name)

    variant = data.get("variant")
    if variant:
        vars_["project_variant"] = str(variant)

    globals_ = data.get("globals") or {}
    if isinstance(globals_, Mapping):
        for key in ("project_name", "project_variant"):
            if key in globals_:
                raise ConfigRefError(
                    f"Project globals must not redefine reserved variable '{key}'."
                )
        vars_.update(_resolve_project_globals(globals_, vars_))
    return vars_


def _resolve_project_globals(
    globals_: Mapping[str, Any],
    base_vars: Mapping[str, Any],
) -> dict[str, Any]:
    raw = {str(key): serialize_project_value(value) for key, value in globals_.items()}
    resolved: dict[str, Any] = {}
    resolving: list[str] = []

    def value_for(key: str) -> Any:
        if key in raw:
            return resolve_key(key)
        if key in base_vars:
            return base_vars[key]
        return _PROJECT_VAR_MISSING

    def interpolate_global(text: str) -> Any:
        match = _INTERPOLATION_RE.fullmatch(text)
        if match:
            key = match.group(1)
            value = value_for(key)
            if value is _PROJECT_VAR_MISSING:
                raise ConfigRefError(
                    f"Unknown interpolation variable '{key}' in project globals."
                )
            if value is None or is_missing_interpolation(value):
                return MissingInterpolation(key)
            return value

        def repl(match: re.Match[str]) -> str:
            key = match.group(1)
            value = value_for(key)
            if value is _PROJECT_VAR_MISSING:
                raise ConfigRefError(
                    f"Unknown interpolation variable '{key}' in project globals."
                )
            if value is None or is_missing_interpolation(value):
                raise ConfigRefError(
                    f"Interpolation variable '{key}' has no value and cannot be embedded "
                    "in project globals."
                )
            return str(value)

        return _INTERPOLATION_RE.sub(repl, text)

    def resolve_key(key: str) -> Any:
        if key in resolved:
            return resolved[key]
        if key in resolving:
            cycle = " -> ".join([*resolving, key])
            raise ConfigRefError(f"Cyclic project global reference: {cycle}")
        resolving.append(key)
        value = raw[key]
        if isinstance(value, str):
            value = interpolate_global(value)
        resolved[key] = value
        resolving.pop()
        return value

    for key in raw:
        resolve_key(key)
    return resolved


def interpolate_config_vars(obj: Any, vars_: Mapping[str, Any]) -> Any:
    """Recursively substitute ${var} in strings using vars_ map."""
    if isinstance(obj, dict):
        return {
            key: interpolate_config_vars(value, vars_) for key, value in obj.items()
        }
    if isinstance(obj, list):
        return [interpolate_config_vars(value, vars_) for value in obj]
    if isinstance(obj, str):
        match = _INTERPOLATION_RE.fullmatch(obj)
        if match:
            key = match.group(1)
            if key in vars_:
                value = vars_[key]
                if value is None or is_missing_interpolation(value):
                    return MissingInterpolation(key)
                return value
            raise ConfigRefError(f"Unknown interpolation variable '{key}'.")

        def repl(match: re.Match[str]) -> str:
            key = match.group(1)
            if key not in vars_:
                raise ConfigRefError(f"Unknown interpolation variable '{key}'.")
            value = vars_[key]
            if value is None or is_missing_interpolation(value):
                raise ConfigRefError(
                    f"Interpolation variable '{key}' has no value and cannot be embedded "
                    "in text."
                )
            return str(value)

        return _INTERPOLATION_RE.sub(repl, obj)
    return obj


def resolve_config_refs(
    obj: Any,
    *,
    project_yaml: Path,
    env: Mapping[str, str] | None = None,
) -> Any:
    resolved_project_yaml = project_yaml.resolve()
    environment = merged_project_env(resolved_project_yaml) if env is None else env
    return _resolve_config_refs(obj, resolved_project_yaml, environment)


def merged_project_env(project_yaml: Path) -> dict[str, str]:
    env = dict(_load_dotenv(project_yaml.parent / ".env"))
    env.update(os.environ)
    return env


def _resolve_config_refs(
    obj: Any,
    project_yaml: Path,
    env: Mapping[str, str],
) -> Any:
    if isinstance(obj, dict):
        return {
            key: _resolve_config_refs(value, project_yaml, env)
            for key, value in obj.items()
        }
    if isinstance(obj, list):
        return [_resolve_config_refs(value, project_yaml, env) for value in obj]
    if isinstance(obj, str):
        return _resolve_string_refs(obj, project_yaml, env)
    return obj


def _resolve_string_refs(
    text: str,
    project_yaml: Path,
    env: Mapping[str, str],
) -> str:
    match = _CONFIG_REF_RE.fullmatch(text)
    if match:
        return _resolve_env_ref(match, project_yaml, env)

    def repl(match: re.Match[str]) -> str:
        return _resolve_env_ref(match, project_yaml, env)

    return _CONFIG_REF_RE.sub(repl, text)


def _resolve_env_ref(
    match: re.Match[str],
    project_yaml: Path,
    env: Mapping[str, str],
) -> str:
    scheme = match.group(1).strip().lower()
    if scheme != "env":
        raise ConfigRefError(
            f"Unsupported config reference scheme '{scheme}' in '{match.group(0)}'."
        )
    name = match.group(2).strip()
    if not name:
        raise ConfigRefError(
            "Config reference '${env:...}' must include a variable name."
        )
    value = env.get(name)
    if value is None:
        raise ConfigRefError(
            f"Missing environment variable '{name}' referenced from "
            f"{project_yaml.parent / '.env'} or process environment."
        )
    return value


def _load_dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_dotenv_line(raw_line)
        if parsed is None:
            continue
        key, value = parsed
        values[key] = value
    return values


def _parse_dotenv_line(line: str) -> tuple[str, str] | None:
    text = line.strip()
    if not text or text.startswith("#"):
        return None
    if text.startswith("export "):
        text = text[len("export ") :].lstrip()
    if "=" not in text:
        return None
    key, raw_value = text.split("=", 1)
    key = key.strip()
    if not key:
        return None

    value = raw_value.strip()
    if not value:
        return key, ""

    quote = value[0]
    if quote in {"'", '"'}:
        index = 1
        while index < len(value):
            if quote == '"' and value[index] == "\\":
                index += 2
                continue
            if value[index] == quote and value[index + 1 :].lstrip().startswith("#"):
                value = value[: index + 1]
                break
            index += 1
    if quote in {"'", '"'} and value.endswith(quote):
        inner = value[1:-1]
        if quote == '"':
            inner = _DOTENV_ESCAPE_RE.sub(
                lambda match: _DOTENV_ESCAPES.get(match.group(1), match.group(0)),
                inner,
            )
        return key, inner

    comment_index = value.find(" #")
    if comment_index != -1:
        value = value[:comment_index].rstrip()
    return key, value
