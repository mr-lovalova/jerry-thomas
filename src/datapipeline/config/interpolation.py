from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class MissingInterpolation:
    """A whole-value interpolation that resolved to null."""

    key: str

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f"<MissingInterpolation key={self.key!r}>"


def is_missing_interpolation(value: object) -> bool:
    return isinstance(value, MissingInterpolation)


def coalesce_missing_interpolation(
    value: Any,
    default: Any = None,
) -> Any:
    return default if is_missing_interpolation(value) else value


def normalize_interpolated_args(
    args: dict[str, Any] | None,
    default: Any = None,
) -> dict[str, Any]:
    if not args:
        return {}
    return {
        key: coalesce_missing_interpolation(value, default)
        for key, value in args.items()
    }
