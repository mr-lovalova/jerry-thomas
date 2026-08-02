from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar


TRecord = TypeVar("TRecord")


class DataParser(ABC, Generic[TRecord]):
    @abstractmethod
    def parse(self, raw: Any) -> TRecord | None: ...


class ParsingError(Exception):
    """Raised when a single source row fails to parse."""

    def __init__(
        self,
        row: Any,
        index: int | None = None,
        original_exc: BaseException | None = None,
    ) -> None:
        self.row = row
        self.index = index
        self.original_exc = original_exc

        prefix = (
            f"Failed to parse row at index {index}: "
            if index is not None
            else "Failed to parse row: "
        )
        message = prefix + repr(row)
        if original_exc is not None:
            message += f" (caused by {type(original_exc).__name__}: {original_exc})"
        super().__init__(message)
