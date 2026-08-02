from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class SourceResource:
    uri: str | None
    stream: Iterable[bytes]


class SourceTransport(ABC):
    """Input transport that yields raw-byte resources."""

    @abstractmethod
    def resources(self) -> Iterator[SourceResource]: ...
