from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any, Literal, Protocol

from jerrythomas.sources.decoders import (
    CsvDecoder,
    Decoder,
    JsonDecoder,
    JsonLinesDecoder,
)
from jerrythomas.sources.ports import SourceTransport


SourceProgressUnit = Literal["items", "records", "rows", "ticks"]


class RowGenerator(Protocol):
    """Generate raw rows for a source parser."""

    def generate(self) -> Iterator[Any]: ...


class BaseDataLoader(ABC):
    @abstractmethod
    def load(self) -> Iterator[Any]: ...

    @property
    def current_resource_uri(self) -> str | None:
        return None

    @property
    def progress_unit(self) -> SourceProgressUnit:
        return "records"


class DataLoader(BaseDataLoader):
    """Decode rows from byte resources supplied by a transport."""

    def __init__(self, transport: SourceTransport, decoder: Decoder) -> None:
        self.transport = transport
        self.decoder = decoder
        self._current_resource_uri: str | None = None

    @property
    def current_resource_uri(self) -> str | None:
        return self._current_resource_uri

    @property
    def progress_unit(self) -> SourceProgressUnit:
        if isinstance(self.decoder, CsvDecoder):
            return "rows"
        if isinstance(self.decoder, (JsonDecoder, JsonLinesDecoder)):
            return "items"
        return "records"

    def load(self) -> Iterator[Any]:
        resources = iter(self.transport.resources())
        try:
            for resource in resources:
                self._current_resource_uri = resource.uri
                stream = iter(resource.stream)
                try:
                    yield from self.decoder.decode(stream)
                finally:
                    close = getattr(stream, "close", None)
                    if callable(close):
                        close()
        finally:
            close = getattr(resources, "close", None)
            if callable(close):
                close()
            self._current_resource_uri = None


class GeneratorLoader(BaseDataLoader):
    """Expose an in-process row generator through the loader contract."""

    def __init__(
        self,
        generator: RowGenerator,
        progress_unit: SourceProgressUnit = "records",
    ) -> None:
        self.generator = generator
        self._progress_unit = progress_unit

    def load(self) -> Iterator[Any]:
        yield from self.generator.generate()

    @property
    def progress_unit(self) -> SourceProgressUnit:
        return self._progress_unit
