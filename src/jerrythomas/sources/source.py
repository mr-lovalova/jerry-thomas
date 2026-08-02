from collections.abc import Iterator
from typing import TypeVar

from jerrythomas.domain.stream import RecordStream
from jerrythomas.sources.loader import BaseDataLoader
from jerrythomas.sources.parser import DataParser, ParsingError


TRecord = TypeVar("TRecord")


class Source(RecordStream[TRecord]):
    """Load and parse source rows lazily."""

    def __init__(self, loader: BaseDataLoader, parser: DataParser[TRecord]) -> None:
        self.loader = loader
        self.parser = parser

    def stream(self) -> Iterator[TRecord]:
        for index, row in enumerate(self.loader.load()):
            try:
                parsed = self.parser.parse(row)
                if parsed is not None:
                    yield parsed
            except Exception as exc:
                raise ParsingError(
                    row=row,
                    index=index,
                    original_exc=exc,
                ) from exc
