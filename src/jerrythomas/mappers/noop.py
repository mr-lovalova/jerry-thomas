from typing import Iterator, Any


def identity(stream: Iterator[Any]) -> Iterator[Any]:
    yield from stream
