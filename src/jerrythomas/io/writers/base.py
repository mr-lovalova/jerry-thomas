from collections.abc import Callable

from jerrythomas.io.sinks.files import AtomicTextFileSink
from jerrythomas.io.sinks.stdout import StdoutTextSink


class LineWriter:
    """Text line writer (uses a text sink + serializer)."""

    def __init__(
        self,
        sink: StdoutTextSink | AtomicTextFileSink,
        serializer: Callable[[object], str],
    ) -> None:
        self.sink = sink
        self.serializer = serializer

    def write(self, item: object) -> None:
        self.sink.write_text(self.serializer(item))

    def close(self) -> None:
        self.sink.close()

    def abort(self) -> None:
        self.sink.abort()
