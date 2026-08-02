import sys
from typing import TextIO


class StdoutTextSink:
    def __init__(self, stream: TextIO | None = None):
        self.stream = sys.stdout if stream is None else stream

    def write_text(self, s: str) -> None:
        self.stream.write(s)

    def flush(self) -> None:
        self.stream.flush()

    def close(self) -> None:
        self.flush()

    def abort(self) -> None:
        self.flush()
