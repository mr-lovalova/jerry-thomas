import gzip
import stat

import pytest

from datapipeline.io.sinks.files import (
    AtomicBinaryFileSink,
    AtomicTextFileSink,
    GzipBinarySink,
)


def test_atomic_text_sink_commits_and_removes_temporary_file(tmp_path) -> None:
    destination = tmp_path / "output.txt"
    sink = AtomicTextFileSink(destination)

    sink.write_text("hello")
    assert not destination.exists()

    sink.close()

    assert destination.read_text(encoding="utf-8") == "hello"
    assert list(tmp_path.iterdir()) == [destination]


def test_atomic_text_sink_preserves_existing_permissions(tmp_path) -> None:
    destination = tmp_path / "output.txt"
    destination.write_text("old", encoding="utf-8")
    destination.chmod(0o640)
    sink = AtomicTextFileSink(destination)

    sink.write_text("new")
    sink.close()

    assert destination.read_text(encoding="utf-8") == "new"
    assert stat.S_IMODE(destination.stat().st_mode) == 0o640


def test_atomic_text_sink_abort_preserves_existing_file(tmp_path) -> None:
    destination = tmp_path / "output.txt"
    destination.write_text("old", encoding="utf-8")
    sink = AtomicTextFileSink(destination)

    sink.write_text("new")
    sink.abort()

    assert destination.read_text(encoding="utf-8") == "old"


def test_atomic_gzip_text_sink_commits_deterministic_output(tmp_path) -> None:
    first = tmp_path / "first.jsonl.gz"
    second = tmp_path / "second.jsonl.gz"
    for destination in (first, second):
        sink = AtomicTextFileSink(destination, compression="gzip")
        sink.write_text('{"value":1}\n')
        sink.close()

    with gzip.open(first, "rt", encoding="utf-8") as stream:
        assert stream.read() == '{"value":1}\n'
    assert first.read_bytes() == second.read_bytes()


def test_atomic_gzip_text_sink_abort_preserves_existing_file(tmp_path) -> None:
    destination = tmp_path / "output.jsonl.gz"
    destination.write_bytes(b"existing")
    sink = AtomicTextFileSink(destination, compression="gzip")

    sink.write_text("replacement\n")
    sink.abort()

    assert destination.read_bytes() == b"existing"
    assert list(tmp_path.iterdir()) == [destination]


def test_atomic_text_sink_rejects_invalid_encoding_before_creating_temp_file(
    tmp_path,
) -> None:
    with pytest.raises(LookupError):
        AtomicTextFileSink(tmp_path / "output.txt", encoding="not-a-codec")

    assert list(tmp_path.iterdir()) == []


def test_atomic_binary_sink_abort_removes_temporary_file(tmp_path) -> None:
    destination = tmp_path / "output.bin"
    sink = AtomicBinaryFileSink(destination)

    sink.write_bytes(b"discarded")
    sink.abort()

    assert list(tmp_path.iterdir()) == []


def test_gzip_sink_commits_readable_file(tmp_path) -> None:
    destination = tmp_path / "output.jsonl.gz"
    sink = GzipBinarySink(destination, compression_level=3)

    sink.write_bytes(b'{"value": 1}\n')
    sink.close()

    with gzip.open(destination, "rb") as stream:
        assert stream.read() == b'{"value": 1}\n'
    assert list(tmp_path.iterdir()) == [destination]
