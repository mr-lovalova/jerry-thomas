import csv
import os

import pytest

from jerrythomas.io.writers.csv_writer import CsvFileWriter


def test_csv_writer_writes_flattened_rows(tmp_path) -> None:
    dest = tmp_path / "out.csv"
    writer = CsvFileWriter(dest)
    writer.write({"key": "k1", "feature": {"temp": 1.0, "wind": [2.0]}})
    writer.write({"key": "k2", "feature": {"temp": 3.0, "wind": [4.0]}})
    writer.close()

    with open(dest, newline="", encoding="utf-8") as fh:
        parsed = list(csv.DictReader(fh))

    assert parsed[0]["key"] == "k1"
    assert parsed[0]["feature.temp"] == "1.0"
    assert parsed[1]["key"] == "k2"


def test_csv_writer_honors_configured_encoding(tmp_path) -> None:
    dest = tmp_path / "out.csv"
    writer = CsvFileWriter(dest, encoding="utf-8-sig")
    writer.write({"key": "k1", "feature": {"temp": 1.0}})
    writer.close()

    raw = dest.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")


def test_csv_writer_disables_text_newline_translation(tmp_path, monkeypatch) -> None:
    dest = tmp_path / "out.csv"
    newlines = []
    fdopen = os.fdopen

    def recording_fdopen(fd, mode, *, encoding=None, newline=None):
        newlines.append(newline)
        return fdopen(fd, mode, encoding=encoding, newline=newline)

    monkeypatch.setattr(os, "fdopen", recording_fdopen)

    writer = CsvFileWriter(dest)
    writer.write({"key": "k1"})
    writer.close()

    assert newlines == [""]


def test_csv_writer_rejects_new_columns_after_header(tmp_path) -> None:
    destination = tmp_path / "out.csv"
    writer = CsvFileWriter(destination)
    writer.write({"key": "k1", "feature": {"temp": 1.0}})

    with pytest.raises(
        ValueError, match="CSV row contains fields not present in header"
    ):
        writer.write(
            {
                "key": "k2",
                "feature": {"temp": 2.0, "new": 9.0},
            }
        )

    writer.abort()
    assert not destination.exists()


def test_csv_writer_locks_header_and_fills_missing_fields(tmp_path) -> None:
    destination = tmp_path / "out.csv"
    writer = CsvFileWriter(destination)
    writer.write({"second": 2, "first": 1})
    writer.write({"first": 3})
    writer.close()

    with destination.open(newline="", encoding="utf-8") as stream:
        assert list(csv.reader(stream)) == [
            ["first", "second"],
            ["1", "2"],
            ["3", ""],
        ]
