import codecs
import csv
import itertools
import json
import math
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from typing import Any


class Decoder(ABC):
    @abstractmethod
    def decode(self, chunks: Iterable[bytes]) -> Iterator[Any]: ...


def _iter_text_lines(chunks: Iterable[bytes], encoding: str) -> Iterator[str]:
    """Yield LF/CRLF physical lines without rewriting their terminators."""

    decoder = codecs.getincrementaldecoder(encoding)()
    fragments: list[str] = []
    for chunk in chunks:
        text = decoder.decode(chunk)
        start = 0
        while (newline := text.find("\n", start)) >= 0:
            line = text[start : newline + 1]
            if fragments:
                fragments.append(line)
                yield "".join(fragments)
                fragments.clear()
            else:
                yield line
            start = newline + 1
        if start < len(text):
            fragments.append(text[start:])

    tail = decoder.decode(b"", final=True)
    if tail:
        fragments.append(tail)
    if fragments:
        yield "".join(fragments)


def _read_all_text(chunks: Iterable[bytes], encoding: str) -> str:
    decoder = codecs.getincrementaldecoder(encoding)()
    parts: list[str] = []
    for chunk in chunks:
        parts.append(decoder.decode(chunk))
    parts.append(decoder.decode(b"", final=True))
    return "".join(parts)


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"JSON object contains duplicate key {key!r}.")
        value[key] = item
    return value


def _reject_non_standard_json_constant(value: str) -> None:
    raise ValueError(f"JSON contains non-standard constant {value}.")


def _parse_finite_json_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"JSON number is outside the finite float range: {value}.")
    return number


def _json_decoder() -> json.JSONDecoder:
    return json.JSONDecoder(
        object_pairs_hook=_reject_duplicate_json_keys,
        parse_constant=_reject_non_standard_json_constant,
        parse_float=_parse_finite_json_float,
    )


class CsvDecoder(Decoder):
    def __init__(
        self,
        *,
        delimiter: str = ";",
        encoding: str = "utf-8",
        error_prefixes: Sequence[str] | None = None,
    ) -> None:
        self.delimiter = delimiter
        self.encoding = encoding
        self._error_prefixes = [p.lower() for p in (error_prefixes or [])]

    def _iter_lines(self, chunks: Iterable[bytes]) -> Iterator[str]:
        lines = _iter_text_lines(chunks, self.encoding)
        try:
            first = next(lines)
        except StopIteration:
            return iter(())
        if self._error_prefixes:
            lowered = first.lstrip().lower()
            if any(lowered.startswith(p) for p in self._error_prefixes):
                raise ValueError(
                    f"csv response looks like error text: {first[:120].rstrip()}"
                )
        return itertools.chain((first,), lines)

    def decode(self, chunks: Iterable[bytes]) -> Iterator[dict]:
        reader = csv.reader(
            self._iter_lines(chunks),
            delimiter=self.delimiter,
            strict=True,
        )
        try:
            fieldnames = next(reader)
        except StopIteration:
            return

        if not fieldnames or any(not fieldname.strip() for fieldname in fieldnames):
            raise ValueError("CSV header contains an empty field name.")
        if len(fieldnames) != len(set(fieldnames)):
            raise ValueError("CSV header contains duplicate field names.")

        for row in reader:
            if not row:
                continue
            if len(row) != len(fieldnames):
                raise ValueError(
                    f"CSV row {reader.line_num} has {len(row)} fields; "
                    f"expected {len(fieldnames)}."
                )
            yield dict(zip(fieldnames, row))


class JsonDecoder(Decoder):
    def __init__(
        self,
        encoding: str = "utf-8",
        array_field: str | None = None,
    ) -> None:
        self.encoding = encoding
        self.array_field = array_field

    def _load_payload(self, chunks: Iterable[bytes]) -> Any:
        text = _read_all_text(chunks, self.encoding)
        data = _json_decoder().decode(text)
        if self.array_field is not None:
            if not isinstance(data, dict):
                raise ValueError("json array_field requires a top-level object")
            if self.array_field not in data:
                raise ValueError(f"json array_field missing: {self.array_field}")
            data = data[self.array_field]
        return data

    def decode(self, chunks: Iterable[bytes]) -> Iterator[Any]:
        data = self._load_payload(chunks)
        if data is None and self.array_field is not None:
            return
        if isinstance(data, list):
            yield from data
            return
        yield data


class JsonLinesDecoder(Decoder):
    def __init__(self, encoding: str = "utf-8") -> None:
        self.encoding = encoding

    def decode(self, chunks: Iterable[bytes]) -> Iterator[dict]:
        decoder = _json_decoder()
        for line in _iter_text_lines(chunks, self.encoding):
            s = line.strip()
            if not s:
                continue
            yield decoder.decode(s)
