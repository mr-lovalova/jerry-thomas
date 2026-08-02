import csv

import pytest

from jerrythomas.sources.decoders import CsvDecoder, JsonDecoder, JsonLinesDecoder


def test_json_lines_preserve_records_across_byte_and_character_boundaries() -> None:
    chunks = [
        b'{"name":"And',
        b"ers \xc3",
        b'\xa5"}\r\n\n{"name":"Maja"}\n',
        b'{"name":"Nils"}',
    ]
    decoder = JsonLinesDecoder()

    assert list(decoder.decode(chunks)) == [
        {"name": "Anders \u00e5"},
        {"name": "Maja"},
        {"name": "Nils"},
    ]


def test_json_top_level_null_is_one_value() -> None:
    decoder = JsonDecoder()

    assert list(decoder.decode([b"null"])) == [None]


def test_json_null_array_field_is_empty() -> None:
    decoder = JsonDecoder(array_field="items")

    assert list(decoder.decode([b'{"items": null}'])) == []


@pytest.mark.parametrize("decoder_type", [JsonDecoder, JsonLinesDecoder])
@pytest.mark.parametrize(
    "payload",
    [
        b'{"name": "first", "name": "second"}',
        b'{"nested": {"name": "first", "name": "second"}}',
    ],
)
def test_json_rejects_duplicate_object_keys(decoder_type, payload: bytes) -> None:
    with pytest.raises(ValueError, match="duplicate key 'name'"):
        list(decoder_type().decode([payload]))


@pytest.mark.parametrize("decoder_type", [JsonDecoder, JsonLinesDecoder])
@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_json_rejects_non_standard_numeric_constants(
    decoder_type,
    constant: str,
) -> None:
    with pytest.raises(ValueError, match=f"non-standard constant {constant}"):
        list(decoder_type().decode([f'{{"value": {constant}}}'.encode()]))


@pytest.mark.parametrize("decoder_type", [JsonDecoder, JsonLinesDecoder])
def test_json_rejects_float_overflow(decoder_type) -> None:
    with pytest.raises(ValueError, match="outside the finite float range: 1e999"):
        list(decoder_type().decode([b'{"value": 1e999}']))


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_csv_preserves_newlines_inside_quoted_fields(newline: str) -> None:
    text = f'name;note{newline}Anders \u00e5;"first line{newline}second line"{newline}'
    payload = text.encode("utf-8")
    chunks = [payload[index : index + 1] for index in range(len(payload))]
    decoder = CsvDecoder()

    assert list(decoder.decode(chunks)) == [
        {"name": "Anders \u00e5", "note": f"first line{newline}second line"}
    ]


def test_csv_handles_rows_split_between_chunks_without_a_final_newline() -> None:
    chunks = [b"name;value\r\nAnd", b"ers;1\r\nMaja;2"]
    decoder = CsvDecoder()

    assert list(decoder.decode(chunks)) == [
        {"name": "Anders", "value": "1"},
        {"name": "Maja", "value": "2"},
    ]


@pytest.mark.parametrize("payload", [b"", b"name;value\n"])
def test_csv_without_data_records_is_empty(payload: bytes) -> None:
    assert list(CsvDecoder().decode([payload])) == []


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"\n", "empty field name"),
        (b";value\nAnders;1\n", "empty field name"),
        (b"name;name\nAnders;1\n", "duplicate field names"),
    ],
)
def test_csv_rejects_ambiguous_headers(payload: bytes, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        list(CsvDecoder().decode([payload]))


@pytest.mark.parametrize(
    ("payload", "actual"),
    [
        (b"name;value\nAnders;1;extra\n", 3),
        (b"name;value\nAnders\n", 1),
    ],
)
def test_csv_rejects_rows_with_the_wrong_number_of_fields(
    payload: bytes,
    actual: int,
) -> None:
    with pytest.raises(
        ValueError,
        match=rf"CSV row 2 has {actual} fields; expected 2\.",
    ):
        list(CsvDecoder().decode([payload]))


def test_csv_rejects_unterminated_quoted_fields() -> None:
    with pytest.raises(csv.Error, match="unexpected end of data"):
        list(CsvDecoder().decode([b'name;value\nAnders;"unfinished']))


def test_csv_ignores_blank_lines_between_records() -> None:
    payload = b"name;value\nAnders;1\n\nMaja;2\n"

    assert list(CsvDecoder().decode([payload])) == [
        {"name": "Anders", "value": "1"},
        {"name": "Maja", "value": "2"},
    ]


def test_csv_preserves_empty_field_values() -> None:
    assert list(CsvDecoder().decode([b"name;value\nAnders;\n"])) == [
        {"name": "Anders", "value": ""}
    ]


def test_csv_error_prefix_is_checked_before_parsing_rows() -> None:
    decoder = CsvDecoder(error_prefixes=("service unavailable",))
    chunks = [b"Service un", b"available\ntry later"]

    with pytest.raises(
        ValueError,
        match="csv response looks like error text: Service unavailable",
    ):
        list(decoder.decode(chunks))


def test_csv_yields_complete_rows_without_reading_a_later_chunk() -> None:
    def chunks():
        yield b"name;value\n"
        yield b"Anders;1\n"
        raise AssertionError("decoder read beyond the requested row")

    rows = CsvDecoder().decode(chunks())

    assert next(rows) == {"name": "Anders", "value": "1"}
    rows.close()
