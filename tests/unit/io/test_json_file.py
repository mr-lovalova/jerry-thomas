from pathlib import Path

import pytest

import datapipeline.io.json_file as json_file
from datapipeline.io.json_file import read_json_object, write_json_object


def test_read_json_object(tmp_path: Path) -> None:
    source = tmp_path / "object.json"
    source.write_text('{"value": 1}\n', encoding="utf-8")

    assert read_json_object(source) == {"value": 1}


def test_read_json_object_rejects_other_json_values(tmp_path: Path) -> None:
    source = tmp_path / "list.json"
    source.write_text("[1, 2]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected JSON object"):
        read_json_object(source)


def test_json_object_serialization_failure_preserves_previous_file(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "artifact.json"
    destination.write_text('{"previous": true}\n', encoding="utf-8")

    with pytest.raises(TypeError):
        write_json_object(destination, {"value": object()})

    assert destination.read_text(encoding="utf-8") == '{"previous": true}\n'
    assert list(tmp_path.iterdir()) == [destination]


def test_json_object_interrupt_preserves_previous_file(
    monkeypatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "artifact.json"
    destination.write_text('{"previous": true}\n', encoding="utf-8")

    def interrupt_dump(payload, handle, indent, sort_keys, allow_nan):
        assert allow_nan is False
        handle.write('{"partial":')
        raise KeyboardInterrupt

    monkeypatch.setattr(json_file.json, "dump", interrupt_dump)

    with pytest.raises(KeyboardInterrupt):
        write_json_object(destination, {"value": 1})

    assert destination.read_text(encoding="utf-8") == '{"previous": true}\n'
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_json_object_rejects_non_finite_numbers(
    tmp_path: Path,
    value: float,
) -> None:
    destination = tmp_path / "artifact.json"

    with pytest.raises(ValueError, match="Out of range float values"):
        write_json_object(destination, {"value": value})

    assert not destination.exists()
