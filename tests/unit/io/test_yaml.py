from pathlib import Path

import pytest

from jerrythomas.io import yaml


@pytest.mark.parametrize(
    "yaml_text",
    [
        "value: first\nvalue: second\n",
        "outer:\n  value: first\n  value: second\n",
    ],
)
def test_load_yaml_rejects_duplicate_mapping_keys(
    tmp_path: Path,
    yaml_text: str,
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(yaml_text, encoding="utf-8")

    with pytest.raises(ValueError, match="found duplicate key 'value'"):
        yaml.load_yaml(path)


def test_load_yaml_allows_explicit_override_of_merged_defaults(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "defaults: &defaults\n  value: default\nconfig:\n"
        "  <<: *defaults\n  value: explicit\n",
        encoding="utf-8",
    )

    assert yaml.load_yaml(path)["config"] == {"value": "explicit"}


def test_load_yaml_rejects_repeated_merge_keys(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "first: &first {a: 1}\nsecond: &second {b: 2}\n"
        "config:\n  <<: *first\n  <<: *second\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="found duplicate key '<<'"):
        yaml.load_yaml(path)


@pytest.mark.parametrize("value", ["ON", "OFF", "on", "off", "yes", "no"])
def test_yaml_legacy_boolean_words_remain_strings(tmp_path: Path, value: str) -> None:
    from pydantic import ValidationError

    from jerrythomas.config.observability import ObservabilityConfig

    path = tmp_path / "config.yaml"
    path.write_text(f"visuals: {value}\n", encoding="utf-8")
    data = yaml.load_yaml(path)
    assert data == {"visuals": value}
    with pytest.raises(ValidationError, match="valid boolean"):
        ObservabilityConfig.model_validate(data)


@pytest.mark.parametrize(("value", "expected"), [("true", True), ("false", False)])
def test_yaml_boolean_resolution_does_not_modify_pyyaml(
    tmp_path: Path, value: str, expected: bool
) -> None:
    import yaml as pyyaml

    path = tmp_path / "config.yaml"
    path.write_text(f"visuals: {value}\n", encoding="utf-8")
    assert yaml.load_yaml(path)["visuals"] is expected
    assert pyyaml.safe_load("value: ON")["value"] is True
