import pytest
from pydantic import ValidationError

from jerrythomas.config.execution import ExecutionConfig


@pytest.mark.parametrize("value", [0, True, 1.5, "100"])
def test_sort_buffer_mb_requires_a_positive_integer(value) -> None:
    with pytest.raises(ValidationError):
        ExecutionConfig(sort_buffer_mb=value)


def test_sort_buffer_mb_converts_to_bytes() -> None:
    assert ExecutionConfig(sort_buffer_mb=2).sort_buffer_bytes == 2 * 1024 * 1024


def test_execution_config_rejects_unknown_settings() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ExecutionConfig.model_validate({"workers": 4})
