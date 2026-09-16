import pytest
from pydantic import ValidationError

from jerrythomas.config.dataset.dataset import SampleConfig


@pytest.mark.parametrize("rounding", ["floor", "ceil", "exact"])
def test_sample_rounding_is_explicit_and_serialized(rounding: str) -> None:
    sample = SampleConfig.model_validate({"cadence": "1h", "rounding": rounding})
    assert sample.model_dump()["rounding"] == rounding


def test_sample_rounding_has_no_default() -> None:
    with pytest.raises(ValidationError) as error:
        SampleConfig.model_validate({"cadence": "1h"})
    assert [(item["loc"], item["type"]) for item in error.value.errors()] == [
        (("rounding",), "missing")
    ]


@pytest.mark.parametrize("rounding", [None, True, "drop", "nearest", "FLOOR", 0])
def test_sample_rounding_rejects_invalid_choices(rounding: object) -> None:
    with pytest.raises(ValidationError, match="rounding"):
        SampleConfig.model_validate({"cadence": "1h", "rounding": rounding})
