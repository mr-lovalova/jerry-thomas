import pytest
from pydantic import ValidationError

from datapipeline.config.dataset.postprocess import CoverageConfig, PostprocessConfig


def test_postprocess_parses_sample_filters() -> None:
    config = PostprocessConfig.model_validate(
        {"samples": {"features": {"threshold": 0.8, "ids": ["price"]}}}
    )

    assert config.samples.features == CoverageConfig(
        threshold=0.8,
        ids=["price"],
    )


def test_postprocess_rejects_data_dependent_column_selection() -> None:
    with pytest.raises(ValidationError):
        PostprocessConfig.model_validate(
            {
                "columns": {
                    "features": {"threshold": 0.8},
                }
            }
        )


@pytest.mark.parametrize("ids", [[], ["price", "price"]])
def test_postprocess_rejects_invalid_filter_ids(ids: list[str]) -> None:
    with pytest.raises(ValidationError):
        PostprocessConfig.model_validate(
            {"samples": {"features": {"threshold": 0.8, "ids": ids}}}
        )
