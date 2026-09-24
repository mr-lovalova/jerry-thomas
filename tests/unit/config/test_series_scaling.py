import pytest
from pydantic import ValidationError

from jerrythomas.config.dataset.dataset import DatasetConfig
from jerrythomas.config.dataset.series import (
    ScalingConfig,
    SeriesConfig,
    TargetSeriesConfig,
)


_BASE_SERIES = {"id": "price", "stream": "prices", "field": "close"}


@pytest.mark.parametrize(
    "scale", [True, {}, {"with_mean": True, "with_std": True, "epsilon": 1e-12}]
)
def test_default_scaling_forms_normalize_identically(scale):
    config = SeriesConfig.model_validate({**_BASE_SERIES, "scale": scale})

    assert config.scale == ScalingConfig()
    assert config.model_dump(mode="json") == SeriesConfig(
        **_BASE_SERIES, scale=ScalingConfig()
    ).model_dump(mode="json")


@pytest.mark.parametrize("values", [{}, {"scale": False}, {"scale": None}])
def test_disabled_scaling_forms_normalize_identically(values):
    config = SeriesConfig.model_validate({**_BASE_SERIES, **values})

    assert config.scale is None
    assert config.model_dump(mode="json") == SeriesConfig(**_BASE_SERIES).model_dump(
        mode="json"
    )


@pytest.mark.parametrize("model", [SeriesConfig, TargetSeriesConfig])
def test_feature_and_target_scaling_allow_partial_overrides(model):
    values = {**_BASE_SERIES, "scale": {"with_mean": False}}
    if model is TargetSeriesConfig:
        values["horizon"] = "0h"
    config = model.model_validate(values)

    assert config.scale == ScalingConfig(with_mean=False, with_std=True, epsilon=1e-12)


@pytest.mark.parametrize(
    "scale",
    [
        0,
        1,
        "true",
        "false",
        [],
        {"unknown": True},
        {"with_mean": "true"},
        {"with_std": 1},
        {"epsilon": "1e-12"},
        {"epsilon": True},
        {"epsilon": 0},
        {"epsilon": -1},
        {"epsilon": float("inf")},
        {"epsilon": float("nan")},
    ],
)
def test_scaling_rejects_invalid_values(scale):
    with pytest.raises(ValidationError):
        SeriesConfig.model_validate({**_BASE_SERIES, "scale": scale})


def test_dataset_dump_roundtrips_mixed_feature_and_target_scaling():
    dataset = DatasetConfig.model_validate(
        {
            "version": "v1",
            "sample": {"cadence": "1d", "rounding": "exact"},
            "features": [
                _BASE_SERIES,
                {**_BASE_SERIES, "id": "centered", "scale": {"with_std": False}},
                {**_BASE_SERIES, "id": "standard", "scale": True},
            ],
            "targets": [
                {
                    "id": "target",
                    "stream": "returns",
                    "field": "value",
                    "horizon": "0h",
                    "scale": {"epsilon": 0.01},
                },
            ],
        }
    )

    for mode in ("python", "json"):
        payload = dataset.model_dump(mode=mode)
        assert "scaling" not in payload
        assert payload["features"][0]["scale"] is None
        assert DatasetConfig.model_validate(payload) == dataset
    assert DatasetConfig.model_validate_json(dataset.model_dump_json()) == dataset


def test_dataset_level_scaling_is_rejected():
    with pytest.raises(ValidationError, match="scaling"):
        DatasetConfig.model_validate(
            {
                "sample": {"cadence": "1d", "rounding": "exact"},
                "features": [_BASE_SERIES],
                "scaling": {"with_mean": False},
            }
        )
