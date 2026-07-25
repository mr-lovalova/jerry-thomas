import pytest
from pydantic import ValidationError

from datapipeline.config.dataset.dataset import DatasetConfig


def test_dataset_requires_sample_config() -> None:
    with pytest.raises(ValidationError, match="sample"):
        DatasetConfig.model_validate({"features": [], "targets": []})


def test_dataset_loads_sample_cadence_and_keys() -> None:
    dataset = DatasetConfig.model_validate(
        {
            "sample": {"cadence": "1d", "keys": ["security_id"]},
            "features": [],
            "targets": [],
        }
    )

    assert dataset.sample.cadence == "1d"
    assert dataset.sample.keys == ["security_id"]


def test_dataset_rejects_targets_without_features() -> None:
    with pytest.raises(ValidationError, match="must define at least one feature"):
        DatasetConfig.model_validate(
            {
                "sample": {"cadence": "1d"},
                "targets": [
                    {
                        "id": "return",
                        "stream": "returns",
                        "field": "value",
                        "horizon": "1d",
                    }
                ],
            }
        )


def test_dataset_requires_an_explicit_target_horizon() -> None:
    with pytest.raises(ValidationError, match="horizon"):
        DatasetConfig.model_validate(
            {
                "sample": {"cadence": "1d"},
                "features": [{"id": "price", "stream": "prices", "field": "close"}],
                "targets": [{"id": "return", "stream": "returns", "field": "value"}],
            }
        )


@pytest.mark.parametrize("horizon", ["0s", "35d"])
def test_dataset_accepts_nonnegative_target_horizons(horizon: str) -> None:
    dataset = DatasetConfig.model_validate(
        {
            "sample": {"cadence": "1d"},
            "features": [{"id": "price", "stream": "prices", "field": "close"}],
            "targets": [
                {
                    "id": "return",
                    "stream": "returns",
                    "field": "value",
                    "horizon": horizon,
                }
            ],
        }
    )

    assert dataset.targets[0].horizon == horizon


@pytest.mark.parametrize("horizon", ["-1s", "tomorrow"])
def test_dataset_rejects_invalid_target_horizons(horizon: str) -> None:
    with pytest.raises(ValidationError, match="horizon"):
        DatasetConfig.model_validate(
            {
                "sample": {"cadence": "1d"},
                "features": [{"id": "price", "stream": "prices", "field": "close"}],
                "targets": [
                    {
                        "id": "return",
                        "stream": "returns",
                        "field": "value",
                        "horizon": horizon,
                    }
                ],
            }
        )


def test_dataset_rejects_target_horizon_on_a_feature() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        DatasetConfig.model_validate(
            {
                "sample": {"cadence": "1d"},
                "features": [
                    {
                        "id": "price",
                        "stream": "prices",
                        "field": "close",
                        "horizon": "0s",
                    }
                ],
            }
        )


def test_dataset_rejects_positive_target_horizons_with_hash_splits() -> None:
    with pytest.raises(
        ValidationError,
        match="hash splits cannot be used with positive target horizons",
    ):
        DatasetConfig.model_validate(
            {
                "sample": {"cadence": "1d"},
                "features": [{"id": "price", "stream": "prices", "field": "close"}],
                "targets": [
                    {
                        "id": "return",
                        "stream": "returns",
                        "field": "value",
                        "horizon": "1d",
                    }
                ],
                "split": {
                    "mode": "hash",
                    "ratios": {"train": 0.8, "test": 0.2},
                    "folds": [
                        {
                            "id": "holdout",
                            "train": ["train"],
                            "test": ["test"],
                        }
                    ],
                },
            }
        )


def test_dataset_accepts_zero_target_horizon_with_hash_split() -> None:
    dataset = DatasetConfig.model_validate(
        {
            "sample": {"cadence": "1d"},
            "features": [{"id": "price", "stream": "prices", "field": "close"}],
            "targets": [
                {
                    "id": "return",
                    "stream": "returns",
                    "field": "value",
                    "horizon": "0s",
                }
            ],
            "split": {
                "mode": "hash",
                "ratios": {"train": 0.8, "test": 0.2},
                "folds": [
                    {
                        "id": "holdout",
                        "train": ["train"],
                        "test": ["test"],
                    }
                ],
            },
        }
    )

    assert dataset.targets[0].horizon == "0s"


def test_dataset_accepts_collected_series_with_hash_split() -> None:
    dataset = DatasetConfig.model_validate(
        {
            "sample": {"cadence": "1d"},
            "features": [
                {
                    "id": "intraday_price",
                    "stream": "prices.hourly",
                    "field": "close",
                    "collect": 24,
                }
            ],
            "split": {
                "mode": "hash",
                "ratios": {"train": 0.8, "test": 0.2},
                "folds": [
                    {
                        "id": "holdout",
                        "train": ["train"],
                        "test": ["test"],
                    }
                ],
            },
        }
    )

    assert dataset.features[0].collect == 24


def test_dataset_rejects_a_future_target_field_as_a_feature() -> None:
    with pytest.raises(
        ValidationError,
        match="feature 'future_return'.*future target",
    ):
        DatasetConfig.model_validate(
            {
                "sample": {"cadence": "1d"},
                "features": [
                    {
                        "id": "future_return",
                        "stream": "returns",
                        "field": "future",
                    }
                ],
                "targets": [
                    {
                        "id": "target",
                        "stream": "returns",
                        "field": "future",
                        "horizon": "1d",
                    }
                ],
            }
        )


def test_dataset_allows_a_contemporaneous_target_field_as_a_feature() -> None:
    dataset = DatasetConfig.model_validate(
        {
            "sample": {"cadence": "1d"},
            "features": [
                {
                    "id": "feature",
                    "stream": "returns",
                    "field": "value",
                }
            ],
            "targets": [
                {
                    "id": "target",
                    "stream": "returns",
                    "field": "value",
                    "horizon": "0s",
                }
            ],
        }
    )

    assert dataset.features[0].field == dataset.targets[0].field


def test_dataset_owns_split_and_postprocess_policy() -> None:
    dataset = DatasetConfig.model_validate(
        {
            "sample": {"cadence": "1d"},
            "split": {
                "mode": "hash",
                "ratios": {"train": 0.8, "test": 0.2},
                "folds": [
                    {
                        "id": "holdout",
                        "train": ["train"],
                        "test": ["test"],
                    }
                ],
            },
            "postprocess": {
                "samples": {"features": {"threshold": 0.9}},
            },
        }
    )

    assert dataset.split is not None
    assert dataset.postprocess.samples.features is not None
    assert dataset.postprocess.samples.features.threshold == 0.9


def test_dataset_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        DatasetConfig.model_validate(
            {
                "sample": {"cadence": "1d"},
                "unexpected": True,
            }
        )


def test_dataset_rejects_zero_cadence() -> None:
    with pytest.raises(ValidationError):
        DatasetConfig.model_validate({"sample": {"cadence": "0min"}})


@pytest.mark.parametrize("field", ["features", "targets"])
def test_dataset_rejects_null_series_lists(field: str) -> None:
    with pytest.raises(ValidationError, match=field):
        DatasetConfig.model_validate({"sample": {"cadence": "1d"}, field: None})


@pytest.mark.parametrize("keys", [[""], ["security_id", "security_id"]])
def test_dataset_rejects_invalid_sample_keys(keys: list[str]) -> None:
    with pytest.raises(ValidationError, match="at least 1 character|sample keys"):
        DatasetConfig.model_validate({"sample": {"cadence": "1d", "keys": keys}})


@pytest.mark.parametrize("duplicate_section", ["features", "targets"])
def test_dataset_rejects_duplicate_series_ids(
    duplicate_section: str,
) -> None:
    feature = {"id": "price", "stream": "prices", "field": "close"}
    target = {**feature, "horizon": "0s"}
    payload = {
        "sample": {"cadence": "1d"},
        "features": [feature],
        "targets": [],
    }
    duplicate = target if duplicate_section == "targets" else feature
    payload[duplicate_section] = [duplicate, duplicate]

    with pytest.raises(ValidationError, match="must be unique"):
        DatasetConfig.model_validate(payload)


def test_dataset_rejects_series_id_shared_by_feature_and_target() -> None:
    series = {"id": "price", "stream": "prices", "field": "close"}

    with pytest.raises(ValidationError, match="must be unique"):
        DatasetConfig.model_validate(
            {
                "sample": {"cadence": "1d"},
                "features": [series],
                "targets": [{**series, "horizon": "0s"}],
            }
        )


def test_dataset_series_preserves_feature_then_target_order() -> None:
    dataset = DatasetConfig.model_validate(
        {
            "sample": {"cadence": "1d"},
            "features": [{"id": "price", "stream": "prices", "field": "close"}],
            "targets": [
                {
                    "id": "return",
                    "stream": "returns",
                    "field": "value",
                    "horizon": "1d",
                }
            ],
        }
    )

    assert [series.id for series in dataset.series] == ["price", "return"]
    assert "series" not in dataset.model_dump()
