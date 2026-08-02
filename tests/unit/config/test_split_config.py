import pytest
from pydantic import TypeAdapter, ValidationError

from jerrythomas.config.dataset.split import (
    DatasetFold,
    HashSplitConfig,
    SplitConfig,
    TimeInterval,
    TimeSplitConfig,
    fold_output_id,
    resolve_fold_output,
    split_output_ids,
)


def _fold(
    fold_id: str = "fold_0",
    train: list[str] | None = None,
    validation: list[str] | None = None,
    test: list[str] | None = None,
) -> DatasetFold:
    return DatasetFold(
        id=fold_id,
        train=train or ["train"],
        validation=validation or [],
        test=test or [],
    )


def _interval(interval_id: str, until: str | None = None) -> TimeInterval:
    return TimeInterval(id=interval_id, until=until)


def test_hash_split_canonicalizes_ratio_order() -> None:
    fold = _fold()
    first = HashSplitConfig(
        ratios={"train": 0.8, "test": 0.2},
        folds=[fold],
    )
    second = HashSplitConfig(
        ratios={"test": 0.2, "train": 0.8},
        folds=[fold],
    )

    assert list(first.ratios.items()) == [("test", 0.2), ("train", 0.8)]
    assert first.ratios == second.ratios


def test_dataset_fold_accepts_optional_validation_and_test_roles() -> None:
    fold = DatasetFold(id="full", train=["train"])

    assert fold.validation == []
    assert fold.test == []


def test_dataset_fold_exposes_ordered_role_labels() -> None:
    fold = DatasetFold(
        id="fold_0",
        train=["train_0"],
        validation=["validation_0"],
    )

    assert fold.role_labels == (
        ("train", ("train_0",)),
        ("validation", ("validation_0",)),
        ("test", ()),
    )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"id": "", "train": ["train"]}, "id must not be empty"),
        ({"id": " fold ", "train": ["train"]}, "outer whitespace"),
        ({"id": "fold", "train": []}, "at least 1 item"),
        ({"id": "fold", "train": ["train", "train"]}, "duplicates"),
        ({"id": "fold", "train": [" "]}, "must not be empty"),
        ({"id": "fold", "train": [" train "]}, "outer whitespace"),
        (
            {
                "id": "fold",
                "train": ["train"],
                "validation": ["train"],
            },
            "only one role",
        ),
        ({"id": "fold", "train": ["train"], "unknown": True}, "Extra inputs"),
    ],
)
def test_dataset_fold_rejects_invalid_config(payload, message) -> None:
    with pytest.raises(ValidationError, match=message):
        DatasetFold.model_validate(payload)


def test_hash_split_allows_labels_to_be_reused_across_folds() -> None:
    config = HashSplitConfig(
        ratios={"train": 0.7, "validation": 0.2, "unused": 0.1},
        folds=[
            _fold("fold_0", validation=["validation"]),
            _fold("fold_1", validation=["validation"]),
        ],
    )

    assert config.folds[0].train == config.folds[1].train == ["train"]


def test_time_split_allows_intervals_to_be_reused_across_folds() -> None:
    config = TimeSplitConfig(
        intervals=[
            _interval("train_0", "2024-01-01T00:00:00Z"),
            _interval("validation_0", "2024-02-01T00:00:00Z"),
            _interval("unused", "2024-03-01T00:00:00Z"),
            _interval("train_1", "2024-04-01T00:00:00Z"),
            _interval("validation_1"),
        ],
        folds=[
            _fold("fold_0", ["train_0"], ["validation_0"]),
            _fold(
                "fold_1",
                ["train_0", "validation_0", "train_1"],
                ["validation_1"],
            ),
        ],
    )

    assert config.folds[1].train == ["train_0", "validation_0", "train_1"]


def test_time_split_accepts_one_open_ended_interval() -> None:
    config = TimeSplitConfig(
        intervals=[_interval("all")],
        folds=[_fold(train=["all"])],
    )

    assert config.intervals == [_interval("all")]


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"ratios": {"train": 1.0}}, "folds"),
        ({"mode": "hash", "folds": [_fold().model_dump()]}, "ratios"),
        ({"ratios": {}, "folds": [_fold().model_dump()]}, "ratios must not be empty"),
        (
            {
                "ratios": {"train": 0.0, "test": 1.0},
                "folds": [_fold().model_dump()],
            },
            "greater than 0",
        ),
        (
            {
                "ratios": {"train": 0.5, "test": 0.4},
                "folds": [_fold().model_dump()],
            },
            "must sum to 1.0",
        ),
        (
            {
                "ratios": {"": 1.0},
                "folds": [{"id": "fold_0", "train": [""]}],
            },
            "labels must not be empty",
        ),
        (
            {
                "ratios": {" train ": 1.0},
                "folds": [{"id": "fold_0", "train": [" train "]}],
            },
            "outer whitespace",
        ),
        (
            {
                "ratios": {"train": 1.0},
                "folds": [_fold().model_dump()],
                "seed": "42",
            },
            "valid integer",
        ),
        (
            {
                "ratios": {"train": 1.0},
                "folds": [
                    _fold("same").model_dump(),
                    _fold("same").model_dump(),
                ],
            },
            "ids must be unique",
        ),
        (
            {
                "ratios": {"train": 1.0},
                "folds": [_fold(train=["unknown"]).model_dump()],
            },
            "unknown hash split labels: unknown",
        ),
        (
            {
                "ratios": {"train": 1.0},
                "folds": [_fold().model_dump()],
                "key": "group",
            },
            "Extra inputs",
        ),
        (
            {
                "ratios": {"train": 1.0},
                "folds": [_fold().model_dump()],
                "output_labels": ["train"],
            },
            "Extra inputs",
        ),
    ],
)
def test_hash_split_rejects_invalid_config(payload, message) -> None:
    with pytest.raises(ValidationError, match=message):
        HashSplitConfig.model_validate(payload)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {"mode": "time", "folds": [_fold().model_dump()]},
            "intervals",
        ),
        (
            {"mode": "time", "intervals": [], "folds": [_fold().model_dump()]},
            "at least 1 item",
        ),
        (
            {
                "mode": "time",
                "intervals": [{"id": "train"}],
                "folds": [_fold().model_dump()],
                "boundaries": [],
                "labels": ["train"],
            },
            "Extra inputs",
        ),
        (
            {
                "mode": "time",
                "intervals": [
                    {"id": "train", "until": "not-a-datetime"},
                    {"id": "test"},
                ],
                "folds": [_fold().model_dump()],
            },
            "Invalid ISO-8601 datetime",
        ),
        (
            {
                "mode": "time",
                "intervals": [
                    {"id": "train", "until": "2024-02-01T00:00:00Z"},
                    {"id": "validation", "until": "2024-01-01T00:00:00Z"},
                    {"id": "test"},
                ],
                "folds": [_fold(validation=["validation"], test=["test"]).model_dump()],
            },
            "strictly increasing",
        ),
        (
            {
                "mode": "time",
                "intervals": [
                    {"id": "train", "until": "2024-01-01T00:00:00Z"},
                    {"id": "train"},
                ],
                "folds": [_fold().model_dump()],
            },
            "ids must be unique",
        ),
        (
            {
                "mode": "time",
                "intervals": [{"id": " "}],
                "folds": [_fold().model_dump()],
            },
            "id must not be empty",
        ),
        (
            {
                "mode": "time",
                "intervals": [{"id": " train "}],
                "folds": [_fold().model_dump()],
            },
            "id must not contain outer whitespace",
        ),
        (
            {
                "mode": "time",
                "intervals": [
                    {"id": "train", "until": ""},
                    {"id": "test"},
                ],
                "folds": [_fold().model_dump()],
            },
            "until must not be empty",
        ),
        (
            {
                "mode": "time",
                "intervals": [
                    {"id": "train", "until": " 2024-01-01T00:00:00Z "},
                    {"id": "test"},
                ],
                "folds": [_fold().model_dump()],
            },
            "until must not contain outer whitespace",
        ),
        (
            {
                "mode": "time",
                "intervals": [
                    {"id": "train"},
                    {"id": "test"},
                ],
                "folds": [_fold().model_dump()],
            },
            "except the final interval requires until",
        ),
        (
            {
                "mode": "time",
                "intervals": [
                    {"id": "train", "until": "2024-01-01T00:00:00Z"},
                ],
                "folds": [_fold().model_dump()],
            },
            "final time interval must omit until",
        ),
        (
            {
                "mode": "time",
                "intervals": [
                    {"id": "train", "until": "2024-01-01T00:00:00Z"},
                    {"id": "test"},
                ],
                "folds": [
                    _fold("same").model_dump(),
                    _fold("same").model_dump(),
                ],
            },
            "ids must be unique",
        ),
        (
            {
                "mode": "time",
                "intervals": [
                    {"id": "train", "until": "2024-01-01T00:00:00Z"},
                    {"id": "test"},
                ],
                "folds": [_fold(validation=["unknown"]).model_dump()],
            },
            "unknown time intervals: unknown",
        ),
        (
            {
                "mode": "time",
                "intervals": [
                    {"id": "validation", "until": "2024-01-01T00:00:00Z"},
                    {"id": "unused", "until": "2024-02-01T00:00:00Z"},
                    {"id": "train"},
                ],
                "folds": [_fold(validation=["validation"]).model_dump()],
            },
            "train intervals before validation intervals",
        ),
        (
            {
                "mode": "time",
                "intervals": [
                    {"id": "train", "until": "2024-01-01T00:00:00Z"},
                    {"id": "test", "until": "2024-02-01T00:00:00Z"},
                    {"id": "validation"},
                ],
                "folds": [_fold(validation=["validation"], test=["test"]).model_dump()],
            },
            "validation intervals before test intervals",
        ),
        (
            {
                "mode": "time",
                "intervals": [
                    {"id": "train", "until": "2024-01-01T00:00:00Z"},
                    {"id": "test"},
                ],
                "folds": [_fold(test=["test"]).model_dump()],
                "output_labels": ["train"],
            },
            "Extra inputs",
        ),
    ],
)
def test_time_split_rejects_invalid_config(payload, message) -> None:
    with pytest.raises(ValidationError, match=message):
        TimeSplitConfig.model_validate(payload)


def test_fold_output_helpers_expose_only_configured_roles() -> None:
    first = _fold("walk_0", ["train_0"], ["validation_0"])
    second = _fold("walk_1", ["train_0", "validation_0", "train_1"], test=["test"])
    config = TimeSplitConfig(
        intervals=[
            _interval("train_0", "2024-01-01T00:00:00Z"),
            _interval("validation_0", "2024-02-01T00:00:00Z"),
            _interval("train_1", "2024-03-01T00:00:00Z"),
            _interval("test"),
        ],
        folds=[first, second],
    )

    assert fold_output_id("walk_0", "validation") == "walk_0.validation"
    assert split_output_ids(config) == (
        "walk_0.train",
        "walk_0.validation",
        "walk_1.train",
        "walk_1.test",
    )
    assert resolve_fold_output(config, "walk_1.train") == (
        second,
        "train",
        ("train_0", "validation_0", "train_1"),
    )

    with pytest.raises(KeyError, match="fold output 'walk_0.test'"):
        resolve_fold_output(config, "walk_0.test")


def test_split_union_requires_an_explicit_mode() -> None:
    with pytest.raises(ValidationError, match="mode"):
        TypeAdapter(SplitConfig).validate_python(
            {
                "ratios": {"train": 1.0},
                "folds": [_fold().model_dump()],
            }
        )


def test_split_config_is_frozen() -> None:
    config = HashSplitConfig(ratios={"train": 1.0}, folds=[_fold()])

    with pytest.raises(ValidationError, match="frozen"):
        config.seed = 7
