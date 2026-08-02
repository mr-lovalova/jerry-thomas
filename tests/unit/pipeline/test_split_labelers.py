from datetime import datetime, timedelta, timezone

import pytest

from jerrythomas.config.dataset.split import (
    DatasetFold,
    HashSplitConfig,
    TimeInterval,
    TimeSplitConfig,
)
from jerrythomas.pipelines.dataset.split import HashLabeler, TimeLabeler


def _fold() -> DatasetFold:
    return DatasetFold(
        id="default",
        train=["train"],
        validation=["val"],
        test=["test"],
    )


def _time_labeler() -> TimeLabeler:
    return TimeLabeler(
        TimeSplitConfig(
            intervals=[
                TimeInterval(id="train", until="1970-01-02T00:00:00Z"),
                TimeInterval(id="test"),
            ],
            folds=[DatasetFold(id="default", train=["train"], test=["test"])],
        )
    )


def test_hash_split_ratio_mapping_order_does_not_change_labels() -> None:
    first = HashLabeler(
        HashSplitConfig(
            ratios={"train": 0.7, "val": 0.2, "test": 0.1},
            folds=[_fold()],
            seed=7,
        )
    )
    second = HashLabeler(
        HashSplitConfig(
            ratios={"test": 0.1, "train": 0.7, "val": 0.2},
            folds=[_fold()],
            seed=7,
        )
    )
    assert [first.label(index) for index in range(1_000)] == [
        second.label(index) for index in range(1_000)
    ]


def test_hash_split_uses_the_whole_group_key() -> None:
    labeler = HashLabeler(
        HashSplitConfig(
            ratios={"train": 0.5, "test": 0.5},
            folds=[
                DatasetFold(id="default", train=["train"], test=["test"]),
            ],
            seed=7,
        )
    )

    first = [labeler.label(("A", index)) for index in range(100)]
    second = [labeler.label(("B", index)) for index in range(100)]

    assert first != second


def test_hash_split_routing_is_stable_and_seed_sensitive() -> None:
    keys = [("tenant", index) for index in (0, 2, 4, 6, 8, 14)]

    def labels(seed: int) -> list[str]:
        labeler = HashLabeler(
            HashSplitConfig(
                ratios={"left": 0.2, "middle": 0.3, "right": 0.5},
                folds=[
                    DatasetFold(
                        id="default",
                        train=["left"],
                        validation=["middle"],
                        test=["right"],
                    )
                ],
                seed=seed,
            )
        )
        return [labeler.label(key) for key in keys]

    assert labels(7) == [
        "right",
        "right",
        "middle",
        "middle",
        "middle",
        "right",
    ]
    assert labels(8) == [
        "right",
        "middle",
        "right",
        "left",
        "right",
        "middle",
    ]


def test_hash_split_thresholds_are_half_open(monkeypatch) -> None:
    labeler = HashLabeler(
        HashSplitConfig(
            ratios={"left": 0.4, "right": 0.6},
            folds=[
                DatasetFold(
                    id="default",
                    train=["left"],
                    test=["right"],
                )
            ],
            seed=7,
        )
    )
    monkeypatch.setattr(labeler, "_hash_token", lambda token, seed: 0.4)

    assert labeler.label("boundary") == "right"


def test_time_labeler_uses_intervals():
    labeler = _time_labeler()

    assert labeler.label(datetime(1970, 1, 1, tzinfo=timezone.utc)) == "train"
    assert labeler.label(datetime(1970, 1, 2, tzinfo=timezone.utc)) == "test"


def test_time_labeler_accepts_iso_string_keys():
    labeler = _time_labeler()

    assert labeler.label("1970-01-02T00:00:00Z") == "test"


def test_time_labeler_treats_naive_datetimes_as_utc() -> None:
    labeler = _time_labeler()

    assert labeler.label(datetime(1970, 1, 1, 23, 59, 59)) == "train"
    assert labeler.label(datetime(1970, 1, 2)) == "test"


def test_time_labeler_compares_aware_datetimes_by_instant() -> None:
    labeler = _time_labeler()
    utc_plus_one = timezone(timedelta(hours=1))

    assert labeler.label(datetime(1970, 1, 2, 0, 59, 59, tzinfo=utc_plus_one)) == (
        "train"
    )
    assert labeler.label(datetime(1970, 1, 2, 1, tzinfo=utc_plus_one)) == "test"


def test_time_labeler_rejects_ambiguous_keys():
    labeler = _time_labeler()

    with pytest.raises(TypeError, match="datetimes or ISO-8601 strings"):
        labeler.label(1)
