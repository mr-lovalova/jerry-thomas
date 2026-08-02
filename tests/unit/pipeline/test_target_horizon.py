from datetime import datetime, timedelta, timezone

from jerrythomas.config.dataset.dataset import DatasetConfig, SampleConfig
from jerrythomas.config.dataset.series import SeriesConfig, TargetSeriesConfig
from jerrythomas.config.dataset.split import DatasetFold, TimeInterval, TimeSplitConfig
from jerrythomas.pipelines.dataset.split import TargetHorizonPolicy


def _time(day: int) -> datetime:
    return datetime(2024, 1, day, tzinfo=timezone.utc)


def test_target_horizon_policy_removes_samples_reaching_the_next_role() -> None:
    fold = DatasetFold(id="fold", train=["train"], validation=["validation"])
    split = TimeSplitConfig(
        intervals=[
            TimeInterval(id="train", until="2024-01-05T00:00:00Z"),
            TimeInterval(id="validation"),
        ],
        folds=[fold],
    )
    policy = TargetHorizonPolicy(split, fold, timedelta(days=2))

    assert policy.allows("train", (_time(2),))
    assert not policy.allows("train", (_time(3),))
    assert policy.allows("validation", (_time(5),))


def test_target_horizon_policy_allows_support_inside_a_purge_gap() -> None:
    fold = DatasetFold(id="fold", train=["train"], validation=["validation"])
    split = TimeSplitConfig(
        intervals=[
            TimeInterval(id="train", until="2024-01-03T00:00:00Z"),
            TimeInterval(id="purge", until="2024-01-05T00:00:00Z"),
            TimeInterval(id="validation"),
        ],
        folds=[fold],
    )
    policy = TargetHorizonPolicy(split, fold, timedelta(days=2))

    assert policy.allows("train", (_time(2),))


def test_target_horizon_policy_uses_the_next_nonempty_role() -> None:
    fold = DatasetFold(id="fold", train=["train"], test=["test"])
    split = TimeSplitConfig(
        intervals=[
            TimeInterval(id="train", until="2024-01-05T00:00:00Z"),
            TimeInterval(id="test"),
        ],
        folds=[fold],
    )
    policy = TargetHorizonPolicy(split, fold, timedelta(days=1))

    assert policy.allows("train", (_time(3),))
    assert not policy.allows("train", (_time(4),))
    assert policy.allows("test", (_time(5),))


def test_dataset_max_target_horizon_uses_the_longest_target() -> None:
    dataset = DatasetConfig(
        sample=SampleConfig(cadence="1d"),
        targets=[
            TargetSeriesConfig(
                id="contemporaneous",
                stream="targets",
                field="now",
                horizon="0s",
            ),
            TargetSeriesConfig(
                id="forward",
                stream="targets",
                field="future",
                horizon="35d",
            ),
        ],
        features=[
            SeriesConfig(
                id="feature",
                stream="features",
                field="value",
            )
        ],
    )

    assert dataset.max_target_horizon == timedelta(days=35)
