import hashlib
from bisect import bisect_right
from datetime import datetime, timedelta
from typing import Any

from datapipeline.config.dataset.split import (
    DatasetFold,
    HashSplitConfig,
    SplitConfig,
    TimeSplitConfig,
)
from datapipeline.utils.time import parse_datetime


class HashLabeler:
    """Assign deterministic labels from a validated hash split."""

    def __init__(self, config: HashSplitConfig) -> None:
        total = 0.0
        thresholds: list[tuple[float, str]] = []
        for label, frac in config.ratios.items():
            total += frac
            thresholds.append((total, label))
        self._thresholds = thresholds
        self._seed = config.seed

    @staticmethod
    def _hash_token(token: str, seed: int) -> float:
        b = (str(seed) + "|" + token).encode("utf-8")
        digest = hashlib.sha256(b).digest()
        num = int.from_bytes(digest[:8], "big")
        return (num % (1 << 53)) / float(1 << 53)

    def label(self, group_key: Any) -> str:
        token = repr(group_key)
        r = self._hash_token(token, self._seed)
        for thresh, label in self._thresholds:
            if r < thresh:
                return label
        return self._thresholds[-1][1]


class TimeLabeler:
    """Assign the interval containing a timestamp."""

    def __init__(self, config: TimeSplitConfig) -> None:
        self._boundaries = tuple(
            parse_datetime(interval.until)
            for interval in config.intervals
            if interval.until is not None
        )
        self._labels = tuple(interval.id for interval in config.intervals)

    def label(self, group_key: Any) -> str:
        return self._labels[bisect_right(self._boundaries, _sample_time(group_key))]


class TargetHorizonPolicy:
    """Define which sample origins are safe for each role in a time fold."""

    def __init__(
        self,
        config: TimeSplitConfig,
        fold: DatasetFold,
        horizon: timedelta,
    ) -> None:
        if horizon < timedelta():
            raise ValueError("target horizon must not be negative")

        interval_positions = {
            interval.id: position for position, interval in enumerate(config.intervals)
        }
        interval_starts: dict[str, datetime] = {}
        for previous, current in zip(config.intervals, config.intervals[1:]):
            assert previous.until is not None
            interval_starts[current.id] = parse_datetime(previous.until)

        roles = (fold.train, fold.validation, fold.test)
        cutoffs: dict[str, datetime | None] = {}
        for role_index, labels in enumerate(roles):
            next_labels = next(
                (candidate for candidate in roles[role_index + 1 :] if candidate),
                (),
            )
            cutoff = (
                None
                if not next_labels
                else interval_starts[
                    min(next_labels, key=interval_positions.__getitem__)
                ]
            )
            for label in labels:
                cutoffs[label] = cutoff

        self._cutoffs = cutoffs
        self._horizon = horizon

    def allows(self, label: str, group_key: Any) -> bool:
        cutoff = self._cutoffs[label]
        return cutoff is None or _sample_time(group_key) + self._horizon < cutoff


def build_labeler(config: SplitConfig) -> HashLabeler | TimeLabeler:
    if isinstance(config, TimeSplitConfig):
        return TimeLabeler(config)
    return HashLabeler(config)


def _sample_time(group_key: Any) -> datetime:
    key = group_key[0] if isinstance(group_key, (list, tuple)) else group_key
    if isinstance(key, datetime):
        return key if key.tzinfo is not None else parse_datetime(key.isoformat())
    if isinstance(key, str):
        return parse_datetime(key)
    raise TypeError("time split keys must be datetimes or ISO-8601 strings")
