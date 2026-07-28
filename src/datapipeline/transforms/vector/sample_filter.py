from collections.abc import Iterator, Sequence

from datapipeline.domain.sample import Sample

from .common import cell_coverage, reject_unknown_values, select_expected_ids


def _threshold(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("threshold must be a number between 0 and 1")
    threshold = float(value)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    return threshold


class FilterFeatureSamplesTransform:
    """Keep samples with sufficient feature coverage."""

    def __init__(
        self,
        expected_ids: Sequence[str],
        threshold: float,
        ids: Sequence[str] | None = None,
    ) -> None:
        self.expected_ids, self.ids = select_expected_ids(expected_ids, ids)
        self.expected_id_set = frozenset(self.expected_ids)
        self.threshold = _threshold(threshold)

    def apply(self, stream: Iterator[Sample]) -> Iterator[Sample]:
        for sample in stream:
            values = sample.features.values
            reject_unknown_values(values, self.expected_id_set)
            coverage = sum(
                cell_coverage(values.get(identifier)) for identifier in self.ids
            ) / len(self.ids)
            if coverage >= self.threshold:
                yield sample


class FilterTargetSamplesTransform:
    """Keep samples with sufficient target coverage."""

    def __init__(
        self,
        expected_ids: Sequence[str],
        threshold: float,
        ids: Sequence[str] | None = None,
    ) -> None:
        self.expected_ids, self.ids = select_expected_ids(expected_ids, ids)
        self.expected_id_set = frozenset(self.expected_ids)
        self.threshold = _threshold(threshold)

    def apply(self, stream: Iterator[Sample]) -> Iterator[Sample]:
        for sample in stream:
            values = sample.targets.values if sample.targets is not None else {}
            reject_unknown_values(values, self.expected_id_set)
            coverage = sum(
                cell_coverage(values.get(identifier)) for identifier in self.ids
            ) / len(self.ids)
            if coverage >= self.threshold:
                yield sample
