from collections.abc import Iterator

from datapipeline.domain.record import TemporalRecord
from datapipeline.transforms.rolling_ols import RollingOls
from datapipeline.transforms.utils import (
    adjacent_partitions,
    clone_record_with_field,
    finite_number_or_none,
    get_field,
)


class RollingOlsTransform:
    """Compute one coefficient from a rolling multivariate OLS model."""

    def __init__(
        self,
        y: str,
        x: tuple[str, ...],
        window: int,
        coefficient: str,
        partition_fields: tuple[str, ...],
        to: str,
    ) -> None:
        self.y = y
        self.x = x
        self.window = window
        self.coefficient_index = x.index(coefficient)
        self.partition_fields = partition_fields
        self.to = to

    def apply(self, stream: Iterator[TemporalRecord]) -> Iterator[TemporalRecord]:
        rolling_ols = RollingOls(
            self.window,
            len(self.x),
            self.coefficient_index,
        )
        for _, records in adjacent_partitions(stream, self.partition_fields):
            rolling_ols.clear()

            for record in records:
                predictors: list[float] = []
                missing_predictor = False
                for field in self.x:
                    value = finite_number_or_none(get_field(record, field), field)
                    if value is None:
                        missing_predictor = True
                    else:
                        predictors.append(value)
                response = finite_number_or_none(get_field(record, self.y), self.y)
                if response is None or missing_predictor:
                    rolling_ols.clear()
                    coefficient = None
                else:
                    rolling_ols.append(predictors, response)
                    coefficient = rolling_ols.result() if rolling_ols.full else None
                yield clone_record_with_field(record, self.to, coefficient)
