from collections.abc import Iterator, Sequence

from datapipeline.domain.record import TemporalRecord
from datapipeline.transforms.utils import (
    adjacent_partitions,
    clone_record_with_field,
    finite_number_or_none,
    get_field,
)


class _RollingOlsWindow:
    def __init__(
        self,
        window: int,
        predictor_count: int,
        coefficient_index: int,
    ) -> None:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - exercised by runtime users
            raise RuntimeError(
                "rolling_ols requires NumPy; install jerry-thomas[numerical]."
            ) from exc

        self._np = np
        self._window = window
        self._predictor_count = predictor_count
        self._coefficient_index = coefficient_index
        self._predictors = np.empty((window, predictor_count), dtype=np.float64)
        self._response = np.empty(window, dtype=np.float64)
        self._position = 0
        self._size = 0

    @property
    def full(self) -> bool:
        return self._size == self._window

    def clear(self) -> None:
        self._position = 0
        self._size = 0

    def append(self, predictors: Sequence[float], response: float) -> None:
        self._predictors[self._position] = predictors
        self._response[self._position] = response
        self._position = (self._position + 1) % self._window
        self._size = min(self._size + 1, self._window)

    def result(self) -> float:
        if not self.full:
            raise RuntimeError("Rolling OLS window is not full")

        np = self._np
        try:
            with np.errstate(over="raise", invalid="raise"):
                predictors = self._predictors - self._predictors.mean(axis=0)
                response = self._response - self._response.mean()
                coefficients, _, rank, _ = np.linalg.lstsq(
                    predictors,
                    response,
                    rcond=None,
                )
        except FloatingPointError as exc:
            raise OverflowError(
                "Rolling OLS exceeds the supported floating-point range"
            ) from exc

        if rank != self._predictor_count:
            raise ValueError("Rolling OLS design matrix is rank deficient")
        coefficient = float(coefficients[self._coefficient_index])
        if not np.isfinite(coefficient):
            raise OverflowError(
                "Rolling OLS exceeds the supported floating-point range"
            )
        return coefficient


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
        rolling_ols = _RollingOlsWindow(
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
