import math
from collections.abc import Sequence

from datapipeline.domain.record import TemporalRecord
from datapipeline.transforms.utils import (
    clone_record_with_field,
    finite_number_or_none,
    get_field,
    is_missing,
)


_RankValue = int | float


def _rank_value_or_none(value: object, field: str) -> _RankValue | None:
    if is_missing(value):
        return None
    if isinstance(value, bool):
        raise TypeError(f"Field {field!r} must contain numeric values")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Field {field!r} must contain finite numeric values")
        return value
    raise TypeError(f"Field {field!r} must contain numeric values")


class RankScoreTransform:
    """Score one numeric field against the complete cross-section."""

    def __init__(self, field: str, to: str, min_samples: int) -> None:
        self.field = field
        self.to = to
        self.min_samples = min_samples

    def apply(
        self,
        records: Sequence[TemporalRecord],
    ) -> tuple[TemporalRecord, ...]:
        values = [
            _rank_value_or_none(get_field(record, self.field), self.field)
            for record in records
        ]
        complete = [
            (position, value)
            for position, value in enumerate(values)
            if value is not None
        ]
        scores: list[float | None] = [None] * len(records)
        if len(complete) >= self.min_samples:
            self._score_complete_values(complete, scores)

        return tuple(
            clone_record_with_field(record, self.to, score)
            for record, score in zip(records, scores, strict=True)
        )

    @staticmethod
    def _score_complete_values(
        complete: list[tuple[int, _RankValue]],
        scores: list[float | None],
    ) -> None:
        ordered = sorted(complete, key=lambda item: item[1])
        sample_count = len(ordered)
        start = 0
        while start < sample_count:
            end = start + 1
            value = ordered[start][1]
            while end < sample_count and ordered[end][1] == value:
                end += 1

            average_rank = ((start + 1) + end) / 2.0
            score = (average_rank - 1.0) / (sample_count - 1) - 0.5
            for position, _ in ordered[start:end]:
                scores[position] = score
            start = end


class OlsResidualTransform:
    """Residualize one numeric field against a complete cross-section."""

    def __init__(
        self,
        y: str,
        x: tuple[str, ...],
        to: str,
        min_samples: int,
    ) -> None:
        self.y = y
        self.x = x
        self.to = to
        self.min_samples = min_samples

    def apply(
        self,
        records: Sequence[TemporalRecord],
    ) -> tuple[TemporalRecord, ...]:
        complete: list[tuple[int, list[float], float]] = []
        for position, record in enumerate(records):
            predictors = [
                finite_number_or_none(get_field(record, field), field)
                for field in self.x
            ]
            response = finite_number_or_none(get_field(record, self.y), self.y)
            if response is None or any(value is None for value in predictors):
                continue
            complete.append(
                (
                    position,
                    [value for value in predictors if value is not None],
                    response,
                )
            )

        residuals: list[float | None] = [None] * len(records)
        if len(complete) >= self.min_samples:
            fitted = self._fit_residuals(complete)
            for (position, _, _), residual in zip(complete, fitted, strict=True):
                residuals[position] = residual

        return tuple(
            clone_record_with_field(record, self.to, residual)
            for record, residual in zip(records, residuals, strict=True)
        )

    def _fit_residuals(
        self,
        complete: list[tuple[int, list[float], float]],
    ) -> tuple[float, ...]:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - exercised by runtime users
            raise RuntimeError(
                "OLS residual requires NumPy; install jerry-thomas[numerical]."
            ) from exc

        predictors = np.asarray(
            [values for _, values, _ in complete],
            dtype=np.float64,
        )
        response = np.asarray(
            [value for _, _, value in complete],
            dtype=np.float64,
        )
        try:
            with np.errstate(over="raise", invalid="raise"):
                centered_predictors = predictors - predictors.mean(axis=0)
                centered_response = response - response.mean()
                coefficients, _, rank, _ = np.linalg.lstsq(
                    centered_predictors,
                    centered_response,
                    rcond=None,
                )
                residuals = centered_response - centered_predictors @ coefficients
        except FloatingPointError as exc:
            raise OverflowError(
                "OLS residual exceeds the supported floating-point range"
            ) from exc

        if rank != len(self.x):
            raise ValueError("OLS residual design matrix is rank deficient")
        if not np.all(np.isfinite(coefficients)) or not np.all(np.isfinite(residuals)):
            raise OverflowError(
                "OLS residual exceeds the supported floating-point range"
            )
        return tuple(float(value) for value in residuals)
