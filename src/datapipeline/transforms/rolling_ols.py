from collections.abc import Sequence


class RollingOls:
    """Maintain one multivariate least-squares model over a fixed window."""

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
        self._predictors = np.empty(
            (window, predictor_count),
            dtype=np.float64,
        )
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
        if len(predictors) != self._predictor_count:
            raise ValueError(
                "Rolling OLS predictor count does not match its configuration"
            )
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
