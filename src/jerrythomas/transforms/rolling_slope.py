from collections import deque
from math import isfinite

from jerrythomas.transforms.rolling_window import CompensatedSum


class RollingSlope:
    """Maintain the least-squares slope of y on x over a fixed window."""

    def __init__(self, window: int) -> None:
        self.window = window
        self.points: deque[tuple[float, float]] = deque()
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.x_offsets = CompensatedSum()
        self.y_offsets = CompensatedSum()
        self.squared_x_offsets = CompensatedSum()
        self.cross_offsets = CompensatedSum()
        self.removals_until_rebase = 0

    @property
    def full(self) -> bool:
        return len(self.points) == self.window

    def clear(self) -> None:
        self.points.clear()
        self._reset(0.0, 0.0)
        self.removals_until_rebase = 0

    def append(self, x: float, y: float) -> None:
        if self.full:
            expired_x, expired_y = self.points.popleft()
            self._remove(expired_x, expired_y)

        if not self.points:
            self._reset(x, y)
            self.removals_until_rebase = 1
        else:
            self._add_offsets(x, y)
        self.points.append((x, y))

    def result(self) -> float:
        sample_count = len(self.points)
        x_sum = self.x_offsets.result()
        y_sum = self.y_offsets.result()
        squared_x_sum = self.squared_x_offsets.result()
        cross_sum = self.cross_offsets.result()
        x_mean = x_sum / sample_count
        y_mean = y_sum / sample_count
        squared_deviations = squared_x_sum - x_sum * x_mean
        cross_deviations = cross_sum - x_sum * y_mean
        if not isfinite(squared_deviations) or not isfinite(cross_deviations):
            raise OverflowError(
                "Rolling slope exceeds the supported floating-point range"
            )

        squared_deviations = max(squared_deviations, 0.0)
        if squared_deviations == 0.0:
            raise ZeroDivisionError(
                "Cannot calculate rolling slope when x has zero variance"
            )
        slope = cross_deviations / squared_deviations
        if not isfinite(slope):
            raise OverflowError(
                "Rolling slope exceeds the supported floating-point range"
            )
        return slope

    def _remove(self, x: float, y: float) -> None:
        self.removals_until_rebase -= 1
        if self.removals_until_rebase == 0:
            self._rebase()
            return

        x_offset, y_offset, squared_x_offset, cross_offset = self._offsets(x, y)
        self.x_offsets.add(-x_offset)
        self.y_offsets.add(-y_offset)
        self.squared_x_offsets.add(-squared_x_offset)
        self.cross_offsets.add(-cross_offset)

    def _reset(self, origin_x: float, origin_y: float) -> None:
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.x_offsets.reset()
        self.y_offsets.reset()
        self.squared_x_offsets.reset()
        self.cross_offsets.reset()

    def _rebase(self) -> None:
        if not self.points:
            self._reset(0.0, 0.0)
            self.removals_until_rebase = 0
            return

        self._reset(*self.points[-1])
        for x, y in self.points:
            self._add_offsets(x, y)
        self.removals_until_rebase = len(self.points)

    def _add_offsets(self, x: float, y: float) -> None:
        x_offset, y_offset, squared_x_offset, cross_offset = self._offsets(x, y)
        self.x_offsets.add(x_offset)
        self.y_offsets.add(y_offset)
        self.squared_x_offsets.add(squared_x_offset)
        self.cross_offsets.add(cross_offset)

    def _offsets(self, x: float, y: float) -> tuple[float, float, float, float]:
        x_offset = x - self.origin_x
        y_offset = y - self.origin_y
        squared_x_offset = x_offset * x_offset
        cross_offset = x_offset * y_offset
        if not all(
            isfinite(value)
            for value in (x_offset, y_offset, squared_x_offset, cross_offset)
        ):
            raise OverflowError(
                "Rolling slope exceeds the supported floating-point range"
            )
        return x_offset, y_offset, squared_x_offset, cross_offset
