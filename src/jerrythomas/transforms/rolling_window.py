from abc import ABC, abstractmethod
from bisect import bisect_left, insort
from collections import deque
from math import sqrt


class CompensatedSum:
    def __init__(self) -> None:
        self.total = 0.0
        self.correction = 0.0

    def add(self, value: float) -> None:
        updated = self.total + value
        if abs(self.total) >= abs(value):
            self.correction += (self.total - updated) + value
        else:
            self.correction += (value - updated) + self.total
        self.total = updated

    def reset(self) -> None:
        self.total = 0.0
        self.correction = 0.0

    def result(self) -> float:
        return self.total + self.correction


class RollingWindow(ABC):
    """Maintain one statistic over a fixed number of ticks."""

    def __init__(self, window: int) -> None:
        if window <= 0:
            raise ValueError("rolling window must be positive")
        self.window = window
        self.ticks: deque[float | None] = deque()
        self.sample_count = 0

    def append(self, value: float | None) -> None:
        if len(self.ticks) == self.window:
            expired = self.ticks.popleft()
            if expired is not None:
                self._remove(expired)
                self.sample_count -= 1

        if value is not None:
            self._add(value)
            self.sample_count += 1
        self.ticks.append(value)

    @abstractmethod
    def _add(self, value: float) -> None: ...

    @abstractmethod
    def _remove(self, value: float) -> None: ...

    @abstractmethod
    def result(self) -> float: ...


class RollingSum(RollingWindow):
    def __init__(self, window: int) -> None:
        super().__init__(window)
        self.values = CompensatedSum()
        self.removals_since_rebase = 0

    def _add(self, value: float) -> None:
        self.values.add(value)

    def _remove(self, value: float) -> None:
        if self.sample_count == 1:
            self.values.reset()
            self.removals_since_rebase = 0
            return

        self.values.add(-value)
        self.removals_since_rebase += 1
        if self.removals_since_rebase == self.window:
            self.values.reset()
            for retained in self.ticks:
                if retained is not None:
                    self.values.add(retained)
            self.removals_since_rebase = 0

    def result(self) -> float:
        return self.values.result()


class RollingMean(RollingSum):
    def result(self) -> float:
        return super().result() / self.sample_count


class RollingMoments(RollingWindow):
    """Maintain numerically stable first and second rolling moments."""

    def __init__(self, window: int) -> None:
        super().__init__(window)
        self.origin = 0.0
        self.offsets = CompensatedSum()
        self.squared_offsets = CompensatedSum()
        self.valid_removals_until_rebase = 0

    def _add(self, value: float) -> None:
        if self.sample_count == 0:
            self._reset(value)
            self.valid_removals_until_rebase = 1
            return

        offset = value - self.origin
        self.offsets.add(offset)
        self.squared_offsets.add(offset * offset)

    def _remove(self, value: float) -> None:
        self.valid_removals_until_rebase -= 1
        if self.valid_removals_until_rebase == 0:
            self._rebase()
            return

        offset = value - self.origin
        self.offsets.add(-offset)
        self.squared_offsets.add(-(offset * offset))

    def _reset(self, origin: float) -> None:
        self.origin = origin
        self.offsets.reset()
        self.squared_offsets.reset()

    def _rebase(self) -> None:
        values = [value for value in self.ticks if value is not None]
        if not values:
            self._reset(0.0)
            self.valid_removals_until_rebase = 0
            return
        self._reset(values[-1])
        for value in values:
            offset = value - self.origin
            self.offsets.add(offset)
            self.squared_offsets.add(offset * offset)
        self.valid_removals_until_rebase = len(values)

    def _squared_deviation_sum(self) -> float:
        offset_sum = self.offsets.result()
        deviation_sum = (
            self.squared_offsets.result() - offset_sum * offset_sum / self.sample_count
        )
        return max(deviation_sum, 0.0)


class RollingSampleStandardDeviation(RollingMoments):
    def result(self) -> float:
        if self.sample_count < 2:
            raise ValueError("sample standard deviation requires at least two samples")
        return sqrt(self._squared_deviation_sum() / (self.sample_count - 1))


class RollingPopulationStandardDeviation(RollingMoments):
    def result(self) -> float:
        return sqrt(self._squared_deviation_sum() / self.sample_count)


class RollingQuantile(RollingWindow):
    """Maintain a linearly interpolated quantile over a rolling window."""

    def __init__(self, window: int, quantile: float) -> None:
        super().__init__(window)
        if not 0.0 <= quantile <= 1.0:
            raise ValueError("rolling quantile must be between 0 and 1")
        self.quantile = quantile
        self.values: list[float] = []

    def _add(self, value: float) -> None:
        insort(self.values, value)

    def _remove(self, value: float) -> None:
        self.values.pop(bisect_left(self.values, value))

    def result(self) -> float:
        if not self.values:
            raise ValueError("rolling quantile requires at least one sample")

        position = (self.sample_count - 1) * self.quantile
        lower_index = int(position)
        weight = position - lower_index
        lower = self.values[lower_index]
        if weight == 0.0:
            return lower

        upper = self.values[lower_index + 1]
        if lower < 0 < upper:
            if weight == 0.5:
                return (lower + upper) / 2
            return lower * (1.0 - weight) + upper * weight
        return lower + (upper - lower) * weight


class RollingMedian(RollingQuantile):
    def __init__(self, window: int) -> None:
        super().__init__(window, quantile=0.5)


class RollingMaximum(RollingWindow):
    def __init__(self, window: int) -> None:
        super().__init__(window)
        self.candidates: deque[float] = deque()

    def _add(self, value: float) -> None:
        while self.candidates and self.candidates[-1] < value:
            self.candidates.pop()
        self.candidates.append(value)

    def _remove(self, value: float) -> None:
        if self.candidates[0] == value:
            self.candidates.popleft()

    def result(self) -> float:
        return self.candidates[0]


class RollingMinimum(RollingWindow):
    def __init__(self, window: int) -> None:
        super().__init__(window)
        self.candidates: deque[float] = deque()

    def _add(self, value: float) -> None:
        while self.candidates and self.candidates[-1] > value:
            self.candidates.pop()
        self.candidates.append(value)

    def _remove(self, value: float) -> None:
        if self.candidates[0] == value:
            self.candidates.popleft()

    def result(self) -> float:
        return self.candidates[0]
