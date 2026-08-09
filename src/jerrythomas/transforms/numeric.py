class CompensatedSum:
    """Accumulate floating-point values with Neumaier compensation."""

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
