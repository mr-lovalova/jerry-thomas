from dataclasses import dataclass
from datetime import datetime


@dataclass
class SandboxOhlcvDTO:
    """Data Transfer Object (DTO) for sandbox OHLCV records."""

    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str
