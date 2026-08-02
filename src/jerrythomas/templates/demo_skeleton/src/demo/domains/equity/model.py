from dataclasses import dataclass

from jerrythomas.domain.record import TemporalRecord


@dataclass
class EquityRecord(TemporalRecord):
    """One equity OHLCV observation."""

    open: float
    high: float
    low: float
    close: float
    volume: float
    dollar_volume: float
    hl_range: float
    ticker: str
