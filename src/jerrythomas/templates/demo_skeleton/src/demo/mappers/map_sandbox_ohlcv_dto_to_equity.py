from collections.abc import Iterator

from demo.domains.equity.model import EquityRecord
from demo.dtos.sandbox_ohlcv_dto import SandboxOhlcvDTO


def map_sandbox_ohlcv_dto_to_equity(
    records: Iterator[SandboxOhlcvDTO],
) -> Iterator[EquityRecord]:
    """Map SandboxOhlcvDTO records to domain-level EquityRecord records."""
    for record in records:
        yield EquityRecord(
            time=record.time,
            open=record.open,
            high=record.high,
            low=record.low,
            close=record.close,
            volume=record.volume,
            dollar_volume=record.close * record.volume,
            hl_range=record.high - record.low,
            ticker=record.symbol,
        )
