"""UTC boundaries for elapsed-time and local calendar periods."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from jerrythomas.utils.time import floor_time_to_cadence


@dataclass(frozen=True)
class FixedPeriod:
    step: timedelta

    def floor(self, time: datetime) -> datetime:
        return floor_time_to_cadence(time, self.step).astimezone(timezone.utc)

    def next(self, start: datetime) -> datetime:
        return start + self.step


@dataclass(frozen=True)
class CalendarMonth:
    zone: ZoneInfo

    def floor(self, time: datetime) -> datetime:
        local = time.astimezone(self.zone)
        return self._start(local.year, local.month)

    def next(self, start: datetime) -> datetime:
        local = start.astimezone(self.zone)
        year, month = divmod(local.year * 12 + local.month, 12)
        return self._start(year, month + 1)

    def _start(self, year: int, month: int) -> datetime:
        # fold=0 selects the first midnight if ambiguous. Conversion to UTC
        # advances a nonexistent midnight across the local clock gap.
        return datetime(year, month, 1, tzinfo=self.zone).astimezone(timezone.utc)
