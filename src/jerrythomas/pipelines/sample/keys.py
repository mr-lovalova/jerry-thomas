from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from jerrythomas.artifacts.models import (
    SampleDomainEntry,
    SampleMetadata,
    Window,
)
from jerrythomas.utils.time import (
    floor_time_to_cadence,
    parse_cadence,
)


_SampleWindow = tuple[tuple[Any, ...], datetime, datetime]


def _is_fixed_offset(value: datetime) -> bool:
    timezone_info = value.tzinfo
    return timezone_info is not None and timezone_info.utcoffset(None) is not None


@dataclass(frozen=True)
class RectangularKeyPlan:
    start: datetime
    end: datetime
    step: timedelta
    windows: tuple[_SampleWindow, ...]
    _windows_by_key: dict[
        tuple[Any, ...],
        tuple[tuple[datetime, datetime], ...],
    ] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        windows_by_key: dict[
            tuple[Any, ...],
            list[tuple[datetime, datetime]],
        ] = {}
        for key_values, window_start, window_end in self.windows:
            windows_by_key.setdefault(key_values, []).append((window_start, window_end))
        object.__setattr__(
            self,
            "_windows_by_key",
            {
                key_values: tuple(windows)
                for key_values, windows in windows_by_key.items()
            },
        )

    def contains(self, key: tuple) -> bool:
        if not key or not isinstance(key[0], datetime):
            return False

        timestamp = key[0]
        origin_timezone = self.start.tzinfo
        if origin_timezone is None:
            if timestamp.tzinfo is not None:
                return False
            lattice_timestamp = timestamp
        elif timestamp.tzinfo is None:
            return False
        elif _is_fixed_offset(self.start):
            lattice_timestamp = timestamp
        else:
            lattice_timestamp = timestamp.astimezone(origin_timezone)

        index, remainder = divmod(
            lattice_timestamp - self.start,
            self.step,
        )
        if index < 0 or remainder:
            return False

        planned_timestamp = self.start + index * self.step
        if planned_timestamp != lattice_timestamp or planned_timestamp > self.end:
            return False

        key_values = key[1:]
        return any(
            window_start <= planned_timestamp <= window_end
            for window_start, window_end in self._windows_by_key.get(key_values, ())
        )

    @property
    def total(self) -> int | None:
        if not self.windows:
            return 0
        if not self._has_consistent_time_lattice():
            return None

        last_plan_index = (self.end - self.start) // self.step
        total = 0
        for _, window_start, window_end in self.windows:
            first_index, remainder = divmod(window_start - self.start, self.step)
            if remainder:
                first_index += 1
            first_index = max(0, first_index)
            last_index = min(
                last_plan_index,
                (window_end - self.start) // self.step,
            )
            total += max(0, last_index - first_index + 1)
        return total

    def keys(self) -> Iterator[tuple]:
        current = self.start
        while current <= self.end:
            for key_values, window_start, window_end in self.windows:
                if window_start <= current <= window_end:
                    yield (current, *key_values)
            current += self.step

    def _has_consistent_time_lattice(self) -> bool:
        origin_timezone = self.start.tzinfo
        # Fixed-offset origins advance in absolute time. Dynamic zones advance in
        # wall time, so their bounds must share the origin's timezone object.
        if _is_fixed_offset(self.start):
            return self.end.tzinfo is not None and all(
                window_start.tzinfo is not None and window_end.tzinfo is not None
                for _, window_start, window_end in self.windows
            )
        return self.end.tzinfo is origin_timezone and all(
            window_start.tzinfo is origin_timezone
            and window_end.tzinfo is origin_timezone
            for _, window_start, window_end in self.windows
        )


def metadata_key_plan(
    window: Window | None,
    sample: SampleMetadata | None,
    cadence: str,
    sample_keys: Sequence[str],
) -> RectangularKeyPlan | None:
    if window is None:
        return None
    if not sample_keys:
        return window_key_plan(window.start, window.end, cadence)
    if sample is None:
        raise RuntimeError("Metadata has no sample domain for configured sample.keys.")
    if sample.cadence != cadence or sample.keys != list(sample_keys):
        raise RuntimeError("Metadata sample configuration does not match the dataset.")
    return sample_domain_key_plan(
        window.start,
        window.end,
        cadence,
        sample_keys,
        sample.domain,
    )


def require_metadata_key_plan(
    window: Window | None,
    sample: SampleMetadata | None,
    cadence: str,
    sample_keys: Sequence[str],
) -> RectangularKeyPlan:
    plan = metadata_key_plan(window, sample, cadence, sample_keys)
    if plan is None:
        raise RuntimeError(
            "Metadata has no rectangular sample window. Rebuild build/metadata.json."
        )
    return plan


def merge_rectangular_key_plans(
    plans: Sequence[RectangularKeyPlan],
) -> RectangularKeyPlan:
    if not plans:
        raise ValueError("Cannot merge an empty sequence of rectangular key plans.")

    reference = plans[0]
    if not reference._has_consistent_time_lattice():
        raise ValueError(
            "Cannot merge rectangular key plans with inconsistent time lattices."
        )

    reference_timezone = reference.start.tzinfo
    for plan in plans:
        if plan.step != reference.step:
            raise ValueError(
                "Cannot merge rectangular key plans with different cadence steps."
            )
        if not plan._has_consistent_time_lattice():
            raise ValueError(
                "Cannot merge rectangular key plans with inconsistent time lattices."
            )

        plan_timezone = plan.start.tzinfo
        if _is_fixed_offset(reference.start):
            compatible_timezone = _is_fixed_offset(plan.start)
        else:
            compatible_timezone = plan_timezone is reference_timezone
        if not compatible_timezone:
            raise ValueError(
                "Cannot merge rectangular key plans with incompatible time lattices."
            )

        _, remainder = divmod(plan.start - reference.start, reference.step)
        if remainder:
            raise ValueError(
                "Cannot merge rectangular key plans with incompatible time lattices."
            )

    start = min(plan.start for plan in plans)
    end = max(plan.end for plan in plans)
    intervals_by_key: dict[tuple[Any, ...], list[tuple[int, int]]] = {}
    for plan in plans:
        for key_values, window_start, window_end in plan.windows:
            clipped_start = max(plan.start, window_start)
            clipped_end = min(plan.end, window_end)
            if clipped_start > clipped_end:
                continue

            first_index, remainder = divmod(clipped_start - start, reference.step)
            if remainder:
                first_index += 1
            last_index = (clipped_end - start) // reference.step
            if first_index <= last_index:
                intervals_by_key.setdefault(key_values, []).append(
                    (first_index, last_index)
                )

    windows: list[_SampleWindow] = []
    for key_values in sorted(intervals_by_key):
        intervals = sorted(intervals_by_key[key_values])
        merged_start, merged_end = intervals[0]
        for interval_start, interval_end in intervals[1:]:
            if interval_start <= merged_end + 1:
                merged_end = max(merged_end, interval_end)
                continue
            windows.append(
                (
                    key_values,
                    start + merged_start * reference.step,
                    start + merged_end * reference.step,
                )
            )
            merged_start, merged_end = interval_start, interval_end
        windows.append(
            (
                key_values,
                start + merged_start * reference.step,
                start + merged_end * reference.step,
            )
        )

    return RectangularKeyPlan(
        start=start,
        end=end,
        step=reference.step,
        windows=tuple(windows),
    )


def window_key_plan(
    start: datetime | None,
    end: datetime | None,
    cadence: str | None,
) -> RectangularKeyPlan | None:
    if start is None or end is None or cadence is None:
        return None
    step = parse_cadence(cadence)
    window_start = floor_time_to_cadence(start, step)
    window_end = floor_time_to_cadence(end, step)
    return RectangularKeyPlan(
        start=window_start,
        end=window_end,
        step=step,
        windows=(((), window_start, window_end),),
    )


def sample_domain_key_plan(
    start: datetime | None,
    end: datetime | None,
    cadence: str,
    sample_keys: Sequence[str],
    domain: Sequence[SampleDomainEntry],
) -> RectangularKeyPlan | None:
    plan = window_key_plan(start, end, cadence)
    if plan is None:
        return None
    if not sample_keys:
        return plan

    prepared: list[_SampleWindow] = []
    for entry in domain:
        if len(entry.key) != len(sample_keys):
            raise ValueError(
                "Vector metadata sample-domain key length does not match sample.keys."
            )
        domain_start = max(
            plan.start,
            floor_time_to_cadence(entry.start, plan.step),
        )
        domain_end = min(
            plan.end,
            floor_time_to_cadence(entry.end, plan.step),
        )
        if domain_start <= domain_end:
            prepared.append((tuple(entry.key), domain_start, domain_end))
    prepared.sort(key=lambda item: item[0])
    return RectangularKeyPlan(
        start=plan.start,
        end=plan.end,
        step=plan.step,
        windows=tuple(prepared),
    )
