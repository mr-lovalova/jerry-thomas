from dataclasses import dataclass, field, fields
from datetime import datetime, timezone


@dataclass
class TemporalRecord:
    """Canonical UTC time-series payload used throughout the pipeline."""

    time: datetime
    _establishes_domain: bool = field(
        default=True,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.time.tzinfo is None or self.time.utcoffset() is None:
            raise ValueError("time must be timezone-aware")
        self.time = self.time.astimezone(timezone.utc)

    def _identity_fields(self) -> dict:
        """Return a mapping of domain fields excluding 'time'."""
        data = public_record_fields(self)
        data.pop("time", None)
        return data

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if not isinstance(other, TemporalRecord):
            return NotImplemented
        return (
            self.time == other.time
            and self._identity_fields() == other._identity_fields()
        )


def public_record_fields(record: TemporalRecord) -> dict[str, object]:
    """Return declared and dynamic public fields in stable declaration order."""
    values = {
        definition.name: getattr(record, definition.name)
        for definition in fields(record)
        if not definition.name.startswith("_")
    }
    values.update(
        (name, value)
        for name, value in vars(record).items()
        if not name.startswith("_")
    )
    return values
