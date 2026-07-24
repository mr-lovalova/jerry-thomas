from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class SeriesRecord:
    id: str
    time: datetime
    value: Any
    entity_key: tuple = ()
    _establishes_domain: bool = field(kw_only=True, repr=False, compare=False)


@dataclass
class SeriesSequence:
    id: str
    time: datetime
    values: list[Any]
    entity_key: tuple = ()
    _establishes_domain: bool = field(kw_only=True, repr=False, compare=False)
