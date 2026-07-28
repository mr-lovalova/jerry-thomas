from dataclasses import dataclass
from typing import Any


@dataclass
class Vector:
    values: dict[str, Any]

    def __len__(self) -> int:
        return len(self.values)

    def keys(self):
        return self.values.keys()

    def __getitem__(self, key: str) -> Any:
        return self.values[key]
