from dataclasses import dataclass
from typing import Any, Literal, Optional

from .vector import Vector


WindowMode = Literal["union", "intersection", "strict"]


@dataclass
class Sample:
    """One dataset row with its identity, feature vector, and optional targets."""

    key: Any
    features: Vector
    targets: Optional[Vector] = None

    def with_targets(self, targets: Optional[Vector]) -> "Sample":
        return Sample(key=self.key, features=self.features, targets=targets)

    def with_features(self, features: Vector) -> "Sample":
        return Sample(key=self.key, features=features, targets=self.targets)
