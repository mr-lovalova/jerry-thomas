from dataclasses import dataclass

from jerrythomas.config.profiles.build import ARTIFACT_MODES, ArtifactMode
from jerrythomas.execution.settings import ObservabilitySettings


@dataclass(frozen=True)
class BuildSettings:
    mode: ArtifactMode
    observability: ObservabilitySettings

    def __post_init__(self) -> None:
        if self.mode not in ARTIFACT_MODES:
            raise ValueError(f"Unknown artifact mode {self.mode!r}.")
