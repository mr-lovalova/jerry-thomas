from typing import Literal

from .base import OperationProfile
from .output import ServeOutputConfig


class InspectProfile(OperationProfile):
    cmd: Literal["inspect"]
    output: ServeOutputConfig | None = None
