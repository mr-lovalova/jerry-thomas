from typing import Literal


RecordPreviewStage = Literal["input", "canonical", "records"]

PreviewStage = Literal[
    "input",
    "canonical",
    "records",
    "series",
    "samples",
    "postprocess",
]

PREVIEW_STAGES: tuple[PreviewStage, ...] = (
    "input",
    "canonical",
    "records",
    "series",
    "samples",
    "postprocess",
)
