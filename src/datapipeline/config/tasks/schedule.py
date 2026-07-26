from typing import Annotated

from pydantic import Field, StringConstraints, field_validator

from .base import ArtifactTask


FieldName = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


class ScheduleTask(ArtifactTask):
    entrypoint: str = Field(default="core.artifact.schedule")
    stream: FieldName
    partition_by: list[FieldName]

    @field_validator("partition_by")
    @classmethod
    def _validate_partition_by(cls, partition_by: list[str]) -> list[str]:
        if len(partition_by) != len(set(partition_by)):
            raise ValueError("partition_by must not contain duplicate fields")
        if "time" in partition_by:
            raise ValueError("partition_by must not contain the reserved field 'time'")
        return partition_by
