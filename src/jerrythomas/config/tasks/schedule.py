from pydantic import Field, field_validator

from jerrythomas.config.constraints import NonEmptyString as FieldName
from .base import ArtifactTask


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
