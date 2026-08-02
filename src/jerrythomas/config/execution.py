from pydantic import BaseModel, ConfigDict, Field, StrictInt


class ExecutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sort_buffer_mb: StrictInt = Field(default=128, ge=1)

    @property
    def sort_buffer_bytes(self) -> int:
        return self.sort_buffer_mb * 1024 * 1024
